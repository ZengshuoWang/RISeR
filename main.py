import copy
import os
from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
from torch import nn
from copy import deepcopy
from typing import List, Tuple
import glob
import numpy as np
import math
import time
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt


# ------------------------
# SuperPoint提取图像的关键点
# ------------------------
def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    """
        SuperPoint Convolutional Detector and Descriptor

        SuperPoint: Self-Supervised Interest Point Detection and Description.
        Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629
    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent / 'models/weights/superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path)))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }


# -----------------------------
# SuperGlue进行图像之间关键点的匹配
# -----------------------------
def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability """
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability """
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1


class SuperGlue(nn.Module):
    """
        SuperGlue feature matching middle-end

        Given two sets of keypoints and locations, we determine the
        correspondences by:
            1. Keypoint Encoding (normalization + visual feature and location fusion)
            2. Graph Neural Network with multiple self and cross-attention layers
            3. Final projection layer
            4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
            5. Thresholding matrix based on mutual exclusivity and a match_threshold

        The correspondence ids use -1 to indicate non-matching points.

        Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
            Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
            Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763
    """
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'], layer_names=self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        assert self.config['weights'] in ['indoor', 'outdoor']
        path = Path(__file__).parent
        path = path / 'models/weights/superglue_{}.pth'.format(self.config['weights'])
        self.load_state_dict(torch.load(str(path)))
        print('Loaded SuperGlue model (\"{}\" weights)'.format(
            self.config['weights']))

    def forward(self, data):
        """ Run SuperGlue on a pair of keypoints and descriptors """
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        # Keypoint's normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return {
            'matches0': indices0,
            'matches1': indices1,
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }


# --------------------------------------------------
# 调用SuperPoint和SuperGlue进行图像对之间关键点的提取和匹配
# --------------------------------------------------
class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))

    def forward(self, data):
        """
            Run SuperPoint (optionally) and SuperGlue
            SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
            Args:
                data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**pred, **self.superglue(data)}

        return pred


# -------------
# 生成图像的mask
# -------------
def generate_mask(img, bw_thresh=9):
    """
        img --->>> 灰度图
        bw_thresh --->>> 二值化的阈值，int类型，这里建议取30
    """
    _, img_bw = cv2.threshold(img, bw_thresh, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    max_idx = np.argmax(area)

    for j in range(len(contours)):
        if j != max_idx:
            cv2.fillPoly(img_bw, [contours[j]], 0)

    return img_bw


# ------------------------
# 去除视网膜区域边缘附近的关键点
# ------------------------
def clean_key_points_near_edges(kp, sc, desc, mask, pixels=5):
    """
        kp --->>> 检测到的关键点（x, y），其中x, y为坐标
        sc --->>> 检测其为关键点的置信度值c
        desc --->>> 关键点对应的描述符
        mask --->>> 对应的mask图像
    """
    kp_temp = kp
    sc_temp = sc
    desc_temp = desc

    t = 0

    for i in range(kp.shape[0]):
        u = int(round(kp[i][0]))
        v = int(round(kp[i][1]))
        if (mask[v-pixels][u] < 1) or (mask[v+pixels][u] < 1) or (mask[v][u-pixels] < 1) or (mask[v][u+pixels] < 1):
            kp_temp = np.delete(kp_temp, i-t, axis=0)
            sc_temp = np.delete(sc_temp, i-t)
            desc_temp = np.delete(desc_temp, i-t, axis=1)
            t = t + 1

    return kp_temp, sc_temp, desc_temp


# ---------------------------------
# 将图像转化为像素值为0~1之间的tensor数据
# ---------------------------------
def frame2tensor(frame, device):

    return torch.from_numpy(frame/255.).float()[None, None].to(device)


# --------------------------
# 可视化输出关键点提取与匹配的结果
# --------------------------
def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10, opencv_display=False, opencv_title='', small_text=[]):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    out_match_kps = 255 * np.ones((H, W), np.uint8)
    out_match_kps[:H0, :W0] = image0
    out_match_kps[:H1, W0 + margin:] = image1
    out_match_kps = np.stack([out_match_kps] * 3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

        cv2.circle(out_match_kps, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out_match_kps, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(24 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out, out_match_kps


# ---------------
# 生成相机的内参矩阵
# ---------------
def get_IntrinsicCameraMatrix(r, u0, v0, location_camera=57.7, k=45.0, rou=12.0):
    """
        r表示图像中的圆形视网膜区域的半径，由图像的大小决定，单位是像素
        u0表示图像的中心在水平方向的坐标，单位是像素，可以理解为图像水平方向尺寸的一半
        v0表示图像的中心在竖直方向的坐标，单位是像素，可以理解为图像竖直方向尺寸的一半
        location_camera表示相机到世界坐标系原点（也就是眼球中心）的距离，单位是毫米mm
        k表示相机视野，单位是度，由相机规格而定
        rou表示初始化阶段球体眼睛模型的半径，单位是毫米mm
    """
    IntrinsicCameraMatrix = np.zeros((3, 3), dtype=float)
    IntrinsicCameraMatrix[0][0] = r*(location_camera+rou*np.cos(k*np.pi/(180*2)))/(rou*np.sin(k*np.pi/(180*2)))
    IntrinsicCameraMatrix[1][1] = IntrinsicCameraMatrix[0][0]
    IntrinsicCameraMatrix[2][2] = 1.0
    IntrinsicCameraMatrix[0][2] = u0
    IntrinsicCameraMatrix[1][2] = v0

    return IntrinsicCameraMatrix


# ---------------------------------------------------------------------
# 射线椭球相交，即建立由2d图像点到3d椭球上的视网膜点的映射模型（加入了四阶径向畸变模型）
# ---------------------------------------------------------------------
def map_2dTo3d(points_2d, K, rvec_camera, tvec_camera, location_camera, rvec_eye, a, b, c, k1, k2):
    """
        points_2d表示视网膜图像中的所有2D坐标点构成的矩阵，每一个2D点用行向量的形式表示，所以其尺寸为 n 行 2 列（必须是float类型的）
        K表示相机的内参矩阵
        rvec_camera表示相机位姿的旋转向量，以弧度为单位，是一个1*3的行向量或3*1的列向量
        tvec_camera表示相机位姿的平移向量，是一个3*1的列向量
        location_camera表示相机在世界坐标系中的位置坐标，是一个3*1的列向量
        rvec_eye表示椭球眼睛模型的旋转向量，即分别绕三个轴旋转的角度，以弧度为单位，是一个1*3的行向量或3*1的列向量，并且每个元素必须是float数据
        a、b、c分别为椭球的半轴长度
        k1、k2是四阶径向畸变模型的两个参数
    """
    points_2d = np.array([points_2d])
    points_2d_undistorted = cv2.undistortPoints(points_2d, K, np.array([k1, k2, 0, 0]), P=K)

    points_2d_undistorted_qici = np.r_[points_2d_undistorted[0].T, np.ones((1, points_2d_undistorted[0].shape[0]))]

    R, _ = cv2.Rodrigues(rvec_camera)
    Rt = np.c_[R, tvec_camera]

    P = np.dot(K, Rt)

    P_jia = np.dot(P.T, np.linalg.inv(np.dot(P, P.T)))

    X1 = np.dot(P_jia, points_2d_undistorted_qici)

    c11 = location_camera[0][0]
    c12 = location_camera[1][0]
    c13 = location_camera[2][0]

    Q, _ = cv2.Rodrigues(rvec_eye)
    A = np.zeros((3, 3), dtype=float)
    A[0][0] = a ** (-2)
    A[1][1] = b ** (-2)
    A[2][2] = c ** (-2)
    QAQ = np.dot(np.dot(Q.T, A), Q)
    y11 = QAQ[0][0]
    y12 = QAQ[0][1]
    y13 = QAQ[0][2]
    y21 = QAQ[1][0]
    y22 = QAQ[1][1]
    y23 = QAQ[1][2]
    y31 = QAQ[2][0]
    y32 = QAQ[2][1]
    y33 = QAQ[2][2]

    aa = (y11 * (c11 ** 2) + y21 * c11 * c12 + y12 * c11 * c12 + y31 * c11 * c13 + y13 * c11 * c13 +
          y22 * (c12 ** 2) + y32 * c12 * c13 + y23 * c12 * c13 + y33 * (c13 ** 2) - 1)
    bb = (2 * y11 * X1[0] * c11 + y21 * X1[0] * c12 + y21 * c11 * X1[1] + y12 * X1[0] * c12 +
          y12 * c11 * X1[1] + y31 * X1[0] * c13 + y31 * c11 * X1[2] + y13 * X1[0] * c13 +
          y13 * c11 * X1[2] + 2 * y22 * X1[1] * c12 + y32 * X1[1] * c13 + y32 * c12 * X1[2] +
          y23 * X1[1] * c13 + y23 * c12 * X1[2] + 2 * y33 * X1[2] * c13 - 2 * X1[3])
    cc = (y11 * (X1[0] * X1[0]) + y21 * X1[0] * X1[1] + y12 * X1[0] * X1[1] + y31 * X1[0] * X1[2] +
          y13 * X1[0] * X1[2] + y22 * (X1[1] * X1[1]) + y32 * X1[1] * X1[2] + y23 * X1[1] * X1[2] +
          y33 * (X1[2] * X1[2]) - X1[3] * X1[3])

    delt = bb * bb - 4 * aa * cc
    delt_jia = np.maximum(delt, 0)

    lam = (-bb - np.sqrt(delt_jia))/(2*aa)
    points_3d_x = (X1[0] + c11*lam)/(X1[3] + lam)
    points_3d_y = (X1[1] + c12*lam)/(X1[3] + lam)
    points_3d_z = (X1[2] + c13*lam)/(X1[3] + lam)

    points_3d_z[delt_jia == 0] = 0

    points_3d = np.r_[np.r_[[points_3d_x], [points_3d_y]], [points_3d_z]].T

    return points_3d


# -----------------------------------------------------
# 建立由3D空间点到2D像素点的映射模型（加入了四阶径向畸变模型）
# -----------------------------------------------------
def map_3dTo2d(points_3d, K, rvec_camera, tvec_camera, k1, k2):
    """
        points_3d表示眼球视网膜上的世界坐标系下的所有3D坐标点坐标构成的矩阵，
            每一个3D点坐标都是用行向量的形式表示，因此points_3d构成了 n 行 3 列大小的矩阵
        K表示相机的内参矩阵
        rvec_camera表示相机位姿的旋转向量，以弧度为单位，是一个1*3的行向量或3*1的列向量
        tvec_camera表示相机位姿的平移向量，是一个3*1的列向量
        k1、k2表示四阶径向畸变模型中的两个参数
    """
    points_3d_qici = np.r_[points_3d.T, np.ones((1, points_3d.shape[0]))]

    R_camera, _ = cv2.Rodrigues(rvec_camera)
    Rt_camera_qici = np.r_[np.c_[R_camera, tvec_camera], [[0, 0, 0, 1]]]

    P_C = np.dot(Rt_camera_qici, points_3d_qici)

    P_C = np.delete(P_C, -1, axis=0)
    P_C_1 = P_C * (1/P_C[2])

    r_2 = P_C_1[0] * P_C_1[0] + P_C_1[1] * P_C_1[1]
    distorted = 1 + k1 * r_2 + k2 * (r_2*r_2)
    P_CJ_1 = P_C_1 * distorted
    P_CJ_1 = np.delete(P_CJ_1, -1, axis=0)
    P_CJ_1 = np.r_[P_CJ_1, np.ones((1, P_CJ_1.shape[1]))]

    P_uv = np.dot(K, P_CJ_1)

    P_uv = np.delete(P_uv, -1, axis=0)
    points_2d = P_uv.T

    return points_2d


# -------------------------------------------------------------------
# 该函数的功能是将参考眼底曲面上的3D点映射到测试眼底曲面上
# -------------------------------------------------------------------
def pts_3D_from_reference_to_test(pts_3D_reference, a_Eye, b_Eye, c_Eye_test, Rvec_Eye):
    """
        pts_3D_reference表示参考眼底曲面上的所有3D点坐标
        a_Eye表示眼球的a半轴长度
        b_Eye表示眼球的b半轴长度
        c_Eye_test表示测试眼球的c半轴长度
        Rvec_Eye表示眼球的姿态参数
    """
    vertex_rotate = np.array([0, 0, 0]).reshape((-1, 1))
    x1 = vertex_rotate[0][0]
    y1 = vertex_rotate[1][0]
    z1 = vertex_rotate[2][0]

    Q, _ = cv2.Rodrigues(Rvec_Eye)
    A = np.zeros((3, 3), dtype=float)
    A[0][0] = a_Eye ** (-2)
    A[1][1] = b_Eye ** (-2)
    A[2][2] = c_Eye_test ** (-2)
    QAQ = np.dot(np.dot(Q.T, A), Q)

    mnp = pts_3D_reference - vertex_rotate.T
    m = mnp[:, 0]
    n = mnp[:, 1]
    p = mnp[:, 2]

    t_a = (QAQ[0][0] * (m ** 2) + QAQ[1][1] * (n ** 2) + QAQ[2][2] * (p ** 2) + (QAQ[0][1] + QAQ[1][0]) * m * n +
           (QAQ[1][2] + QAQ[2][1]) * n * p + (QAQ[0][2] + QAQ[2][0]) * p * m)
    t_b = (QAQ[0][0] * 2 * x1 * m + QAQ[1][1] * 2 * y1 * n + QAQ[2][2] * 2 * z1 * p +
           (QAQ[0][1] + QAQ[1][0]) * (y1 * m + x1 * n) + (QAQ[1][2] + QAQ[2][1]) * (z1 * n + y1 * p) +
           (QAQ[0][2] + QAQ[2][0]) * (x1 * p + z1 * m))
    t_c = (QAQ[0][0] * (x1 ** 2) + QAQ[1][1] * (y1 ** 2) + QAQ[2][2] * (z1 ** 2) + (QAQ[0][1] + QAQ[1][0]) * x1 * y1 +
           (QAQ[1][2] + QAQ[2][1]) * y1 * z1 + (QAQ[0][2] + QAQ[2][0]) * z1 * x1 - 1)

    delt = t_b * t_b - 4 * t_a * t_c

    t = (-t_b + np.sqrt(delt)) / (2 * t_a)

    pts_3D_Test_x = m * t + x1
    pts_3D_Test_y = n * t + y1
    pts_3D_Test_z = p * t + z1
    assert pts_3D_Test_z.min() > 0
    pts_3D_Test = np.r_[np.r_[[pts_3D_Test_x], [pts_3D_Test_y]], [pts_3D_Test_z]].T

    return pts_3D_Test


# ---------------------------------------------------------------------------
# 该函数的功能是根据最优化得到的参数完成测试图像的配准任务（以参考图像为基准），并输出配准结果
# ---------------------------------------------------------------------------
def test_image_registration_result(reference_image, test_image, K_refer, K_test, rvec_test_camera, tvec_test_camera,
                                   rvec_eye, a_reference, a_test, b_reference, b_test, c_reference, c_test,
                                   rvec_ref_camera, tvec_ref_camera, location_ref_camera,
                                   k1_reference, k2_reference, k1_test, k2_test, path_out):
    """
        reference_image表示mask了的参考图像，颜色顺序是RGB，主要用来设置test_image_registration图像的尺寸，二者保持一致
        test_image表示待配准的mask了的测试图像，颜色顺序为RGB，虽然用cv2进行读取的为BGR，但读取后进行了转换，转换成了RGB
        K_refer表示参考相机的内参矩阵
        K_test表示测试相机的内参矩阵
        rvec_test_camera表示测试相机位姿的旋转向量，以弧度为单位，是一个1*3的行向量或3*1的列向量
        tvec_test_camera表示测试相机位姿的平移向量，是一个3*1的列向量
        rvec_eye表示椭球眼睛模型的旋转向量，即分别绕三个轴旋转的角度，以弧度为单位，是一个1*3的行向量或3*1的列向量，并且每个元素必须是float数据
        a、b、c_reference、c_test分别为椭球的半轴长度
        rvec_ref_camera表示参考相机位姿的旋转向量，以弧度为单位，是一个1*3的行向量或3*1的列向量
        tvec_ref_camera表示参考相机位姿的平移向量，是一个3*1的列向量
        location_ref_camera表示参考相机在世界坐标系中的位置坐标，是一个3*1的列向量
        k1_reference、k2_reference是参考相机四阶径向畸变模型中的两个参数
        k1_test、k2_test是参考相机四阶径向畸变模型中的两个参数
    """
    v_reference = np.shape(reference_image)[0]
    u_reference = np.shape(reference_image)[1]
    test_image_registration = np.zeros([v_reference, u_reference, 3], np.uint8)
    v_test = np.shape(test_image)[0]
    u_test = np.shape(test_image)[1]

    points_2d_registration = np.c_[np.array([np.full(v_reference, 0, dtype=float)]).T,
                                   np.array([np.arange(0, v_reference, dtype=float)]).T]
    for i in range(1, u_reference):
        points_2d_registration = np.r_[points_2d_registration,
                                       np.c_[np.array([np.full(v_reference, i, dtype=float)]).T,
                                             np.array([np.arange(0, v_reference, dtype=float)]).T]]

    points_3d_Reference = map_2dTo3d(points_2d_registration, K_refer, rvec_ref_camera, tvec_ref_camera,
                                     location_ref_camera, rvec_eye, a_reference, b_reference, c_reference,
                                     k1_reference, k2_reference)

    points_3d_Test = pts_3D_from_reference_to_test(points_3d_Reference, a_test, b_test, c_test, rvec_eye)

    points_2d_test_float = map_3dTo2d(points_3d_Test, K_test, rvec_test_camera, tvec_test_camera, k1_test, k2_test)

    points1 = np.floor(points_2d_test_float).astype(np.int_)
    points2 = np.c_[np.array([points1[:, 0]]).T, np.array([points1[:, 1]+1]).T]
    points3 = np.c_[np.array([points1[:, 0]+1]).T, np.array([points1[:, 1]]).T]
    points4 = np.c_[np.array([points1[:, 0]+1]).T, np.array([points1[:, 1]+1]).T]
    delt_u = points_2d_test_float[:, 0]-points1[:, 0]
    delt_v = points_2d_test_float[:, 1]-points1[:, 1]

    for j in range(points_2d_registration.shape[0]):
        if (points1[j][0] < 0) or (points1[j][1] < 0):
            test_image_registration[int(points_2d_registration[j][1]), int(points_2d_registration[j][0]), :] = 0
        elif (points1[j][0] > u_test-2) or (points1[j][1] > v_test-2):
            test_image_registration[int(points_2d_registration[j][1]), int(points_2d_registration[j][0]), :] = 0
        else:
            test_image_registration[int(points_2d_registration[j][1]), int(points_2d_registration[j][0]), :] = \
                (1-delt_u[j])*(1-delt_v[j])*test_image[points1[j][1], points1[j][0], :] + \
                (1-delt_u[j])*delt_v[j]*test_image[points2[j][1], points2[j][0], :] + \
                (1-delt_v[j])*delt_u[j]*test_image[points3[j][1], points3[j][0], :] + \
                delt_u[j]*delt_v[j]*test_image[points4[j][1], points4[j][0], :]

    test_image_registration_BGR = cv2.cvtColor(test_image_registration, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path_out, test_image_registration_BGR)

    return


# ------------------------------------------
# 定义适应度函数，首次配准，两幅图像，优化所有参数
# ------------------------------------------
def fit_fun_first2all(S, points_2d_reference, points_2d_test, K_refer, K_test):
    """
        S表示待优化的参数，由19个参数组成
            （1）测试相机的位姿
                旋转向量R的三个元素——r_theta, r_fai, r_omiga
                平移向量t的三个元素——t_x, t_y, t_z
            （2）眼球模型
                椭球模型的三个半轴长度——a_reference_eye, a_test_eye, b_reference_eye, b_test_eye, c_reference_eye, c_test_eye
                椭球模型旋转向量Q的三个元素——r_a, r_b, r_c
            （3）相机的四阶径向畸变模型参数k1_reference, k2_reference, k1_test, k2_test
        points_2d_reference表示参考图像上的所有特征点坐标构成的矩阵，是N行2列的大小
        points_2d_test表示测试图像上的所有特征点坐标构成的矩阵，是N行2列的大小
        K是相机的内参矩阵
    """
    Rv_camera_reference = np.array([0.0, 0.0, 0.0]).reshape((1, -1))
    t_camera_reference = np.array([0.0, 0.0, 57.7]).reshape((-1, 1))
    located_camera_reference = np.array([0.0, 0.0, -57.7]).reshape((-1, 1))
    fitness = np.zeros(S.shape[0])
    for num in range(0, S.shape[0]):
        Rv_camera_test = np.array([S[num, 0], S[num, 1], S[num, 2]]).reshape((1, -1))
        t_camera_test = np.array([S[num, 3], S[num, 4], S[num, 5]]).reshape((-1, 1))
        Rm_camera_test, _ = cv2.Rodrigues(Rv_camera_test)
        located_camera_test = -np.dot(np.linalg.inv(Rm_camera_test), t_camera_test)
        a_reference_eye = S[num, 6]
        a_test_eye = S[num, 7]
        b_reference_eye = S[num, 8]
        b_test_eye = S[num, 9]
        c_reference_eye = S[num, 10]
        c_test_eye = S[num, 11]
        Rv_eye = np.array([S[num, 12], S[num, 13], S[num, 14]]).reshape((1, -1))
        k1_camera_reference = S[num, 15]
        k2_camera_reference = S[num, 16]
        k1_camera_test = S[num, 17]
        k2_camera_test = S[num, 18]
        point_num = points_2d_reference.shape[0]
        points_3d_reference = map_2dTo3d(points_2d_reference, K_refer, Rv_camera_reference, t_camera_reference,
                                         located_camera_reference, Rv_eye,
                                         a_reference_eye, b_reference_eye, c_reference_eye,
                                         k1_camera_reference, k2_camera_reference)
        points_3d_test = map_2dTo3d(points_2d_test, K_test, Rv_camera_test, t_camera_test, located_camera_test,
                                    Rv_eye, a_test_eye, b_test_eye, c_test_eye, k1_camera_test, k2_camera_test)

        points_3d_reference_test = pts_3D_from_reference_to_test(points_3d_reference,
                                                                 a_test_eye, b_test_eye, c_test_eye, Rv_eye)

        distance_3d = np.sqrt(np.square(points_3d_test[:, 0] - points_3d_reference_test[:, 0]) +
                              np.square(points_3d_test[:, 1] - points_3d_reference_test[:, 1]) +
                              np.square(points_3d_test[:, 2] - points_3d_reference_test[:, 2]))
        distance_3d[points_3d_reference[:, 2] < 1] = 50
        distance_3d[points_3d_test[:, 2] < 1] = 50
        distance_3d.sort()
        if int(distance_3d[-1] + 0.5) == 50:
            fitness[num] = np.sum(distance_3d)
        else:
            fitness[num] = np.sum(distance_3d[:math.floor(point_num * 0.8)])

    return fitness


# --------------------------------------------------------
# 定义适应度函数，后续配准，三幅图像，仅优化测试相机和测试眼球对应的参数
# --------------------------------------------------------
def fit_fun_follow3onlytest(S, points_2d_reference_1, points_2d_reference_2, points_2d_test_1, points_2d_test_2,
                            K_refer_1, K_refer_2, K_test, S_reference_1, S_reference_2, Rv_eye,
                            kps_match_number_refer1_test, kps_match_number_refer2_test):
    """
        S表示待优化的参数，由11个参数组成
            (1) 测试相机的位姿
                旋转向量R的三个元素——r_theta, r_fai, r_omiga
                平移向量t的三个元素——t_x, t_y, t_z
            (2) 测试眼球的形状
                椭球模型的三个半轴长度——a_test_eye, b_test_eye, c_test_eye
            (3) 测试相机的四阶径向畸变模型参数——k1_test, k2_test
        points_2d_reference_1表示参考图像1上的所有特征点坐标构成的矩阵，是N行2列的大小
        points_2d_reference_2表示参考图像2上的所有特征点坐标构成的矩阵，是N行2列的大小
        points_2d_test_1和points_2d_test_2分别是测试图像和参考图像1、参考图像2的匹配特征点坐标构成的矩阵，每个的大小都是N行2列
        K_refer_1表示参考相机1的内参矩阵
        K_refer_2表示参考相机2的内参矩阵
        K_test表示测试相机的内参矩阵
        S_reference_1表示参考1的相关参数，由11个参数组成
            (1) 参考相机1的位姿
                旋转向量R的三个元素——r_theta, r_fai, r_omiga
                平移向量t的三个元素——t_x, t_y, t_z
            (2) 参考眼球1的形状
                椭球模型的三个半轴长度——a_reference1_eye, b_reference1_eye, c_reference1_eye
            (3) 参考相机1的四阶径向畸变模型参数——k1_reference1, k2_reference1
        S_reference_2表示参考2的相关参数，由11个参数组成
            (1) 参考相机2的位姿
                旋转向量R的三个元素——r_theta, r_fai, r_omiga
                平移向量t的三个元素——t_x, t_y, t_z
            (2) 参考眼球2的形状
                椭球模型的三个半轴长度——a_reference2_eye, b_reference2_eye, c_reference2_eye
            (3) 参考相机2的四阶径向畸变模型参数——k1_reference2, k2_reference2
        Rv_eye表示眼球模型的姿态，是1行3列的大小
        kps_match_number_refer1_test表示参考图像1和测试图像之间匹配特征点的数量
        kps_match_number_refer2_test表示参考图像2和测试图像之间匹配特征点的数量
    """
    assert points_2d_reference_1.shape[0] == kps_match_number_refer1_test
    assert points_2d_reference_2.shape[0] == kps_match_number_refer2_test
    weight_refer1_test = max([kps_match_number_refer1_test,
                              kps_match_number_refer2_test]) / kps_match_number_refer1_test
    weight_refer2_test = max([kps_match_number_refer1_test,
                              kps_match_number_refer2_test]) / kps_match_number_refer2_test
    assert min([weight_refer1_test, weight_refer2_test]) == 1.0

    Rv_camera_reference_1 = np.array([S_reference_1[0], S_reference_1[1], S_reference_1[2]]).reshape((1, -1))
    t_camera_reference_1 = np.array([S_reference_1[3], S_reference_1[4], S_reference_1[5]]).reshape((-1, 1))
    Rm_camera_reference_1, _ = cv2.Rodrigues(Rv_camera_reference_1)
    located_camera_reference_1 = -np.dot(np.linalg.inv(Rm_camera_reference_1), t_camera_reference_1)
    assert located_camera_reference_1[2][0] == -57.7
    a_reference1_eye = S_reference_1[6]
    b_reference1_eye = S_reference_1[7]
    c_reference1_eye = S_reference_1[8]
    k1_camera_reference1 = S_reference_1[9]
    k2_camera_reference1 = S_reference_1[10]

    Rv_camera_reference_2 = np.array([S_reference_2[0], S_reference_2[1], S_reference_2[2]]).reshape((1, -1))
    t_camera_reference_2 = np.array([S_reference_2[3], S_reference_2[4], S_reference_2[5]]).reshape((-1, 1))
    Rm_camera_reference_2, _ = cv2.Rodrigues(Rv_camera_reference_2)
    located_camera_reference_2 = -np.dot(np.linalg.inv(Rm_camera_reference_2), t_camera_reference_2)
    a_reference2_eye = S_reference_2[6]
    b_reference2_eye = S_reference_2[7]
    c_reference2_eye = S_reference_2[8]
    k1_camera_reference2 = S_reference_2[9]
    k2_camera_reference2 = S_reference_2[10]

    fitness = np.zeros(S.shape[0])
    for num in range(0, S.shape[0]):
        Rv_camera_test = np.array([S[num, 0], S[num, 1], S[num, 2]]).reshape((1, -1))
        t_camera_test = np.array([S[num, 3], S[num, 4], S[num, 5]]).reshape((-1, 1))
        Rm_camera_test, _ = cv2.Rodrigues(Rv_camera_test)
        located_camera_test = -np.dot(np.linalg.inv(Rm_camera_test), t_camera_test)
        a_test_eye = S[num, 6]
        b_test_eye = S[num, 7]
        c_test_eye = S[num, 8]
        k1_camera_test = S[num, 9]
        k2_camera_test = S[num, 10]

        point_num_1 = points_2d_reference_1.shape[0]
        points_3d_reference_1 = map_2dTo3d(points_2d_reference_1, K_refer_1, Rv_camera_reference_1,
                                           t_camera_reference_1, located_camera_reference_1, Rv_eye,
                                           a_reference1_eye, b_reference1_eye, c_reference1_eye,
                                           k1_camera_reference1, k2_camera_reference1)
        points_3d_test_1 = map_2dTo3d(points_2d_test_1, K_test, Rv_camera_test, t_camera_test,
                                      located_camera_test, Rv_eye, a_test_eye, b_test_eye, c_test_eye,
                                      k1_camera_test, k2_camera_test)
        points_3d_reference1_test = pts_3D_from_reference_to_test(points_3d_reference_1,
                                                                  a_test_eye, b_test_eye, c_test_eye, Rv_eye)

        distance_3d_1 = np.sqrt(np.square(points_3d_test_1[:, 0] - points_3d_reference1_test[:, 0]) +
                                np.square(points_3d_test_1[:, 1] - points_3d_reference1_test[:, 1]) +
                                np.square(points_3d_test_1[:, 2] - points_3d_reference1_test[:, 2]))
        distance_3d_1[points_3d_reference_1[:, 2] < 1] = 50
        distance_3d_1[points_3d_test_1[:, 2] < 1] = 50
        distance_3d_1.sort()
        if int(distance_3d_1[-1] + 0.5) == 50:
            dis_3d_1_sum = np.sum(distance_3d_1)
        else:
            dis_3d_1_sum = np.sum(distance_3d_1[:math.floor(point_num_1 * 0.8)])

        point_num_2 = points_2d_reference_2.shape[0]
        points_3d_reference_2 = map_2dTo3d(points_2d_reference_2, K_refer_2, Rv_camera_reference_2,
                                           t_camera_reference_2, located_camera_reference_2, Rv_eye,
                                           a_reference2_eye, b_reference2_eye, c_reference2_eye,
                                           k1_camera_reference2, k2_camera_reference2)
        points_3d_test_2 = map_2dTo3d(points_2d_test_2, K_test, Rv_camera_test, t_camera_test,
                                      located_camera_test, Rv_eye, a_test_eye, b_test_eye, c_test_eye,
                                      k1_camera_test, k2_camera_test)

        points_3d_reference2_test = pts_3D_from_reference_to_test(points_3d_reference_2,
                                                                  a_test_eye, b_test_eye, c_test_eye, Rv_eye)

        distance_3d_2 = np.sqrt(np.square(points_3d_test_2[:, 0] - points_3d_reference2_test[:, 0]) +
                                np.square(points_3d_test_2[:, 1] - points_3d_reference2_test[:, 1]) +
                                np.square(points_3d_test_2[:, 2] - points_3d_reference2_test[:, 2]))
        distance_3d_2[points_3d_reference_2[:, 2] < 1] = 50
        distance_3d_2[points_3d_test_2[:, 2] < 1] = 50
        distance_3d_2.sort()
        if int(distance_3d_2[-1] + 0.5) == 50:
            dis_3d_2_sum = np.sum(distance_3d_2)
        else:
            dis_3d_2_sum = np.sum(distance_3d_2[:math.floor(point_num_2 * 0.8)])

        fitness[num] = dis_3d_1_sum * weight_refer1_test + dis_3d_2_sum * weight_refer2_test

    return fitness


torch.set_grad_enabled(False)


if __name__ == '__main__':
    time_code_start = time.time()
    parser = argparse.ArgumentParser(
        description='Retinal image sequence registration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_images', type=str, default='RISeR-real/Eye-10/',
        help='Path to an image directory. '
    )
    parser.add_argument(
        '--output_dir', type=str, default='results_evaluation/RISeR-real/riser/Eye-10',
        help='Directory where to write output frames (If None, no output)'
    )
    parser.add_argument(
        '--reference_image_name', type=str, default='2013',
        help='The reference image name we choose. '
    )
    parser.add_argument(
        '--reduce_ratio', type=int, default=4,
        help='The reduce ratio of image pairs. '
    )
    parser.add_argument(
        '--image_glob', type=str, default='*.jpg',
        help='Glob if a directory of images is specified'
    )

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights'
    )
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.001,
        help='SuperPoint keypoint detector confidence threshold'
    )
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)'
    )
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue'
    )
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold'
    )
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             '(\'-1\' keeps all keypoints)'
    )
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints'
    )
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.'
    )

    opt = parser.parse_args()
    print(opt)

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }

    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']
    keys_all = ['keypoints', 'scores', 'descriptors', 'image']

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(parents=True, exist_ok=True)

    search = os.path.join(opt.input_images, opt.image_glob)
    images_path = glob.glob(search)
    images_path.sort()

    time_initial_end = time.time()
    print("\n完成初始化设置用时：", time_initial_end - time_code_start)

    control_points_path = os.path.join(
        opt.input_images,
        "control_points_" + opt.output_dir.split('/')[-1] + ".txt"
    )
    control_points = np.loadtxt(control_points_path, dtype=np.float_, delimiter=' ')
    assert len(images_path) * 2 == control_points.shape[1]
    control_points_images_index = list(range(len(images_path)))

    reference_image_path = None
    control_points_refer_index = None
    for i in range(len(images_path)):
        sub_image_path = images_path[i].split('\\')[-1]
        image_name = sub_image_path.split('.')[0]
        if image_name == opt.reference_image_name:
            reference_image_path = images_path[i]
            images_path.pop(i)
            control_points_refer_index = control_points_images_index[i]
            control_points_images_index.pop(i)
            break
    reference_image_gray = cv2.imread(reference_image_path, 0)
    reference_image_BGR = cv2.imread(reference_image_path)
    mask_refer = generate_mask(reference_image_gray)
    reference_image_gray_masked = cv2.add(reference_image_gray,
                                          np.zeros(np.shape(reference_image_gray), dtype=np.uint8),
                                          mask=mask_refer)
    reference_image_BGR_masked = cv2.add(reference_image_BGR,
                                         np.zeros(np.shape(reference_image_BGR), dtype=np.uint8), mask=mask_refer)
    reference_image_RGB_masked = cv2.cvtColor(reference_image_BGR_masked, cv2.COLOR_BGR2RGB)

    control_pts_refer = control_points[:, 2 * control_points_refer_index:2 * control_points_refer_index + 2]

    v_refer, u_refer = np.shape(mask_refer)
    u0_refer = u_refer / 2
    v0_refer = v_refer / 2
    d = 0
    for i in range(u_refer):
        if mask_refer[int(v0_refer)][i] != 0:
            d = d + 1
    r_refer = d / 2
    K_refer = get_IntrinsicCameraMatrix(r_refer, u0_refer, v0_refer)

    reference_image_gray_resize = cv2.resize(reference_image_gray, None,
                                             fx=1 / opt.reduce_ratio, fy=1 / opt.reduce_ratio,
                                             interpolation=cv2.INTER_AREA)
    mask_refer_resize = generate_mask(reference_image_gray_resize)
    reference_image_gray_resize_masked = cv2.add(reference_image_gray_resize,
                                                 np.zeros(np.shape(reference_image_gray_resize), dtype=np.uint8),
                                                 mask=mask_refer_resize)
    reference_image_gray_resize_masked_tensor = frame2tensor(reference_image_gray_resize_masked, device)
    last_data = matching.superpoint({'image': reference_image_gray_resize_masked_tensor})
    last_data = {k + '0': last_data[k] for k in keys}
    last_data['image0'] = reference_image_gray_resize_masked_tensor
    kp_refer, sc_refer, desc_refer = clean_key_points_near_edges(last_data['keypoints0'][0].detach().numpy(),
                                                                 last_data['scores0'][0].detach().numpy(),
                                                                 last_data['descriptors0'][0].detach().numpy(),
                                                                 mask_refer_resize)
    kp_refer_tensor = torch.from_numpy(kp_refer).to(device)
    sc_refer_tensor = torch.from_numpy(sc_refer).to(device)
    desc_refer_tensor = torch.from_numpy(desc_refer).to(device)
    last_data['keypoints0'][0] = kp_refer_tensor
    last_data['scores0'] = [sc_refer_tensor]
    last_data['descriptors0'][0] = desc_refer_tensor

    time_refer_end = time.time()
    print("\n完成参考图像读取及相关处理用时：", time_refer_end - time_initial_end)

    errors_frame2refer = []
    control_pts_test_registered = []

    sub_image_path = images_path[0].split('\\')[-1]
    test_image_name = sub_image_path.split('.')[0]
    test_image_path = images_path[0]
    test_image_gray = cv2.imread(test_image_path, 0)
    test_image_BGR = cv2.imread(test_image_path)
    mask_test = generate_mask(test_image_gray)
    test_image_gray_masked = cv2.add(test_image_gray, np.zeros(np.shape(test_image_gray), dtype=np.uint8),
                                     mask=mask_test)
    test_image_BGR_masked = cv2.add(test_image_BGR, np.zeros(np.shape(test_image_BGR), dtype=np.uint8),
                                    mask=mask_test)
    test_image_RGB_masked = cv2.cvtColor(test_image_BGR_masked, cv2.COLOR_BGR2RGB)

    test_image_gray_resize = cv2.resize(test_image_gray, None, fx=1 / opt.reduce_ratio, fy=1 / opt.reduce_ratio,
                                        interpolation=cv2.INTER_AREA)
    mask_test_resize = generate_mask(test_image_gray_resize)
    test_image_gray_resize_masked = cv2.add(test_image_gray_resize,
                                            np.zeros(np.shape(test_image_gray_resize), dtype=np.uint8),
                                            mask=mask_test_resize)
    test_image_gray_resize_masked_tensor = frame2tensor(test_image_gray_resize_masked, device)
    now_data = matching.superpoint({'image': test_image_gray_resize_masked_tensor})
    now_data = {k + '1': now_data[k] for k in keys}
    now_data['image1'] = test_image_gray_resize_masked_tensor
    kp_test, sc_test, desc_test = clean_key_points_near_edges(now_data['keypoints1'][0].detach().numpy(),
                                                              now_data['scores1'][0].detach().numpy(),
                                                              now_data['descriptors1'][0].detach().numpy(),
                                                              mask_test_resize)
    kp_test_tensor = torch.from_numpy(kp_test).to(device)
    sc_test_tensor = torch.from_numpy(sc_test).to(device)
    desc_test_tensor = torch.from_numpy(desc_test).to(device)
    now_data['keypoints1'][0] = kp_test_tensor
    now_data['scores1'] = [sc_test_tensor]
    now_data['descriptors1'][0] = desc_test_tensor

    match_pred = matching({**last_data, **now_data})
    kps0 = last_data['keypoints0'][0].cpu().numpy()
    kps1 = now_data['keypoints1'][0].cpu().numpy()
    matches = match_pred['matches0'][0].cpu().numpy()
    confidence = match_pred['matching_scores0'][0].cpu().numpy()
    valid = matches > -1
    mkps0 = kps0[valid]
    mkps1 = kps1[matches[valid]]
    color = cm.jet(confidence[valid])
    text = ['SuperPoint + SuperGlue',
            'Keypoints: {}:{}'.format(len(kps0), len(kps1)),
            'Matches: {}'.format(len(mkps0))]
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = ['Keypoint Threshold: {:.4f}'.format(k_thresh),
                  'Match Threshold: {:.2f}'.format(m_thresh)]
    out, out_match_kps = make_matching_plot_fast(reference_image_gray_resize_masked, test_image_gray_resize_masked,
                                                 kps0, kps1, mkps0, mkps1, color, text, path=None,
                                                 show_keypoints=opt.show_keypoints, small_text=small_text)
    if opt.output_dir is not None:
        stem1 = 'matched_lines_' + opt.reference_image_name + '-' + test_image_name
        out_matched_1 = str(Path(opt.output_dir, stem1 + '.png'))
        print('\nWriting match lines of two images to {}'.format(out_matched_1))
        cv2.imwrite(out_matched_1, out)

        stem2 = 'matched_kps_' + opt.reference_image_name + '-' + test_image_name
        out_matched_2 = str(Path(opt.output_dir, stem2 + '.png'))
        print('\nWriting match kps of two images to {}'.format(out_matched_2))
        cv2.imwrite(out_matched_2, out_match_kps)

    mkps0 = mkps0 * opt.reduce_ratio
    mkps1 = mkps1 * opt.reduce_ratio
    kps0 = kps0 * opt.reduce_ratio
    kps1 = kps1 * opt.reduce_ratio

    v_test, u_test = np.shape(mask_test)
    u0_test = u_test / 2
    v0_test = v_test / 2
    d = 0
    for i in range(u_test):
        if mask_test[int(v0_test)][i] != 0:
            d = d + 1
    r_test = d / 2
    K_test = get_IntrinsicCameraMatrix(r_test, u0_test, v0_test)

    time_test_end = time.time()
    print("\n完成测试图像读取及相关处理用时：", time_test_end - time_refer_end)

    rvec_camera_reference = np.array([0.0, 0.0, 0.0]).reshape((1, -1))
    tvec_camera_reference = np.array([0.0, 0.0, 57.7]).reshape((-1, 1))
    location_camera_reference = np.array([0.0, 0.0, -57.7]).reshape((-1, 1))
    rvec_eye_initial = np.array([0.0, 0.0, 0.0]).reshape((1, -1))
    a_reference_eye_initial = 12.0
    a_test_eye_initial = 12.0
    b_reference_eye_initial = 12.0
    b_test_eye_initial = 12.0
    c_reference_eye_initial = 12.0
    c_test_eye_initial = 12.0
    k1_reference = -0.5623
    k2_reference = 0.3317
    k1_test = -0.5623
    k2_test = 0.3317

    pts_3D_reference = map_2dTo3d(mkps0, K_refer, rvec_camera_reference, tvec_camera_reference,
                                  location_camera_reference, rvec_eye_initial, a_reference_eye_initial,
                                  b_reference_eye_initial, c_reference_eye_initial, k1_reference, k2_reference)

    pts_3D_test = pts_3D_from_reference_to_test(pts_3D_reference, a_test_eye_initial, b_test_eye_initial,
                                                c_test_eye_initial, rvec_eye_initial)

    _, rvec_camera_test_initial, tvec_camera_test_initial, ins = cv2.solvePnPRansac(pts_3D_test, mkps1, K_test,
                                                                                    distCoeffs=None,
                                                                                    iterationsCount=300,
                                                                                    flags=cv2.SOLVEPNP_EPNP)
    print("\n初始化求得的测试相机的旋转向量为：\n", rvec_camera_test_initial)
    print("\n初始化求得的测试相机的平移向量为：\n", tvec_camera_test_initial)

    time_model_initial_end = time.time()
    print("\n完成模型参数的初始化用时：", time_model_initial_end - time_test_end)

    particle_number = 10000
    iter_number = 300
    dim = 19
    options = {'c1': 1.6, 'c2': 1.4, 'w': 0.6}
    oh_strategy = {'w': 'exp_decay', 'c1': 'nonlin_mod', 'c2': 'lin_variation'}
    position_max = np.array([1.0, 1.0, 1.0,
                             2.0, 2.0, 2.0,
                             4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                             2.0, 2.0, 2.0,
                             1.0, 1.0,
                             1.0, 1.0])
    constraints = (
        np.array([rvec_camera_test_initial[0][0] - position_max[0] / 2,
                  rvec_camera_test_initial[1][0] - position_max[1] / 2,
                  rvec_camera_test_initial[2][0] - position_max[2] / 2,
                  tvec_camera_test_initial[0][0] - position_max[3] / 2,
                  tvec_camera_test_initial[1][0] - position_max[4] / 2,
                  tvec_camera_test_initial[2][0] - position_max[5] / 2,
                  a_reference_eye_initial - position_max[6] / 2, a_test_eye_initial - position_max[7] / 2,
                  b_reference_eye_initial - position_max[8] / 2, b_test_eye_initial - position_max[9] / 2,
                  c_reference_eye_initial - position_max[10] / 2, c_test_eye_initial - position_max[11] / 2,
                  0 - position_max[12] / 2, 0 - position_max[13] / 2, 0 - position_max[14] / 2,
                  k1_reference - position_max[15] / 2, k2_reference - position_max[16] / 2,
                  k1_test - position_max[17] / 2, k2_test - position_max[18] / 2]),
        np.array([rvec_camera_test_initial[0][0] + position_max[0] / 2,
                  rvec_camera_test_initial[1][0] + position_max[1] / 2,
                  rvec_camera_test_initial[2][0] + position_max[2] / 2,
                  tvec_camera_test_initial[0][0] + position_max[3] / 2,
                  tvec_camera_test_initial[1][0] + position_max[4] / 2,
                  tvec_camera_test_initial[2][0] + position_max[5] / 2,
                  a_reference_eye_initial + position_max[6] / 2, a_test_eye_initial + position_max[7] / 2,
                  b_reference_eye_initial + position_max[8] / 2, b_test_eye_initial + position_max[9] / 2,
                  c_reference_eye_initial + position_max[10] / 2, c_test_eye_initial + position_max[11] / 2,
                  0 + position_max[12] / 2, 0 + position_max[13] / 2, 0 + position_max[14] / 2,
                  k1_reference + position_max[15] / 2, k2_reference + position_max[16] / 2,
                  k1_test + position_max[17] / 2, k2_test + position_max[18] / 2])
    )
    vel_max = np.array([0.01, 0.01, 0.01,
                        0.1, 0.1, 0.1,
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                        0.01, 0.01, 0.01,
                        0.01, 0.01,
                        0.01, 0.01])
    vel_clamp = (
        np.array([-vel_max[0], -vel_max[1], -vel_max[2],
                  -vel_max[3], -vel_max[4], -vel_max[5],
                  -vel_max[6], -vel_max[7], -vel_max[8], -vel_max[9], -vel_max[10], -vel_max[11],
                  -vel_max[12], -vel_max[13], -vel_max[14],
                  -vel_max[15], -vel_max[16],
                  -vel_max[17], -vel_max[18]]),
        np.array([vel_max[0], vel_max[1], vel_max[2],
                  vel_max[3], vel_max[4], vel_max[5],
                  vel_max[6], vel_max[7], vel_max[8], vel_max[9], vel_max[10], vel_max[11],
                  vel_max[12], vel_max[13], vel_max[14],
                  vel_max[15], vel_max[16],
                  vel_max[17], vel_max[18]])
    )

    optimizer = ps.single.GlobalBestPSO(n_particles=particle_number, dimensions=dim, options=options,
                                        bounds=constraints, oh_strategy=oh_strategy,
                                        bh_strategy='nearest', velocity_clamp=vel_clamp,
                                        vh_strategy='zero')

    cost, best_position = optimizer.optimize(fit_fun_first2all, iters=iter_number, n_processes=10,
                                             points_2d_reference=mkps0, points_2d_test=mkps1,
                                             K_refer=K_refer, K_test=K_test)
    time_PSO_end = time.time()
    pts_num = math.floor(pts_3D_reference.shape[0] * 0.8)
    print("\n最优位置：\n", best_position)
    print("\n最优适应度值：\n", cost)
    print("\n参与计算最优适应度值的点对数量：\n", pts_num)
    print("\n参与计算最优适应度值的点对在3D眼底上的平均欧氏距离（单位是mm）：\n", cost / pts_num)
    print("\n最优化过程用时：", time_PSO_end - time_model_initial_end)
    plot_cost_history(cost_history=optimizer.cost_history)
    if opt.output_dir is not None:
        stem = 'PSO_' + opt.reference_image_name + '-' + test_image_name
        out_PSO = str(Path(opt.output_dir, stem + '.png'))
        print('\nWriting PSO curve to {}'.format(out_PSO))
        plt.savefig(out_PSO)

    rvec_camera_test = np.array([best_position[0], best_position[1], best_position[2]]).reshape((1, -1))
    tvec_camera_test = np.array([best_position[3], best_position[4], best_position[5]]).reshape((-1, 1))
    R_camera_test, _ = cv2.Rodrigues(rvec_camera_test)
    location_camera_test = -np.dot(np.linalg.inv(R_camera_test), tvec_camera_test)
    rvec_eye = np.array([best_position[12], best_position[13], best_position[14]]).reshape((1, -1))
    a_reference = best_position[6]
    a_test = best_position[7]
    b_reference = best_position[8]
    b_test = best_position[9]
    c_reference = best_position[10]
    c_test = best_position[11]
    k1_reference = best_position[15]
    k2_reference = best_position[16]
    k1_test = best_position[17]
    k2_test = best_position[18]

    time_out_start = time.time()
    if opt.output_dir is not None:
        out_registration_result = str(Path(opt.output_dir, test_image_name + '.jpg'))
        print('\nWriting registration result to {}'.format(out_registration_result))
        test_image_registration_result(reference_image_RGB_masked, test_image_RGB_masked, K_refer, K_test,
                                       rvec_camera_test, tvec_camera_test, rvec_eye, a_reference, a_test,
                                       b_reference, b_test, c_reference, c_test,
                                       rvec_camera_reference, tvec_camera_reference, location_camera_reference,
                                       k1_reference, k2_reference, k1_test, k2_test, out_registration_result)
    time_out_end = time.time()
    print("\n生成配准结果用时：", time_out_end - time_out_start)

    time_register_end = time.time()
    print("\n完成首次两幅图像的配准用时：", time_register_end - time_initial_end)

    control_pts_test = control_points[:, 2 * control_points_images_index[0]:2 * control_points_images_index[0] + 2]
    print("\n参考图像上的控制点坐标为：\n", control_pts_refer)
    print("\n测试图像上的控制点坐标为：\n", control_pts_test)
    points_3d_Test = map_2dTo3d(control_pts_test, K_test, rvec_camera_test, tvec_camera_test,
                                location_camera_test, rvec_eye, a_test, b_test, c_test, k1_test,
                                k2_test)
    points_3d_Reference = pts_3D_from_reference_to_test(points_3d_Test,
                                                        a_reference, b_reference, c_reference, rvec_eye)
    points_2d_refer_float = map_3dTo2d(points_3d_Reference, K_refer, rvec_camera_reference, tvec_camera_reference,
                                       k1_reference, k2_reference)
    print("\n测试图像上的控制点配准到参考图像上的坐标为：\n", points_2d_refer_float)
    control_pts_test_registered.append(points_2d_refer_float)

    distances = np.sqrt(np.square(control_pts_refer[:, 0] - points_2d_refer_float[:, 0]) +
                        np.square(control_pts_refer[:, 1] - points_2d_refer_float[:, 1]))
    print("\n所有控制点对之间的配准误差（以像素为单位）分别为：\n", distances)
    average_distance = np.mean(distances)
    errors_frame2refer.append(average_distance)
    print("\n所有控制点对的平均配准误差为（以像素为单位）：", average_distance)

    S_refer_1 = np.array([0.0, 0.0, 0.0,
                          0.0, 0.0, 57.7,
                          a_reference, b_reference, c_reference,
                          k1_reference, k2_reference])
    for j in range(1, len(images_path)):
        time_follow_register_start = time.time()
        S_refer_2 = np.array([rvec_camera_test[0][0], rvec_camera_test[0][1], rvec_camera_test[0][2],
                              tvec_camera_test[0][0], tvec_camera_test[1][0], tvec_camera_test[2][0],
                              a_test, b_test, c_test,
                              k1_test, k2_test])
        K_reference_2 = copy.deepcopy(K_test)

        last_data_2 = copy.deepcopy(now_data)
        last_data_2 = {k + '0': last_data_2[k + '1'] for k in keys_all}

        reference2_image_name = test_image_name
        reference2_image_gray = copy.deepcopy(test_image_gray)
        reference2_image_BGR = copy.deepcopy(test_image_BGR)
        mask_refer2 = copy.deepcopy(mask_test)
        reference2_image_gray_masked = copy.deepcopy(test_image_gray_masked)
        reference2_image_BGR_masked = copy.deepcopy(test_image_BGR_masked)
        reference2_image_RGB_masked = copy.deepcopy(test_image_RGB_masked)
        reference2_image_gray_resize = copy.deepcopy(test_image_gray_resize)
        mask_refer2_resize = copy.deepcopy(mask_test_resize)
        reference2_image_gray_resize_masked = copy.deepcopy(test_image_gray_resize_masked)
        reference2_image_gray_resize_masked_tensor = copy.deepcopy(test_image_gray_resize_masked_tensor)

        sub_image_path = images_path[j].split('\\')[-1]
        test_image_name = sub_image_path.split('.')[0]
        test_image_path = images_path[j]
        test_image_gray = cv2.imread(test_image_path, 0)
        test_image_BGR = cv2.imread(test_image_path)
        mask_test = generate_mask(test_image_gray)
        test_image_gray_masked = cv2.add(test_image_gray, np.zeros(np.shape(test_image_gray), dtype=np.uint8),
                                         mask=mask_test)
        test_image_BGR_masked = cv2.add(test_image_BGR, np.zeros(np.shape(test_image_BGR), dtype=np.uint8),
                                        mask=mask_test)
        test_image_RGB_masked = cv2.cvtColor(test_image_BGR_masked, cv2.COLOR_BGR2RGB)

        test_image_gray_resize = cv2.resize(test_image_gray, None, fx=1 / opt.reduce_ratio, fy=1 / opt.reduce_ratio,
                                            interpolation=cv2.INTER_AREA)
        mask_test_resize = generate_mask(test_image_gray_resize)
        test_image_gray_resize_masked = cv2.add(test_image_gray_resize,
                                                np.zeros(np.shape(test_image_gray_resize), dtype=np.uint8),
                                                mask=mask_test_resize)
        test_image_gray_resize_masked_tensor = frame2tensor(test_image_gray_resize_masked, device)
        now_data = matching.superpoint({'image': test_image_gray_resize_masked_tensor})
        now_data = {k + '1': now_data[k] for k in keys}
        now_data['image1'] = test_image_gray_resize_masked_tensor
        kp_test, sc_test, desc_test = clean_key_points_near_edges(now_data['keypoints1'][0].detach().numpy(),
                                                                  now_data['scores1'][0].detach().numpy(),
                                                                  now_data['descriptors1'][0].detach().numpy(),
                                                                  mask_test_resize)
        kp_test_tensor = torch.from_numpy(kp_test).to(device)
        sc_test_tensor = torch.from_numpy(sc_test).to(device)
        desc_test_tensor = torch.from_numpy(desc_test).to(device)
        now_data['keypoints1'][0] = kp_test_tensor
        now_data['scores1'] = [sc_test_tensor]
        now_data['descriptors1'][0] = desc_test_tensor

        v_test, u_test = np.shape(mask_test)
        u0_test = u_test / 2
        v0_test = v_test / 2
        d = 0
        for i in range(u_test):
            if mask_test[int(v0_test)][i] != 0:
                d = d + 1
        r_test = d / 2
        K_test = get_IntrinsicCameraMatrix(r_test, u0_test, v0_test)

        match_pred_1 = matching({**last_data, **now_data})
        kps0_refer1 = last_data['keypoints0'][0].cpu().numpy()
        kps1_test1 = now_data['keypoints1'][0].cpu().numpy()
        matches_refer1_test = match_pred_1['matches0'][0].cpu().numpy()
        confidence_1 = match_pred_1['matching_scores0'][0].cpu().numpy()
        valid_1 = matches_refer1_test > -1
        mkps0_refer1_test = kps0_refer1[valid_1]
        mkps1_refer1_test = kps1_test1[matches_refer1_test[valid_1]]
        color_1 = cm.jet(confidence_1[valid_1])
        text_1 = ['SuperPoint + SuperGlue',
                  'Keypoints: {}:{}'.format(len(kps0_refer1), len(kps1_test1)),
                  'Matches: {}'.format(len(mkps0_refer1_test))]
        k_thresh_1 = matching.superpoint.config['keypoint_threshold']
        m_thresh_1 = matching.superglue.config['match_threshold']
        small_text_1 = ['Keypoint Threshold: {:.4f}'.format(k_thresh_1),
                        'Match Threshold: {:.2f}'.format(m_thresh_1)]
        out_1, out_match_kps_1 = make_matching_plot_fast(reference_image_gray_resize_masked,
                                                         test_image_gray_resize_masked,
                                                         kps0_refer1, kps1_test1, mkps0_refer1_test, mkps1_refer1_test,
                                                         color_1, text_1, path=None, show_keypoints=opt.show_keypoints,
                                                         small_text=small_text_1)
        if opt.output_dir is not None:
            stem1_refer1_test = 'matched_lines_' + opt.reference_image_name + '-' + test_image_name
            out_matched_1_refer1_test = str(Path(opt.output_dir, stem1_refer1_test + '.png'))
            print('\nWriting match lines of two images to {}'.format(out_matched_1_refer1_test))
            cv2.imwrite(out_matched_1_refer1_test, out_1)

            stem2_refer1_test = 'matched_kps_' + opt.reference_image_name + '-' + test_image_name
            out_matched_2_refer1_test = str(Path(opt.output_dir, stem2_refer1_test + '.png'))
            print('\nWriting match kps of two images to {}'.format(out_matched_2_refer1_test))
            cv2.imwrite(out_matched_2_refer1_test, out_match_kps_1)

        mkps0_refer1_test = mkps0_refer1_test * opt.reduce_ratio
        mkps1_refer1_test = mkps1_refer1_test * opt.reduce_ratio
        kps0_refer1 = kps0_refer1 * opt.reduce_ratio
        kps1_test1 = kps1_test1 * opt.reduce_ratio
        assert len(mkps0_refer1_test) == mkps0_refer1_test.shape[0]
        assert len(mkps0_refer1_test) == len(mkps1_refer1_test)
        kps_match_num_refer1_test = len(mkps0_refer1_test)

        match_pred_2 = matching({**last_data_2, **now_data})
        kps0_refer2 = last_data_2['keypoints0'][0].cpu().numpy()
        kps1_test2 = now_data['keypoints1'][0].cpu().numpy()
        matches_refer2_test = match_pred_2['matches0'][0].cpu().numpy()
        confidence_2 = match_pred_2['matching_scores0'][0].cpu().numpy()
        valid_2 = matches_refer2_test > -1
        mkps0_refer2_test = kps0_refer2[valid_2]
        mkps1_refer2_test = kps1_test2[matches_refer2_test[valid_2]]
        color_2 = cm.jet(confidence_2[valid_2])
        text_2 = ['SuperPoint + SuperGlue',
                  'Keypoints: {}:{}'.format(len(kps0_refer2), len(kps1_test2)),
                  'Matches: {}'.format(len(mkps0_refer2_test))]
        k_thresh_2 = matching.superpoint.config['keypoint_threshold']
        m_thresh_2 = matching.superglue.config['match_threshold']
        small_text_2 = ['Keypoint Threshold: {:.4f}'.format(k_thresh_2),
                        'Match Threshold: {:.2f}'.format(m_thresh_2)]
        out_2, out_match_kps_2 = make_matching_plot_fast(reference2_image_gray_resize_masked,
                                                         test_image_gray_resize_masked,
                                                         kps0_refer2, kps1_test2, mkps0_refer2_test, mkps1_refer2_test,
                                                         color_2, text_2, path=None, show_keypoints=opt.show_keypoints,
                                                         small_text=small_text_2)
        if opt.output_dir is not None:
            stem1_refer2_test = 'matched_lines_' + reference2_image_name + '-' + test_image_name
            out_matched_1_refer2_test = str(Path(opt.output_dir, stem1_refer2_test + '.png'))
            print('\nWriting match lines of two images to {}'.format(out_matched_1_refer2_test))
            cv2.imwrite(out_matched_1_refer2_test, out_2)

            stem2_refer2_test = 'matched_kps_' + reference2_image_name + '-' + test_image_name
            out_matched_2_refer2_test = str(Path(opt.output_dir, stem2_refer2_test + '.png'))
            print('\nWriting match kps of two images to {}'.format(out_matched_2_refer2_test))
            cv2.imwrite(out_matched_2_refer2_test, out_match_kps_2)

        mkps0_refer2_test = mkps0_refer2_test * opt.reduce_ratio
        mkps1_refer2_test = mkps1_refer2_test * opt.reduce_ratio
        kps0_refer2 = kps0_refer2 * opt.reduce_ratio
        kps1_test2 = kps1_test2 * opt.reduce_ratio
        assert len(mkps0_refer2_test) == mkps0_refer2_test.shape[0]
        assert len(mkps0_refer2_test) == len(mkps1_refer2_test)
        kps_match_num_refer2_test = len(mkps0_refer2_test)

        k1_test_initial = -0.5623
        k2_test_initial = 0.3317
        pts_3D_reference1 = map_2dTo3d(mkps0_refer1_test, K_refer, rvec_camera_reference, tvec_camera_reference,
                                       location_camera_reference, rvec_eye,
                                       a_reference, b_reference, c_reference,
                                       k1_reference, k2_reference)
        pts_3D_test1 = pts_3D_from_reference_to_test(pts_3D_reference1, a_test_eye_initial, b_test_eye_initial,
                                                     c_test_eye_initial, rvec_eye)
        _, rvec_camera_test_initial, tvec_camera_test_initial, ins = cv2.solvePnPRansac(pts_3D_test1,
                                                                                        mkps1_refer1_test, K_test,
                                                                                        distCoeffs=None,
                                                                                        iterationsCount=300,
                                                                                        flags=cv2.SOLVEPNP_EPNP)
        print("\n初始化求得的测试相机的旋转向量为：\n", rvec_camera_test_initial)
        print("\n初始化求得的测试相机的平移向量为：\n", tvec_camera_test_initial)
        time_follow_test_end = time.time()
        print("\n完成测试图像读取及相关处理用时：", time_follow_test_end - time_follow_register_start)

        particle_number = 10000
        iter_number = 300
        dim = 11
        options = {'c1': 1.6, 'c2': 1.4, 'w': 0.6}
        oh_strategy = {'w': 'exp_decay', 'c1': 'nonlin_mod', 'c2': 'lin_variation'}
        position_max = np.array([1.0, 1.0, 1.0,
                                 2.0, 2.0, 2.0,
                                 4.0, 4.0, 4.0,
                                 1.0, 1.0])
        constraints = (
            np.array([rvec_camera_test_initial[0][0] - position_max[0] / 2,
                      rvec_camera_test_initial[1][0] - position_max[1] / 2,
                      rvec_camera_test_initial[2][0] - position_max[2] / 2,
                      tvec_camera_test_initial[0][0] - position_max[3] / 2,
                      tvec_camera_test_initial[1][0] - position_max[4] / 2,
                      tvec_camera_test_initial[2][0] - position_max[5] / 2,
                      a_test_eye_initial - position_max[6] / 2,
                      b_test_eye_initial - position_max[7] / 2,
                      c_test_eye_initial - position_max[8] / 2,
                      k1_test_initial - position_max[9] / 2,
                      k2_test_initial - position_max[10] / 2]),
            np.array([rvec_camera_test_initial[0][0] + position_max[0] / 2,
                      rvec_camera_test_initial[1][0] + position_max[1] / 2,
                      rvec_camera_test_initial[2][0] + position_max[2] / 2,
                      tvec_camera_test_initial[0][0] + position_max[3] / 2,
                      tvec_camera_test_initial[1][0] + position_max[4] / 2,
                      tvec_camera_test_initial[2][0] + position_max[5] / 2,
                      a_test_eye_initial + position_max[6] / 2,
                      b_test_eye_initial + position_max[7] / 2,
                      c_test_eye_initial + position_max[8] / 2,
                      k1_test_initial + position_max[9] / 2,
                      k2_test_initial + position_max[10] / 2])
        )
        vel_max = np.array([0.01, 0.01, 0.01,
                            0.1, 0.1, 0.1,
                            0.1, 0.1, 0.1,
                            0.01, 0.01])
        vel_clamp = (
            np.array([-vel_max[0], -vel_max[1], -vel_max[2],
                      -vel_max[3], -vel_max[4], -vel_max[5],
                      -vel_max[6], -vel_max[7], -vel_max[8],
                      -vel_max[9], -vel_max[10]]),
            np.array([vel_max[0], vel_max[1], vel_max[2],
                      vel_max[3], vel_max[4], vel_max[5],
                      vel_max[6], vel_max[7], vel_max[8],
                      vel_max[9], vel_max[10]])
        )
        optimizer = ps.single.GlobalBestPSO(n_particles=particle_number, dimensions=dim, options=options,
                                            bounds=constraints, oh_strategy=oh_strategy,
                                            bh_strategy='nearest', velocity_clamp=vel_clamp,
                                            vh_strategy='zero')
        cost_follow, best_position_follow = optimizer.optimize(fit_fun_follow3onlytest,
                                                               iters=iter_number, n_processes=10,
                                                               points_2d_reference_1=mkps0_refer1_test,
                                                               points_2d_reference_2=mkps0_refer2_test,
                                                               points_2d_test_1=mkps1_refer1_test,
                                                               points_2d_test_2=mkps1_refer2_test,
                                                               K_refer_1=K_refer,
                                                               K_refer_2=K_reference_2,
                                                               K_test=K_test,
                                                               S_reference_1=S_refer_1,
                                                               S_reference_2=S_refer_2,
                                                               Rv_eye=rvec_eye,
                                                               kps_match_number_refer1_test=kps_match_num_refer1_test,
                                                               kps_match_number_refer2_test=kps_match_num_refer2_test)
        time_follow_PSO_end = time.time()
        pts_num = math.floor(kps_match_num_refer1_test * 0.8 + kps_match_num_refer2_test * 0.8)
        print("\n最优位置：\n", best_position_follow)
        print("\n最优适应度值：\n", cost_follow)
        print("\n参与计算最优适应度值的点对数量：\n", pts_num)
        print("\n参与计算最优适应度值的点对在3D眼底上的平均欧氏距离（单位是mm）：\n", cost_follow / pts_num)
        print("\n最优化过程用时：", time_follow_PSO_end - time_follow_test_end)
        plot_cost_history(cost_history=optimizer.cost_history)
        if opt.output_dir is not None:
            stem = 'PSO_' + opt.reference_image_name + '-' + reference2_image_name + '-' + test_image_name
            out_PSO = str(Path(opt.output_dir, stem + '.png'))
            print('\nWriting PSO curve to {}'.format(out_PSO))
            plt.savefig(out_PSO)

        rvec_camera_test = np.array([best_position_follow[0], best_position_follow[1],
                                     best_position_follow[2]]).reshape((1, -1))
        tvec_camera_test = np.array([best_position_follow[3], best_position_follow[4],
                                     best_position_follow[5]]).reshape((-1, 1))
        R_camera_test, _ = cv2.Rodrigues(rvec_camera_test)
        location_camera_test = -np.dot(np.linalg.inv(R_camera_test), tvec_camera_test)
        a_test = best_position_follow[6]
        b_test = best_position_follow[7]
        c_test = best_position_follow[8]
        k1_test = best_position_follow[9]
        k2_test = best_position_follow[10]

        time_follow_out_start = time.time()
        if opt.output_dir is not None:
            out_registration_result = str(Path(opt.output_dir, test_image_name + '.jpg'))
            print('\nWriting registration result to {}'.format(out_registration_result))
            test_image_registration_result(reference_image_RGB_masked, test_image_RGB_masked, K_refer, K_test,
                                           rvec_camera_test, tvec_camera_test, rvec_eye, a_reference, a_test,
                                           b_reference, b_test, c_reference, c_test,
                                           rvec_camera_reference, tvec_camera_reference, location_camera_reference,
                                           k1_reference, k2_reference, k1_test, k2_test, out_registration_result)
        time_follow_out_end = time.time()
        print("\n生成配准结果用时：", time_follow_out_end - time_follow_out_start)

        time_follow_register_end = time.time()
        print("\n完成一次图像对的配准用时：", time_follow_register_end - time_follow_register_start)

        control_pts_test = control_points[:, 2 * control_points_images_index[j]:2 * control_points_images_index[j] + 2]
        print("\n参考图像上的控制点坐标为：\n", control_pts_refer)
        print("\n测试图像上的控制点坐标为：\n", control_pts_test)
        points_3d_Test = map_2dTo3d(control_pts_test, K_test, rvec_camera_test, tvec_camera_test,
                                    location_camera_test, rvec_eye, a_test, b_test, c_test, k1_test,
                                    k2_test)
        points_3d_Reference = pts_3D_from_reference_to_test(points_3d_Test,
                                                            a_reference, b_reference, c_reference, rvec_eye)
        points_2d_refer_float = map_3dTo2d(points_3d_Reference, K_refer, rvec_camera_reference, tvec_camera_reference,
                                           k1_reference, k2_reference)
        print("\n测试图像上的控制点配准到参考图像上的坐标为：\n", points_2d_refer_float)
        control_pts_test_registered.append(points_2d_refer_float)
        distances = np.sqrt(np.square(control_pts_refer[:, 0] - points_2d_refer_float[:, 0]) +
                            np.square(control_pts_refer[:, 1] - points_2d_refer_float[:, 1]))
        print("\n所有控制点对之间的配准误差（以像素为单位）分别为：\n", distances)
        average_distance = np.mean(distances)
        errors_frame2refer.append(average_distance)
        print("\n所有控制点对的平均配准误差为（以像素为单位）：", average_distance)

    print("\n序列中所有图像和参考图像之间的配准误差（以像素为单位）分别为：\n", errors_frame2refer)
    avg_errors_frame2refer = np.mean(errors_frame2refer)
    print("\n序列中所有图像和参考图像之间的平均配准误差为（以像素为单位）：", avg_errors_frame2refer)

    errors_frame2frame = []
    for j in range(len(control_pts_test_registered)-1):
        distances = np.sqrt(np.square(control_pts_test_registered[j][:, 0] - control_pts_test_registered[j + 1][:, 0]) +
                            np.square(control_pts_test_registered[j][:, 1] - control_pts_test_registered[j + 1][:, 1]))
        print("\n相邻帧之间的所有控制点对的配准误差（以像素为单位）分别为：\n", distances)
        average_distance = np.mean(distances)
        errors_frame2frame.append(average_distance)
        print("\n相邻帧之间的所有控制点对的平均配准误差为（以像素为单位）：", average_distance)
    print("\n序列中所有图像和前一帧图像之间的配准误差（以像素为单位）分别为：\n", errors_frame2frame)
    avg_errors_frame2frame = np.mean(errors_frame2frame)
    print("\n序列中所有图像和前一帧图像之间的平均配准误差为（以像素为单位）：", avg_errors_frame2frame)

    errors_frame2refer_frame2frame = errors_frame2refer + errors_frame2frame
    avg_errors_frame2refer_frame2frame = np.mean(errors_frame2refer_frame2frame)
    print("\n序列中所有帧到参考和帧到帧的平均配准误差为（以像素为单位）：", avg_errors_frame2refer_frame2frame)

    time_code_end = time.time()
    print("\n程序总用时：", time_code_end - time_code_start)
