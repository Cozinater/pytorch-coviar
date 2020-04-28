"""Model definition."""

import torch
from torch import nn
import torch.nn.functional as F
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
import torchvision
import numpy as np

dp_rate = 0.2
norm_layer = nn.BatchNorm2d

def conv1(in_channels, out_channels, relu=True):
    blocks = nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, 1, bias=False),
        nn.BatchNorm1d(out_channels),
    )
    if relu:
        blocks = nn.Sequential(
            blocks,
            nn.ReLU(inplace=True),
        )
    return blocks

def conv2(in_channels, out_channels, relu=True):
    blocks = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        norm_layer(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 1, bias=False),
        norm_layer(out_channels),
    )
    if relu:
        blocks = nn.Sequential(
            blocks,
            nn.ReLU(inplace=True),
        )
    return blocks

def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def cal_sim(x_c, mu):
    # similarity
    z_d = torch.bmm(x_c.permute(0, 2, 1), mu)  # B x N x K
    # normalization
    z_d = F.softmax(z_d, dim=2)
    z_d = z_d / (1e-6 + z_d.sum(dim=1, keepdim=True))
    z_d = z_d.permute(0, 2, 1)  # B x K x N
    return z_d

class PPool(nn.Module):
    """
    module for bases
    """

    def __init__(self, channels, topk, feat_sz):
        super(PPool, self).__init__()
        if isinstance(feat_sz, list) or isinstance(feat_sz, tuple):
            cell_num = np.prod(feat_sz)
        else:
            cell_num = feat_sz
        self.convk = conv1(cell_num, topk)
        self.convc = conv1(channels, channels)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)  # B x C x N
        x = self.convk(x.permute(0, 2, 1).contiguous())
        x = self.convc(x.permute(0, 2, 1).contiguous())
        return x

class BaseNL(nn.Module):
    """Basic NL block from CVPR 2018.
    """
    def __init__(self, in_channels, scale_factor=1, use_norm=False):
        super(BaseNL, self).__init__()
        self.scale_factor = scale_factor
        self.use_norm = use_norm
        self.t = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.z = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x

        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=1/self.scale_factor, mode='bilinear', align_corners=True)

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.size()

        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)
        g = g.view(b, c, -1).permute(0, 2, 1)

        att = torch.bmm(t, p)

        if self.use_norm:
            att = att.div(c**0.5)

        att = self.softmax(att)
        x = torch.bmm(att, g)

        x = x.permute(0, 2, 1)
        x = x.contiguous()
        x = x.view(b, c, h, w)
        x = self.z(x)
        x = self.bn(x)
        if self.scale_factor != 1:
            x = F.interpolate(x, size=(residual.shape[2], residual.shape[3]), mode='bilinear', align_corners=True)

        x = F.relu(x, inplace=True) + residual
        return x
    
class A2Block(nn.Module):
    """
    A2-Net from NIPS 2018
    """
    def __init__(self, in_channels, out_channels):
        super(A2Block, self).__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 1)
        self.up = nn.Conv2d(out_channels, in_channels, 1)
        self.gather_down = nn.Conv2d(in_channels, out_channels, 1)
        self.distribue_down = nn.Conv2d(in_channels, out_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        res = x
        A = self.down(res)
        B = self.gather_down(res)
        b, c, h, w = A.size()
        A = A.view(b, c, -1)  # (b, c, h*w)
        B = B.view(b, c, -1)  # (b, c, h*w)
        B = self.softmax(B)
        B = B.permute(0, 2, 1)  # (b, h*w, c)

        G = torch.bmm(A, B)  # (b,c,c)

        C = self.distribue_down(res)
        C = C.view(b, c, -1)  # (b, c, h*w)
        C = self.softmax(C)
        C = C.permute(0, 2, 1)  # (b, h*w, c)

        atten = torch.bmm(C, G)  # (b, h*w, c)
        atten = atten.permute(0, 2, 1).view(b, c, h, -1)
        atten = self.up(atten)
        out = res + atten
        return out
    
#TopKAtt feature
class TopKAtt(nn.Module):
    def __init__(self, in_channels, out_channels, topk=24, poolsz=[14, 14]):
        super(TopKAtt, self).__init__()
        self.ppool = PPool(out_channels, topk, poolsz)
        self.in_conv = nn.Sequential(
            nn.Dropout2d(dp_rate),
            conv2(in_channels, out_channels),
        )
        self.feat_conv = conv2(out_channels, out_channels)
        self.gp_conv = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            conv1(out_channels, out_channels),
        )
        self.scores = nn.Sequential(
            conv1(out_channels, out_channels, relu=False),
            nn.Softmax(dim=1),
        )
        self.topk = topk

    def forward(self, x):
        x_res = x
        x = self.in_conv(x)
        B, C, H, W = x.shape
        x_c = self.feat_conv(x.clone()).view(B, C, -1)
        x_gp = self.gp_conv(x_c)
        x_score = self.scores(x_gp)
        mu = self.ppool(x_c)  # B x C x K
        mu = cat([mu, x_gp], dim=2)
        z_d = cal_sim(x_c, mu)
        featk = (x_score * mu.matmul(z_d)).view(B, -1, H, W) + x_res
        return featk, mu

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Model(nn.Module):
    def __init__(self, num_class, num_segments, representation, feature_branch, topk,
                 base_model='resnet152'):
        super(Model, self).__init__()
        self._representation = representation
        self.num_segments = num_segments
        self.feature_branch = feature_branch
        self.topk = topk

        print(self.feature_branch)

        print(("""
Initializing model:
    base model:         {}.
    input_representation:     {}.
    num_class:          {}.
    num_segments:       {}.
        """.format(base_model, self._representation, num_class, self.num_segments)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)

    def _prepare_tsn(self, num_class):

        feature_dim = getattr(self.base_model, 'fc').in_features
        setattr(self.base_model, 'fc', nn.Linear(feature_dim, num_class))

        if self.topk is not None:
            self.topfeat = nn.Sequential(TopKAtt(256, 256, self.topk),)
        self.a2block =  nn.Sequential(A2Block(256, 256),)
        self.basenl =  nn.Sequential(BaseNL(256),)
        
        if self._representation == 'mv':
            setattr(self.base_model, 'conv1',
                    nn.Conv2d(2, 64, 
                              kernel_size=(7, 7),
                              stride=(2, 2),
                              padding=(3, 3),
                              bias=False))
            self.data_bn = nn.BatchNorm2d(2)
        if self._representation == 'residual':
            self.data_bn = nn.BatchNorm2d(3)

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(pretrained=True)

            self._input_size = 224
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    @staticmethod
    def basic_forward(basenet, x):
        x = basenet.conv1(x)
        x = basenet.bn1(x)
        x = basenet.relu(x)
        x = basenet.maxpool(x)

        x = basenet.layer1(x)
        x = basenet.layer2(x)
        x = basenet.layer3(x)
        return x

    def forward(self, input):
        input = input.view((-1, ) + input.size()[-3:])
        if self._representation in ['mv', 'residual']:
            input = self.data_bn(input)

        if self.feature_branch in ['topKAtt', 'basenl', 'a2block']:
            feat_flow_0 = Model.basic_forward(self.base_model, input)
            
            if self.feature_branch == 'topKAtt':
                feat_flow_1, _ = self.topfeat(feat_flow_0)
            if self.feature_branch == 'basenl':
                feat_flow_1 = self.basenl(feat_flow_0)
            if self.feature_branch == 'a2block':
                feat_flow_1 = self.a2block(feat_flow_0)
                
            x = self.base_model.layer4(feat_flow_1)
            x = self.base_model.avgpool(x)
            x = torch.flatten(x, 1)
            base_out = self.base_model.fc(x)
        elif self.feature_branch is None:
            base_out = self.base_model(input)

        return base_out

    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224

    def get_augmentation(self):
        if self._representation in ['mv', 'residual']:
            scales = [1, .875, .75]
        else:
            scales = [1, .875, .75, .66]

        print('Augmentation scales:', scales)
        return torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales),
             GroupRandomHorizontalFlip(is_mv=(self._representation == 'mv'))])
