# from utils import *
from os import path
import torchvision
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models.Model3D import InceptionI3d
from models.Resnet import resnet18

import torch.nn as nn
import torch


class RGBEncoder(nn.Module):
    def __init__(self, config):
        super(RGBEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=3)
        # download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        pretrained_dict = torch.load('/data/hlf/imbalance/unimodal/checkpoint/rgb_imagenet.pt')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.rgbmodel = model

    def forward(self, x):
        out = self.rgbmodel(x)
        return out  # BxNx2048


class OFEncoder(nn.Module):
    def __init__(self, config):
        super(OFEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=2)
        # download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        pretrained_dict = torch.load('/data/hlf/imbalance/unimodal/checkpoint/flow_imagenet.pt')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.ofmodel = model

    def forward(self, x):
        out = self.ofmodel(x)
        return out  # BxNx2048


class DepthEncoder(nn.Module):
    def __init__(self, config):
        super(DepthEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=1)
        # download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        pretrained_dict = torch.load('/data/hlf/imbalance/unimodal/checkpoint/rgb_imagenet.pt')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.depthmodel = model

    def forward(self, x):
        out = self.depthmodel(x)
        return out  # BxNx2048


class RGBClsModel(nn.Module):
    def __init__(self, config):
        super(RGBClsModel, self).__init__()
        self.rgb_encoder = RGBEncoder(config)

        self.hidden_dim = 1024
        self.cls_r = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, x1):
        rgb = x1
        rgb_feat = self.rgb_encoder(rgb)
        result_r = self.cls_r(rgb_feat)
        return result_r


class OFClsModel(nn.Module):
    def __init__(self, config):
        super(OFClsModel, self).__init__()
        self.of_encoder = OFEncoder(config)

        self.hidden_dim = 1024
        self.cls_o = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, x1):
        of = x1
        of_feat = self.of_encoder(of)
        result_o = self.cls_o(of_feat)
        return result_o


class DepthClsModel(nn.Module):
    def __init__(self, config):
        super(DepthClsModel, self).__init__()
        self.depth_encoder = DepthEncoder(config)

        self.hidden_dim = 1024
        self.cls_d = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, x1):
        depth = x1
        depth_feat = self.depth_encoder(depth)
        result_d = self.cls_d(depth_feat)
        return result_d


class JointClsModel(nn.Module):
    def __init__(self, config):
        super(JointClsModel, self).__init__()
        self.rgb_encoder = RGBEncoder(config)
        self.of_encoder = OFEncoder(config)
        self.depth_encoder = DepthEncoder(config)
        self.hidden_dim = 1024
        self.cls_r = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )
        self.cls_o = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )
        self.cls_d = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, x1, x2, x3):
        rgb = x1
        of = x2
        depth = x3

        rgb_feat = self.rgb_encoder(rgb)
        result_r = self.cls_r(rgb_feat)

        of_feat = self.of_encoder(of)
        result_o = self.cls_o(of_feat)

        depth_feat = self.depth_encoder(depth)
        result_d = self.cls_d(depth_feat)

        return result_r, result_o, result_d


class ShareClassfier(nn.Module):
    def __init__(self, args, mask_model=1, act_fun=nn.GELU()):
        super(ShareClassfier, self).__init__()
        self.rgb_encoder = RGBEncoder(args)
        self.of_encoder = OFEncoder(args)
        self.depth_encoder = DepthEncoder(args)
        # self.hidden_dim = 1024
        # self.cls_shared = nn.Sequential(
        #     nn.Linear(self.hidden_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.Linear(64, 25)
        # )
        self.cls_m1 = nn.Linear(1024, 25)
        self.cls_m2 = nn.Linear(1024, 25)
        self.cls_m3 = nn.Linear(1024, 25)
        self.cls_mm = nn.Linear(1024*3, 25)

    def forward(self, rgb, of, depth):
        # 获取音频和视频的特征
        # rgb:(2,3,64,224,224)
        # of:(2,2,80,224,224)
        # depth:(2,1,64,224,224)
        rgb_feature = self.rgb_encoder(rgb) # (bs,1024)
        of_feature = self.of_encoder(of) # (bs, 1024)
        depth_feature = self.depth_encoder(depth) # (bs,1024)
        out_m1 = self.cls_m1(rgb_feature)
        out_m2 = self.cls_m2(of_feature)
        out_m3 = self.cls_m3(depth_feature)
        out_mm = self.cls_mm(torch.cat((rgb_feature, of_feature, depth_feature), dim=1))


        return out_m1, out_m2, out_m3, out_mm
        # return out_mm