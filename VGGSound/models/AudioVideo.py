from utils import *
from os import path
from collections import OrderedDict
import torchvision
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import defaultdict
from Resnet import resnet18, resnet34, resnet50


class AudioEncoder(nn.Module):
    def __init__(self, config=None, mask_model=1):
        super(AudioEncoder, self).__init__()
        self.mask_model = mask_model
        if config['text']["name"] == 'resnet18':
            self.audio_net = resnet18(modality='audio')
        elif config['text']["name"] == 'resnet34':
            self.audio_net = resnet34(modality='audio')
        elif config['text']["name"] == 'resnet50':
            self.audio_net = resnet50(modality='audio')

        # self.audio_net = resnet18(modality='audio')
        # self.norm = nn.Sequential(
        #     nn.BatchNorm1d(512), #-----------添加
        #     nn.GELU(),#-----------添加
        # )

    def forward(self, audio, step=0, balance=0, s=400, a_bias=0):
        a = self.audio_net(audio)
        a = F.adaptive_avg_pool2d(a, 1)  # [512,1]
        a = torch.flatten(a, 1)  # [512]
        return a

class VideoEncoder(nn.Module):
    def __init__(self, config=None, fps=1, mask_model=1):
        super(VideoEncoder, self).__init__()
        self.mask_model = mask_model
        if config['visual']["name"] == 'resnet18':
            self.video_net = resnet18(modality='visual')
        elif config['visual']["name"] == 'resnet34':
            self.video_net = resnet34(modality='visual')
        elif config['visual']["name"] == 'resnet50':
            self.video_net = resnet50(modality='visual')
        # self.video_net = resnet18(modality='visual')
        self.fps = fps

    def forward(self, video, step=0, balance=0, s=400, v_bias=0):
        v = self.video_net(video)
        (_, C, H, W) = v.size()
        B = int(v.size()[0] / self.fps)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        return v

class AudioClassifier(nn.Module):
    def __init__(self, config, mask_model=1, act_fun=nn.GELU()):
        super(AudioClassifier, self).__init__()
        self.audio_encoder = AudioEncoder(config, mask_model)

        self.hidden_dim = 512
        # self.linear1 = nn.Linear(self.hidden_dim, 256)
        # self.linear2 = nn.Linear(self.hidden_dim, 256)
        # self.linear3 = nn.Linear(self.hidden_dim, 256)
        # self.linear4 = nn.Linear(self.hidden_dim, 256)
        self.cls_a = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, audio):
        a_feature = self.audio_encoder(audio)
        # a_feature1 = self.linear1(a_feature)
        # a_feature2 = self.linear2(a_feature)
        # a_feature3 = self.linear3(a_feature)
        # a_feature4 = self.linear4(a_feature)
        # result_a = (self.cls_a(a_feature1) + self.cls_a(a_feature2)) / 2.0
        # result_a = (self.cls_a(a_feature1) + self.cls_a(a_feature2) + self.cls_a(a_feature3) + self.cls_a(a_feature4)) / 4.0
        result_a = self.cls_a(a_feature)
        return result_a


class VideoClassifier(nn.Module):
    def __init__(self, config, mask_model=1, act_fun=nn.GELU()):
        super(VideoClassifier, self).__init__()
        self.video_encoder = VideoEncoder(config, config['fps'], mask_model)

        self.hidden_dim = 512
        if config['visual']["name"] == 'resnet50':
            self.hidden_dim = 2048
        # self.linear1 = nn.Linear(self.hidden_dim, 256)
        # self.linear2 = nn.Linear(self.hidden_dim, 256)
        # self.linear3 = nn.Linear(self.hidden_dim, 256)
        # self.linear4 = nn.Linear(self.hidden_dim, 256)
        self.cls_v = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, video):
        v_feature = self.video_encoder(video)
        # v_feature1 = self.linear1(v_feature)
        # v_feature2 = self.linear2(v_feature)
        # v_feature3 = self.linear3(v_feature)
        # v_feature4 = self.linear4(v_feature)
        # result_v = (self.cls_v(v_feature1) + self.cls_v(v_feature2)) / 2.0
        # result_v = (self.cls_v(v_feature1) + self.cls_v(v_feature2) + self.cls_v(v_feature3) + self.cls_v(v_feature4)) / 4.0
        result_v = self.cls_v(v_feature)
        return result_v

class AVClassifier(nn.Module):
    def __init__(self, config, mask_model=1, act_fun=nn.GELU()):
        super(AVClassifier, self).__init__()
        self.audio_encoder = AudioEncoder(config, mask_model)
        self.video_encoder = VideoEncoder(config, config['fps'], mask_model)
        self.hidden_dim = 512
        self.v2a = None
        if config['visual']["name"] == 'resnet50':
            # self.hidden_dim = 2048
            self.v2a = nn.Linear(2048, 512)

        self.cls_a = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config['setting']['num_class'])
        )
        # self.alinear1 = nn.Linear(self.hidden_dim, 256)
        # self.alinear2 = nn.Linear(self.hidden_dim, 256)
        # self.alinear3 = nn.Linear(self.hidden_dim, 256)
        # self.alinear4 = nn.Linear(self.hidden_dim, 256)
        # self.vlinear1 = nn.Linear(self.hidden_dim, 256)
        # self.vlinear2 = nn.Linear(self.hidden_dim, 256)
        # self.vlinear3 = nn.Linear(self.hidden_dim, 256)
        # self.vlinear4 = nn.Linear(self.hidden_dim, 256)
        self.cls_v = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config['setting']['num_class'])
        )
        # self.cls_a = nn.Sequential(
        #     nn.Linear(self.hidden_dim, 256),
        #     nn.ReLU(),
        # )
        # self.cls_v = nn.Sequential(
        #     nn.Linear(self.hidden_dim, 256),
        #     nn.ReLU(),
        # )
        # self.fc = nn.Linear(256, config['setting']['num_class'])
    def forward(self, audio, video):
        a_feature = self.audio_encoder(audio)
        v_feature = self.video_encoder(video)
        if self.v2a is not None:
            v_feature = self.v2a(v_feature)
        # a_feature1 = self.alinear1(a_feature)
        # a_feature2 = self.alinear2(a_feature)
        # a_feature3 = self.alinear3(a_feature)
        # a_feature4 = self.alinear4(a_feature)
        # result_a = (self.cls_a(a_feature1) + self.cls_a(a_feature2) + self.cls_a(a_feature3) + self.cls_a(
        #     a_feature4)) / 4.0
        result_a = self.cls_a(a_feature)
        # v_feature1 = self.vlinear1(v_feature)
        # v_feature2 = self.vlinear2(v_feature)
        # v_feature3 = self.vlinear3(v_feature)
        # v_feature4 = self.vlinear4(v_feature)
        # result_v = (self.cls_v(v_feature1) + self.cls_v(v_feature2) + self.cls_v(v_feature3) + self.cls_v(
        #     v_feature4)) / 4.0
        result_v = self.cls_v(v_feature)
        # result_a = self.fc(self.cls_a(a_feature))
        #
        # result_v = self.fc(self.cls_v(v_feature))
        return result_a, result_v

class AVEClassifier(nn.Module):
    def __init__(self, config, mask_model=1, act_fun=nn.GELU()):
        super(AVEClassifier, self).__init__()
        self.audio_encoder = AudioEncoder(config, mask_model)
        self.video_encoder = VideoEncoder(config, config['fps'], mask_model)
        self.hidden_dim = 1024
        if config['visual']["name"] == 'resnet50':
            self.hidden_dim = 2048 * 2
        self.cls = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, audio, video):
        a_feature = self.audio_encoder(audio)
        v_feature = self.video_encoder(video)
        feature = torch.cat((a_feature, v_feature), dim=1)
        result = self.cls(feature)
        return result

class AVShareClassifier(nn.Module):
    def __init__(self, config, mask_model=1):
        super(AVShareClassifier, self).__init__()
        self.audio_encoder = AudioEncoder(config, mask_model)
        self.video_encoder = VideoEncoder(config, config['fps'], mask_model)
        self.hidden_dim = 512

        self.embedding_a = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
        )
        self.embedding_v = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
        )
        self.num_class = config['setting']['num_class']

        self.fc_out = nn.Linear(256, self.num_class)
        self.additional_layers_a = nn.ModuleList()
        self.additional_layers_v = nn.ModuleList()
        self.relu = nn.ReLU()
        self.rein_network1 = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, config['setting']['num_layers']),
            nn.Sigmoid()
        )

    def forward(self, audio, video):
        a_feature = self.audio_encoder(audio)
        v_feature = self.video_encoder(video)
        return a_feature, v_feature

    def add_layer(self, is_a=True):
        new_layer = nn.Linear(self.hidden_dim, 256).cuda()
        # nn.init.xavier_normal_(new_layer.weight)
        # nn.init.constant_(new_layer.bias, 0)
        if is_a:
            with torch.no_grad():  # 禁用梯度计算以避免不必要的计算
                new_layer.weight.copy_(self.embedding_v[0].weight.data)
                new_layer.bias.copy_(self.embedding_v[0].bias.data)
            self.additional_layers_a.append(new_layer)
        else:
            with torch.no_grad():  # 禁用梯度计算以避免不必要的计算
                new_layer.weight.copy_(self.embedding_a[0].weight.data)
                new_layer.bias.copy_(self.embedding_a[0].bias.data)
            self.additional_layers_v.append(new_layer)

    def classfier(self, x, w, is_a=True):
        if is_a:
            result_a = self.embedding_a(x)
            r = torch.mean(result_a, 0, True)
            feature = self.fc_out(result_a)
            i = 0
            for layer in self.additional_layers_a:
                addf = self.relu(layer(x))
                r += torch.mean(addf, 0, True)
                feature = feature + w[i]*self.fc_out(addf)
                i=i+1
            feature = feature/(i+1)
        else:
            result_v = self.embedding_v(x)
            r = torch.mean(result_v, 0, True)
            feature = self.fc_out(result_v)
            j = 0
            for layer in self.additional_layers_v:
                addf = self.relu(layer(x))
                r += torch.mean(addf, 0, True)
                feature = feature + w[j]*self.fc_out(addf)
                j=j+1
            feature = feature/(j+1)
        return feature, r

class AVShareClassifier_noRein(nn.Module):
    def __init__(self, config, mask_model=1):
        super(AVShareClassifier_noRein, self).__init__()
        self.audio_encoder = AudioEncoder(config, mask_model)
        self.video_encoder = VideoEncoder(config, config['fps'], mask_model)
        self.hidden_dim = 512

        self.embedding_a = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
        )
        self.embedding_v = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
        )
        self.num_class = config['setting']['num_class']

        self.fc_out = nn.Linear(256, self.num_class)
        self.additional_layers_a = nn.ModuleList()
        self.additional_layers_v = nn.ModuleList()
        self.relu = nn.ReLU()
        # self.weight_a_ = nn.Linear(config['setting']['num_layers'], 1)
        # self.weight_a = self.weight_a_.weight
        # self.weight_v_ = nn.Linear(config['setting']['num_layers'], 1)
        # self.weight_v = self.weight_v_.weight
        self.boost_rate = nn.Parameter(torch.full((2, 10), 1.0, requires_grad=True, device="cuda"))
    def forward(self, audio, video):
        a_feature = self.audio_encoder(audio)
        v_feature = self.video_encoder(video)
        return a_feature, v_feature

    def add_layer(self, is_a=True):
        new_layer = nn.Linear(self.hidden_dim, 256).cuda()
        # nn.init.xavier_normal_(new_layer.weight)
        # nn.init.constant_(new_layer.bias, 0)
        if is_a:
            with torch.no_grad():  # 禁用梯度计算以避免不必要的计算
                new_layer.weight.copy_(self.embedding_v[0].weight.data)
                new_layer.bias.copy_(self.embedding_v[0].bias.data)
            self.additional_layers_a.append(new_layer)
        else:
            with torch.no_grad():  # 禁用梯度计算以避免不必要的计算
                new_layer.weight.copy_(self.embedding_a[0].weight.data)
                new_layer.bias.copy_(self.embedding_a[0].bias.data)
            self.additional_layers_v.append(new_layer)

    def classfier(self, x, is_a=True):
        if is_a:
            result_a = self.embedding_a(x)
            r = torch.mean(result_a, 0, True)
            feature = self.fc_out(result_a)
            i = 0
            for layer in self.additional_layers_a:
                addf = self.relu(layer(x))
                r += torch.mean(addf, 0, True)
                # feature = feature + self.weight_a[0][i]*self.fc_out(addf)
                # feature = feature + self.fc_out(addf)
                feature = feature + self.boost_rate[0][i] * self.fc_out(addf)
                i=i+1
            feature = feature/(i+1)
        else:
            result_v = self.embedding_v(x)
            r = torch.mean(result_v, 0, True)
            feature = self.fc_out(result_v)
            j = 0
            for layer in self.additional_layers_v:
                addf = self.relu(layer(x))
                r += torch.mean(addf, 0, True)
                # feature = feature + self.weight_v[0][j]*self.fc_out(addf)
                # feature = feature + self.fc_out(addf)
                feature = feature + self.boost_rate[1][j] * self.fc_out(addf)
                j=j+1
            feature = feature/(j+1)
        return feature, r

class AVGBShareClassifier(nn.Module):
    def __init__(self, config, mask_model=1):
        super(AVGBShareClassifier, self).__init__()
        self.audio_encoder = AudioEncoder(config, mask_model)
        self.video_encoder = VideoEncoder(config, config['fps'], mask_model)
        self.hidden_dim = 512

        self.embedding_a = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
        )
        self.embedding_v = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
        )
        self.num_class = config['setting']['num_class']

        self.fc_out = nn.Linear(256, self.num_class)
        self.additional_layers_a = nn.ModuleList()
        self.additional_layers_v = nn.ModuleList()
        self.relu = nn.ReLU()
        self.rein_network1 = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, audio, video):
        a_feature = self.audio_encoder(audio)
        v_feature = self.video_encoder(video)
        return a_feature, v_feature
