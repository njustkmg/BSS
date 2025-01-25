import argparse
import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pdb
# from dataset.sarcasm import dataSarcasm, train_transform, test_transform
from models.basic_model import CLIP_1
from utils.utils import setup_seed, weight_init
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
# from transformers import BertModel, BertTokenizer
# import clip


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    print('GPU设备数量为:', torch.cuda.device_count())
