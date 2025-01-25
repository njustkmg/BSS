import sys
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from zz_mmnas.utils.utilss import pre_caption, clean_str
import os
import random
import torch
import numpy as np
import librosa
from torchvision import transforms
import glob, json, re, torch, en_vectors_web_lg, random
from zz_mmnas.model.model import TextEncoder
from dataset.randaugment import RandomAugment

# class dataSarcasm(Dataset):
#     def __init__(self, ann_file, transform, image_root, max_words=30):
#         self.info = pd.read_csv(ann_file, sep='\t')
#         self.transform = transform
#         self.image_root = image_root
#         self.max_words = max_words
#
#         # Loading all txt
#         total_text_list = []
#         ann_file1 = '/data/php_code/data_processing/Sarcasm/annotations/train.tsv'
#         ann_file2 = '/data/php_code/data_processing/Sarcasm/annotations/valid.tsv'
#         ann_file3 = '/data/php_code/data_processing/Sarcasm/annotations/test.tsv'
#         info1_ = pd.read_csv(ann_file1, sep='\t')
#         info2_ = pd.read_csv(ann_file2, sep='\t')
#         info3_ = pd.read_csv(ann_file3, sep='\t')
#         info1 = info1_['String']
#         info2 = info2_['String']
#         info3 = info3_['String']
#         for i in range(len(info1)):
#             total_text_list.append(info1[i])
#         for j in range(len(info2)):
#             total_text_list.append(info2[j])
#         for k in range(len(info3)):
#             total_text_list.append(info3[k])
#         self.token_to_ix, self.pretrained_emb, max_token = self.tokenize(total_text_list, use_glove=True)
#         self.token_size = len(self.token_to_ix)
#
#     def tokenize(self, stat_caps_list, use_glove=None):
#         max_token = 0
#         token_to_ix = {
#             'PAD': 0,
#             'UNK': 1,
#             'CLS': 2,
#         }
#
#         spacy_tool = None
#         pretrained_emb = []
#         if use_glove:
#             spacy_tool = en_vectors_web_lg.load()
#             pretrained_emb.append(spacy_tool('PAD').vector)
#             pretrained_emb.append(spacy_tool('UNK').vector)
#             pretrained_emb.append(spacy_tool('CLS').vector)
#
#         for cap in stat_caps_list:
#             words = re.sub(r"([.,'!?\"()*#:;])", '', cap.lower()).replace('-', ' ').replace('/', ' ').split()
#             max_token = max(len(words), max_token)
#             for word in words:
#                 if word not in token_to_ix:
#                     token_to_ix[word] = len(token_to_ix)
#                     if use_glove:
#                         pretrained_emb.append(spacy_tool(word).vector)
#
#         pretrained_emb = np.array(pretrained_emb)
#
#         return token_to_ix, pretrained_emb, max_token
#
#     def proc_cap(self, cap, token_to_ix, max_token):
#         cap_ix = np.zeros(max_token, np.int64)
#         words = re.sub(r"([.,'!?\"()*#:;])", '', cap.lower()).replace('-', ' ').replace('/', ' ').split()
#
#         for ix, word in enumerate(words):
#             if word in token_to_ix:
#                 cap_ix[ix] = token_to_ix[word]
#             else:
#                 cap_ix[ix] = token_to_ix['UNK']
#
#             if ix + 1 == max_token:
#                 break
#
#         return cap_ix
#
#     def __len__(self):
#         return len(self.info)
#
#     def __getitem__(self, index):
#         label = self.info['Label'][index]
#         label = int(label)
#
#         text = self.info['String'][index]
#         text = clean_str(text)
#         text = pre_caption(text, self.max_words)
#
#         # text1 = self.info['String'][index]
#         # text2 = clean_str(text1)  # 文本清洗：去除一些符号
#         # text3 = pre_caption(text2, self.max_words)  # # 文本清洗：大写变小写
#         # text4 = self.proc_cap(text3, self.token_to_ix, max_token=50)  # 当前文本，word-level
#         # text = torch.from_numpy(text4)  # 从numpy数组-->torch张量
#
#         imagePath = self.image_root + self.info['ImageID'][index]
#         image = Image.open(imagePath).convert('RGB')
#         image = self.transform(image)
#
#         return image, text, label

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
class dataSarcasm(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}
        self.cls_num = 2
        self.labels = []
        for i in self.ann:
            self.labels.append(int(i['label']))

    def __len__(self):
        return len(self.ann)

    def get_num_classes(self):
        return self.cls_num

    def __getitem__(self, index):
        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'].split('/')[-1])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['text'], self.max_words)
        target = int(ann['label'])
        return image, caption, target

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(),
    RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                          'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
    transforms.ToTensor(),
    normalize,
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
])