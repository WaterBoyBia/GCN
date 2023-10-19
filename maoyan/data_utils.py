import os
import random
import time

import matplotlib.pyplot as plt
# import jieba
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
# from google_drive_downloader import GoogleDriveDownloader as gdd
from torch import tensor
from torch.utils.data import DataLoader, Dataset
# from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm, tqdm_notebook


class MaoYanDataset(nn.Module):
    def __init__(self, root):
        super(MaoYanDataset, self).__init__()
        self.root = root
        self.df = pd.read_csv(root, index_col=False, low_memory=False)
        self.label = None

    # 只要文本数据
    def get_text_data(self):
        text_list = self.df.iloc[:, 4].to_list()
        movie_name_list = self.df.iloc[:, 2].to_list()

        # 将target不为1的改为0
        # self.df.iloc[:, 0][self.df.iloc[:, 0] != 1] = 0
        # targets = self.df.iloc[:, 0].to_list()
        targets_list = self.df.iloc[:, 0].tolist()
        # print(targets_list)
        # targets_list = [0 if str(i) != '1' or str(i) != '1.0' else 1 for i in targets_list]
        for n in range(len(targets_list)):
            try:
                if int(targets_list[n]) == 1:
                    targets_list[n] = 1
                else:
                    targets_list[n] = 0
            except:
                targets_list[n] = 0

        # print(targets_list)
        return text_list, targets_list, movie_name_list

    def get_text_user_data(self):
        pass



def xx(path='./data/总表.csv'):
    """
    统计用户重名情况
    :param path:
    :return:
    """
    df = pd.read_csv(path)
    user_name = df.iloc[:, 11]
    name_list = {}
    for i in user_name:
        try:
            if name_list.get(i) != 0:
                name_list[i] += 1
        except:
            name_list[i] = 1
    count = 0
    for k, i in name_list.items():
        if i != 1:
            count += 1
            print(k)

    print(count)
    print(len(name_list))


if __name__ == '__main__':
    # maoyan_Dataset_test = MaoYanDataset('./data/总表.csv')
    # print(maoyan_Dataset_test.get_text_data())

    xx()