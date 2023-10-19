import random

import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer
# from pytorch_pretrained import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset

from maoyan.data_utils import MaoYanDataset

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


class GHADataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=128, balance=True, flag=0,
                 movie_dic='./graph_section/电影名单.csv'):

        """

        :param data: [text, label, user_feature, movie_name]
        :param tokenizer:
        :param max_seq_len:
        :param balance:
        :param flag: 0 只用文本特征；1文本+用户；2文本+图；3 全特征
        :param movie_dic: 电影序号对应表
        """
        super(GHADataset, self).__init__()
        self.flag = flag
        self.movie_dic = pd.read_csv(movie_dic)
        self.movie_dic = {name: ind for name, ind in
                          enumerate(self.movie_dic.name.to_list(), len(self.movie_dic.index.to_list()))}

        self.data = data
        self.text = data[0]
        self.targets = data[1]
        # self.movie_ind = [self.movie_dic.get(i) for i in range(3)]  # 可删除？？
        self.user_feature = data[2]

        self.tokenizer = tokenizer
        self.max_length = max_seq_len

        # 数据均衡，让虚假评论和真实评论一样多
        if balance:
            fake_count = 0
            fake_ind = []
            # 统计虚假评论的index
            for i in range(len(self.targets)):
                if self.targets[i] == 1:
                    fake_count += 1
                    fake_ind.append(i)

            ind_set = set(fake_ind)  # 存评论的index
            count = len(fake_ind)
            while count < 2 * len(fake_ind):
                ind = random.randint(0, len(self.targets) - 1)
                if ind in ind_set:
                    continue
                else:
                    if self.targets[ind] == 0:
                        count += 1
                        ind_set.add(ind)

            # 数据均衡
            self.text = [self.text[i] for i in ind_set]
            self.targets = [self.targets[i] for i in ind_set]
            # print(len(self.text))
            # print(len(self.targets))
            #
            assert len(self.targets) == len(fake_ind) * 2

    def __getitem__(self, item):
        text = self.text[item]
        # print(text)
        target = self.targets[item]
        seq, seq_mask, _ = self.trunate_and_pad(text, self.max_length)
        # ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(seq)])
        ids = torch.tensor(seq)
        seq_mask = torch.tensor(seq_mask)
        return ids, seq_mask, target

    def __len__(self):
        return len(self.targets)

    def trunate_and_pad(self, seq, max_seq_len=128):
        """
        1. 因为本类处理的是单句序列，按照BERT中的序列处理方式，需要在输入序列头尾分别拼接特殊字符'CLS'与'SEP'，
           因此不包含两个特殊字符的序列长度应该小于等于max_seq_len-2，如果序列长度大于该值需要那么进行截断。
        2. 对输入的序列 最终形成['CLS',seq,'SEP']的序列，该序列的长度如果小于max_seq_len，那么使用0进行填充。

        入参:
            seq         : 输入序列，在本处其为单个句子。
            max_seq_len : 拼接'CLS'与'SEP'这两个特殊字符后的序列长度

        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。

        """
        # 分词
        seq = self.tokenizer.tokenize(seq)
        # 对超长序列进行截断
        if len(seq) > max_seq_len:
            seq = seq[0:max_seq_len]
        # 分别在首尾拼接特殊符号
        # seq = ['[CLS]'] + seq + ['[SEP]']

        # ID化
        seq = self.tokenizer.convert_tokens_to_ids(seq)
        # seq = self.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = [0] * len(seq) + padding
        # 对seq拼接填充序列
        seq += padding
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        assert len(seq_segment) == max_seq_len
        return seq, seq_mask, seq_segment


def get_dataloader(dataset, batch_size, shuffle=True, train=True, val_rate=0.2, test_rate=0.2, random_noise=1):
    """

    :param dataset:
    :param batch_size:
    :param shuffle:
    :param train:
    :param rate:
    :return:
    """
    if not train:
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return test_loader
    len_test = int(len(dataset) * test_rate)
    len_val = int(len(dataset) * val_rate)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                             [len(dataset) - len_test - len_val,
                                                                              len_val, len_test])

    if random_noise:
        random_data = [list(i) for i in train_dataset]
        for i in range(len(random_data)):
            random_flag = random.random()
            if random_flag <= random_noise:
                random_data[i][-1] = 0 if random_data[i][-1] == 1 else 1
        train_loader = DataLoader(random_data, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, val_loader, test_loader



if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese-vocab.txt')
    maoyan_Dataset_test = MaoYanDataset('./maoyan/data/总表.csv')
    data = maoyan_Dataset_test.get_text_data()
    da = GHADataset(data, tokenizer)
    print(da[1])
    # s = tokenizer.tokenize('[CLS]其实我…“”“”“都不评论的，因为懒首先我想说，我爱古风没有10年也有8.9年了吧，从小喜欢中国历史和诗词，这应该受我爸影响。看上看见有人说什么不讲乐理不讲历史强行拔高，增加虚妄的民族自尊感，我想说那是你不懂古风，不懂我们这些爱古风的孩子是怎么用自己的力量去弘扬古风！我相信每一个爱古风的孩子都给身边的人以各种途径安利过关于古风的一切，换来的不是告诉你听不懂，就是笑着问你你要穿越啊？我永远记得第一次听见楼姐《又何用》时的惊艳，也是从那时候更努力的读历史，背诗词，想着有一天我也能填出这样一首词，所以我不认为那是拔高，更何况民族自尊感自豪感都是与生俱来的，当你读过历史，就更应该为我大华夏自豪！')
    # print(tokenizer.convert_tokens_to_ids(s))
