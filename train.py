import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pytorch_pretrained_bert import BertTokenizer
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from config import Config
from maoyan.data_utils import MaoYanDataset
from model import ANNModel, GHAModel
from data_loader import GHADataset, get_dataloader
from torch.utils.tensorboard import SummaryWriter

from utils import DataDeal
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

config = Config()


def train(batch_size=config.batch_size, lr=config.lr, epoches=config.epoch, save_path='./model', check_point=None):

    device = config.DEVICE
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese-vocab.txt')
    maoyan_Dataset_test = MaoYanDataset('./maoyan/data/总表.csv')
    data = maoyan_Dataset_test.get_text_data()
    bert_dataset = GHADataset(data, tokenizer)
    train_loader, val_loader, test_loader = get_dataloader(bert_dataset, batch_size=batch_size, val_rate=config.val_rate,
                                                           test_rate=config.test_rate, random_noise=config.random_noise)

    if check_point:
        model = torch.load(check_point)
    else:
        # model_normal = NormalCnn(0, 0)
        # model = ModelWithAttention()
        # todo
        model = ANNModel(batch_size=batch_size)  # GHAModel()
        model = model.to(device)

        model.train()

    # 优化算法
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam([p for p in model_normal.parameters() if p.requires_grad], lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    if config.cos_train:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2)

    # 处理loss 和 混淆矩阵
    data_deal = DataDeal()

    for epoch in range(epoches):
        # train
        pbar = tqdm(train_loader)  # 进度条
        epoch_loss = []
        for text, mask, target in pbar:
            # print(len(pbar))
            optimizer.zero_grad()
            model.zero_grad()
            text, mask, target = text.to(device), mask.to(device), target.to(device)

            output = model(text)
            loss = criterion(output, target)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()
            pbar.set_description(f"train：epoch{epoch} loss:{sum(epoch_loss) / len(pbar)}")

        train_loss = sum(epoch_loss) / len(pbar)
        data_deal.add_train_loss(train_loss)
        pbar.close()
        # print(f"train：epoch{epoch} loss{train_loss}")
        if config.cos_train:
            scheduler.step()  # 余弦退火
        cur_lr = optimizer.param_groups[-1]['lr']
        # cur_lr_list.append(cur_lr)
        print('cur_lr:', cur_lr)

        # val
        with torch.no_grad():
            pbar = tqdm(val_loader)
            val_epoch_loss = []
            confusion_matrix = torch.zeros(2, 2)

            for text, mask, target in pbar:
                text, mask, target = text.to(device), mask.to(device), target.to(device)
                output = model(text)
                loss = criterion(output, target)

                val_epoch_loss.append(loss.item())

                result = output.argmax(1)
                for t, p in zip(result.view(-1), target.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

            score = data_deal.add_confusion_matrix_and_cal(confusion_matrix)
            val_loss = sum(val_epoch_loss) / len(pbar)

            data_deal.add_val_loss(val_loss)
            pbar.close()
            # pbar.set_description(f"test: epoch{epoch} loss{val_loss} acc{score['准确率']} recall{score['召回率']}")
            print(f"test: epoch{epoch} loss{val_loss} acc{score['准确率']} recall{score['召回率']}")
            if epoch > 300 and (epoch + 1) % 20 == 0:
                name = f"./model/batch_size{batch_size}-loss{val_loss}-acc{score['准确率']}-recall{score['召回率']}.csv"
                torch.save(model, name)

        # 画损失图
        data_deal.add_tensorboard_scalars(f'epoch{epoches}-lr{lr}-batch_size{batch_size}/loss',
                                          {'train loss': train_loss, 'val loss': val_loss}, epoch)
        # 画结果图
        data_deal.add_tensorboard_scalars(f'epoch{epoches}-lr{lr}-batch_size{batch_size}/result',
                                          {'acc': score['准确率'], 'recall': score['召回率'], 'pre': score['精确率']}, epoch)

    data_deal.write_confusion_matrix('./result.csv')


if __name__ == '__main__':
    train()
