# 训练数据处理相关的函数
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import os


class DataDeal():
    """
    每次开始训练前先初始化
    根据混淆矩阵求 准确率、精确率、召回率、F1
    将某次训练结果的混淆矩阵和结果写入表格
    保存模型
    计算loss
    """

    def __init__(self, tensorboard_log='./log'):
        self.confusion_matrix_list = []  # 存每轮epoch的混淆矩阵
        self.acc_list = []
        self.recall_list = []
        self.F1_list = []
        self.precise_list = []
        self.loss_list = []

        self.train_loss_list = []
        self.val_loss_list = []
        self.test_loss_list = []

        # 记录最大值
        self.acc_max = 0
        self.recall_max = 0
        self.F1_max = 0
        self.precise_max = 0

        # 是否使用tensorboard
        if tensorboard_log:
            self.writer = SummaryWriter(tensorboard_log)

    def add_confusion_matrix_and_cal(self, confusion_matrix: torch):
        """
        先result，再target
        tp[1,1]
        fp[1,0]
        tn[0,0]
        fn[0,1]
        :param confusion_matrix:
        :return:{"准确率": accuracy,
            "召回率": recall,
            "精确率": precise,
            "F1": F1}
        """
        self.confusion_matrix_list.append(confusion_matrix)
        tp = confusion_matrix[1, 1]
        tn = confusion_matrix[0, 0]
        fp = confusion_matrix[1, 0]
        fn = confusion_matrix[0, 1]

        # 准确率 模型判断正确的数据(TP+TN)占总数据的比例
        accuracy = float((tp + tn) / (tp + tn + fn + fp))
        # 召回率：针对正例而言 召回率也叫查全率
        recall = float(tp / (tp + fn))
        precise = float(tp / (tp + fp))
        F1 = float(2 * precise * recall / (precise + recall))

        self.acc_list.append(accuracy)
        self.recall_list.append(recall)
        self.precise_list.append(precise)
        self.F1_list.append(F1)

        #
        if accuracy > self.acc_max:
            self.acc_max = accuracy
        if recall > self.recall_max:
            self.recall_max = recall
        if precise > self.precise_max:
            self.precise_max = precise
        if F1 > self.F1_max:
            self.F1_max = F1

        return {
            "准确率": accuracy,
            "召回率": recall,
            "精确率": precise,
            "F1": F1
        }

    def write_confusion_matrix(self, save_path):
        """
        将训练的混淆矩阵和结果写入表格
        :param save_path:
        :return:
        """
        confusion_matrix_list = [i.resize(4).numpy().tolist() for i in
                                      self.confusion_matrix_list]  # 将混淆矩阵二维tensor转成一维list
        confusion_matrix_list = torch.tensor(confusion_matrix_list)  # 转成tensor
        tn = confusion_matrix_list[:, 0]
        fn = confusion_matrix_list[:, 1]
        fp = confusion_matrix_list[:, 2]
        tp = confusion_matrix_list[:, 3]

        df = pd.DataFrame({
            'tn': tn.numpy().tolist(),
            'fn': fn.numpy().tolist(),
            'tp': tp.numpy().tolist(),
            'fp': fp.numpy().tolist(),
            '准确率': self.acc_list,
            '精确率': self.precise_list,
            '召回率': self.recall_list,
            'loss': self.loss_list,
            'F1': self.F1_list
        })
        df.to_csv(save_path)

    def add_tensorboard_scalars(self, path, dir, epoch):
        """
        一张图多个曲线
        eg: writer.add_scalars(path, {'train loss': train_loss, 'val loss': val_loss}, epoch)
        :param path:
        :param dir:
        :param eopch:
        :return:
        """
        self.writer.add_scalars(path, dir, epoch)

    def add_tensorboard_scalar(self, path, data, epoch):
        # 一张图一条曲线
        self.writer.add_scalars(path, data, epoch)

    # 记录训练、验证、测试误差
    def add_train_loss(self, train_loss):
        self.train_loss_list.append(train_loss)

    def add_val_loss(self, val_loss):
        self.val_loss_list.append(val_loss)

    def add_test_loss(self, test_loss):
        self.test_loss_list.append(test_loss)


class DealConfusionMatrix:
    def __init__(self, save_path, write_all_flag=True):
        self.save_path = save_path
        self.write_all_flag = write_all_flag  # 一次性全部写入表格

    def write_matrix(self, data):
        """
        把混淆矩阵写入
        :param data:
        :return:
        """
        if self.write_all_flag:
            df = pd.DataFrame(data, columns=['TP', 'FP', 'TN', 'FN'])
            df.to_csv(self.save_path)
            return 0

        # 按行写入
        if os.path.exists(self.save_path):
            df = pd.DataFrame(data)
            df.to_csv(self.save_path, mode='a', header=False, index=False)
        else:
            # 之前没写过 就写个头进去
            df = pd.DataFrame(data, columns=['TP', 'FP', 'TN', 'FN'])
            df.to_csv(self.save_path, mode='a')


if __name__ == '__main__':
    tt = ['sdf']
    print(torch.tensor(tt))
    x = torch.tensor([[1, 2], [3, 4]]).resize(4)
    y = torch.ones([2, 2]).resize(4)
    z = [x, y]
    print(z)
    print(list(z))
    print(float(x[0]/x[1]))
    # t = torch.tensor(z)
    # df = pd.DataFrame(t)
    # df.to_csv('ss.csv')

