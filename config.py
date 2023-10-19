import torch


class Config:
    def __init__(self):
        # 训练参数
        self.tokenizer = None
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epoch = 100  # 500
        self.lr = 0.01  # 0.001
        self.batch_size = 128  # 64 8
        self.weight_decay = 0
        self.cos_train = False

        # 模型通用参数
        self.vocab_size = 51429  # 词表大小 21128

        # 图参数
        self.A = None  # 邻接矩阵
        self.x = None   # todo
        self.embeding_dimension_g = 128     # 词嵌入维度
        self.hidden_size_g = 64,   # 隐藏层维度
        self.n_layer_g = 1      # lstm隐藏层数
        self.pretrained_lstm = None     # 预训练的lstm路径
        self.output_dim_g = 64    # 图最后输出的维度

        # ANN模型参数
        self.embeding_dimension_a = 48  # 128
        self.output_dim_a = 48  # 128
        self.bert_path = './bert_pretrain'


        # 用户特征输出参数
        self.input_dim_u = 11   # 用户特征数 10
        self.output_dim_u = 20    # 输出维度 5

        # 训练集测试集比例
        self.test_rate = 0.2
        self.val_rate = 0.2
        self.random_noise = 0.2
