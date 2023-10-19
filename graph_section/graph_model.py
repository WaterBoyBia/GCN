import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(A, symmetric=True):
    # A = A+I
    A = A + torch.eye(A.size(0))
    # 所有节点的度
    d = A.sum(1)
    if symmetric:
        # D = D^-1/2
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)
    else:
        # D=D^-1
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)


class GCN(nn.Module):
    """
    Z = AXW
    """

    def __init__(self, A, dim_in, dim_out):
        super(GCN, self).__init__()
        self.A = A
        self.fc1 = nn.Linear(dim_in, dim_in, bias=False)
        self.fc2 = nn.Linear(dim_in, dim_in // 2, bias=False)
        self.fc3 = nn.Linear(dim_in // 2, dim_out, bias=False)

    def forward(self, X):
        """
        计算三层gcn
        :param X:
        :return:
        """
        X = F.relu(self.fc1(self.A.mm(X)))
        X = F.relu(self.fc2(self.A.mm(X)))
        return self.fc3(self.A.mm(X))


class KnowledgeGraph(nn.Module):
    def __init__(self,
                 vocab_size,  # 分词表大小
                 A,     # 邻接矩阵
                 x,     # 所有长评+电影简介
                 graph_out_size,    # 图嵌入后输出的大小
                 embedding_dimension=100,
                 hidden_size=128,
                 n_layers=1,
                 pretrained_lstm=None):
        super(KnowledgeGraph, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.A = A  # 邻接矩阵
        self.x = x
        self.graph_out_size = graph_out_size    # 图嵌入后的输出大小
        self.encoder = nn.Embedding(vocab_size, embedding_dimension)    # [sentence_len, emb_dim]

        if pretrained_lstm:
            self.bi_lstm = torch.load(pretrained_lstm)
        else:
            self.bi_lstm = nn.LSTM(input_size=embedding_dimension, hidden_size=self.hidden_size,
                                   num_layers=self.n_layers)  # (input_size,hidden_size,num_layers)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(len(x)*self.hidden_size, 1200)
        self.linear2 = nn.Linear(1200, 8*self.graph_out_size)

        self.gcn = GCN(self.A, 8*self.graph_out_size, self.graph_out_size)   # 邻接矩阵、输入维度、输出维度

    def forward(self, movie_ids):
        #   输入是所有长评
        encode = self.encoder(self.x)   #[node_count, sentence_len, emb_dim]
        x = self.bi_lstm(encode)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)

        x = self.gcn(x)     #[movie&long_comment_count, out_dim]

        # 返回指定ids的值
        x = torch.ones([len(x), self.graph_out_size])   # todo

        return x    # 所有电影嵌入的结果





