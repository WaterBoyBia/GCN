import math
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from config import *
from graph_section.graph_model import KnowledgeGraph

config = Config()
device = config.DEVICE


# hidden_size, head_num, drop_out
class SelfAttention(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        """
        假设 hidden_size = 128, num_attention_heads = 8, dropout_prob = 0.2
        即隐层维度为128，注意力头设置为8个
        """
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:  # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        # 参数定义
        self.num_attention_heads = num_attention_heads  # 8
        self.attention_head_size = int(hidden_size / num_attention_heads)  # 16  每个注意力头的维度
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)
        # all_head_size = 128 即等于hidden_size, 一般自注意力输入输出前后维度不变

        # query, key, value 的线性变换（上述公式2）
        self.query = nn.Linear(hidden_size, self.all_head_size)  # 128, 128
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # dropout
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)  # [bs, 8, seqlen, 16]

    def forward(self, hidden_states, attention_mask):
        # eg: attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])  shape=[bs, seqlen]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [bs, 1, 1, seqlen] 增加维度
        attention_mask = (1.0 - attention_mask) * -10000.0  # padding的token置为-10000，exp(-1w)=0

        # 线性变换
        mixed_query_layer = self.query(hidden_states)  # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(hidden_states)  # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(hidden_states)  # [bs, seqlen, hid_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [bs, 8, seqlen, 16]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 计算query与title之间的点积注意力分数，还不是权重（个人认为权重应该是和为1的概率分布）
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [bs, 8, seqlen, seqlen]
        # 除以根号注意力头的数量，可看原论文公式，防止分数过大，过大会导致softmax之后非0即1
        attention_scores = attention_scores + attention_mask
        # 加上mask，将padding所在的表示直接-10000

        # 将注意力转化为概率分布，即注意力权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # 矩阵相乘，[bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
        context_layer = torch.matmul(attention_probs, value_layer)  # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [bs, seqlen, 128]
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer  # [bs, seqlen, 128] 得到输出


class AttentionBlock(nn.Module):
    def __init__(self, input_channel, output_channel, input_size, mask=None):
        super(AttentionBlock, self).__init__()
        self.attention = SelfAttention(hidden_size=input_size, num_attention_heads=2, dropout_prob=0.2)
        self.in_channel = input_channel
        self.out_channel = output_channel
        self.mask = mask.to(device)

    def attention_one(self, x):
        result = torch.zeros([x.shape[0], x.shape[2], x.shape[3]]).to(device)  # 方便注意力后累加
        for i in range(self.in_channel):
            single_channel = x[:, i, :, :].to(device)  # [batch_size, sen_len, emb_dim]
            single_channel = self.attention(single_channel, self.mask)
            result += single_channel
        result = result.unsqueeze(1)  # 增加一个维度[batch_size, 1, se_len, embedding_dim]
        return result

    def forward(self, x):
        result = []
        for i in range(self.out_channel):
            result.append(self.attention_one(x))
        result = torch.cat(result, dim=1)  # [batch_size, out_channel, se_len, embedding_dim]
        return result


# 文本特征提取模型
class ANNModel(nn.Module):
    def __init__(self,
                 n_vocab=21128,  # 词表
                 embed=128,  # embeding 维度
                 output_dim=2,  # 结果输出维度
                 batch_size=8,  # 用于生成 mask
                 ):
        super(ANNModel, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embed)
        self.mask = torch.zeros([batch_size, embed])
        self.attention_layer1 = AttentionBlock(input_channel=1, output_channel=3, input_size=embed, mask=self.mask)
        self.pool = nn.MaxPool2d(2)
        self.attention_layer2 = AttentionBlock(input_channel=3, output_channel=8, input_size=64,
                                               mask=torch.zeros([batch_size, 64]))

        self.layer_normal = nn.LayerNorm(128)
        self.layer_normal2 = nn.LayerNorm(64)
        self.sigmoid = nn.Sigmoid()
        # self.sigmoid2 = nn.Sigmoid()

        self.flatten = nn.Flatten()
        self.lieaner = nn.Linear(64 * 64 * 8, 100)
        self.lieaner2 = nn.Linear(100, 2)

        # 新增部分
        self.linear3 = nn.Linear(4, 100)
        self.linear4 = nn.Linear(100, output_dim)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # 加一层 channel

        x = self.attention_layer1(x)
        x = self.layer_normal(x)
        y = self.pool(x)
        y = self.sigmoid(y)

        z = self.attention_layer2(y)
        z = self.layer_normal2(z)
        z = self.sigmoid(z)

        z = self.flatten(z)
        z = self.lieaner(z)
        z = self.lieaner2(z)

        # 新增
        # output = torch.cat((z, star_level), 1)
        # output = self.linear3(output)
        # z = self.linear4(output)

        return z

# 用户特征提取模型
# class UserModel(nn.Module):
#     def __init__(self):
#         super(UserModel, self).__init__()
#         self.input_dim = config.input_dim_u
#         self.output_dim = config.output_dim_u
#         self.linear1 = nn.Linear(self.input_dim, 20)
#         self.linear2 = nn.Linear(20, self.output_dim)
#
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.linear2(x)
#
#         return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.in_dim = config.output_dim_u + config.output_dim_g + config.output_dim_a
        self.l1 = nn.Linear(self.in_dim, 100)
        self.l2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)

        return x


# 图+多层注意力模型
class GHAModel(nn.Module):
    def __init__(self,
                 graph_dim=64,  # 图 电影表示的维度
                 ANN_dim=128,  # 文本特征输出维度
                 uf_dim=10,  # 用户特征输出维度
                 ):
        super(GHAModel, self).__init__()
        # 外部知识图模型
        self.knowledge_graph = KnowledgeGraph(config.vocab_size, config.A, config.x, config.output_dim_g,
                                              config.embeding_dimension_g, config.hidden_size_g, config.n_layer_g,
                                              config.pretrained_lstm)
        # 文本特征提取模型
        self.ANN = ANNModel(config.vocab_size, config.embeding_dimension_a, config.output_dim_a, config.batch_size)
        # 用户特征模型
        # self.user_feature_model = UserModel()
        # 分类器模型
        self.classifier = Classifier()

    def forward(self, text, user_features, movie_ids):
        """

        :param text: 评论文本
        :param user_features: 用户特征
        :param movie_ids: 电影序号
        :return:
        """
        text_feature = self.ANN(text)
        all_feature = text_feature
        if user_features:   # 使用用户特征
            user_feature = self.user_feature_model(user_features)
            all_feature = torch.cat((text_feature, user_feature), 1)

        if movie_ids:   # 使用图特征
            graph_feature = self.knowledge_graph(movie_ids)
            all_feature = torch.cat((all_feature, graph_feature), 1)

        result = self.classifier(all_feature)
        return result




# https://www.cnblogs.com/jfdwd/p/11445135.html
if __name__ == '__main__':
    x = torch.randint(100, [5, 128])
    # mask = torch.zeros([5, 128])
    # model_attention = ModelWithAttention()
    # # y = model_attention(x, mask)
    # self_attention = SelfAttention(128, 8, 0.2)
    # emb = nn.Embedding(6000, 128)
    # x = torch.zeros([5, 1, 128, 128])
    # # x = emb(x)
    # attention_block = AttentionBlock(1, 3, 128, mask)
    # y = attention_block(x)
    # print(y.shape)
    writer = SummaryWriter('./logs')

    x = torch.randint(100, [8, 128]).to(device)
    model = GHAModel().to(device)
    # y = model(x)

    x = torch.tensor([x, [1, 0]])

    writer.add_graph(model, x)
    writer.close()
    # nn.MultiheadAttention(x, 8)
