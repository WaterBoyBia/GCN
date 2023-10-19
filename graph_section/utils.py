import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import re


def movie_name_to_dic(movie_dir, save_path):
    """
    将所有有评论的电影的名字转成 序号 一一对应
    :param movie_dir: 电影评论保存目录
    :param save_path: 保存 电影名-序号 字典 的文件
    :return:
    """
    id_list = []
    name_list = []
    year_list = [os.path.join(movie_dir, i) for i in os.listdir(movie_dir)]
    for one_year_data in year_list:
        movie_name_lsit = os.listdir(one_year_data)
        for movie_name in movie_name_lsit:
            id = movie_name.split('_')[0]
            name = movie_name.split('_')[-1].split('.')[0]
            # 如果能正常读文件，并且里面成功写入东西，则为有效电影
            try:
                movie_path = os.path.join(one_year_data, movie_name)
                df = pd.read_csv(movie_path)
                if len(df) == 0:
                    print(f"{one_year_data}--{name}里面没有数据")
                    continue
                id_list.append(id)
                name_list.append(name)
            except:
                print(f'{one_year_data}-{name}打开失败')
                continue
    # print(id_list, name_list)
    indx_list = [i for i in range(len(id_list))]
    df = pd.DataFrame({
        'index': indx_list,
        'id': id_list,
        'name': name_list
    })
    df.to_csv(save_path, encoding="utf_8_sig")


def maoyan_label_movie_to_dic(path, save_path):
    """
    将打过标签的总表里面的电影名字整理成字典 {name: index}
    :param path:
    :param save_path:
    :return:
    """
    df = pd.read_csv(path)
    names = set()
    for i in df.iloc[:, 2]:
        names.add(i)

    save_df = pd.DataFrame(
        {'name': list(names)}
    )
    save_df.to_csv(save_path, encoding="utf_8_sig")


def get_movie_type(movie_dir):
    """
    查看一共有多少电影类型
    :return:
    """
    types = set()
    year_list = [os.path.join(movie_dir, i) for i in os.listdir(movie_dir)]
    for one_year_data in year_list:
        df = pd.read_csv(one_year_data)
        for i in df.类型:
            # for i in df.iloc[:, 9]:
            try:
                x = i.split(',')
                types.update(x)
            except:
                continue
    types = list(types)
    df = pd.DataFrame({
        '类型': types
    })
    df.to_csv('电影类型.csv', encoding="utf_8_sig")


def dic_movie2comments(dic='电影名单.csv'):
    """
    读出打过标签的表的路径--》{名字: 路径}
    :param save_path:
    :param dic:
    :param comments_dir:
    :return:
    """
    dic_df = pd.read_csv(dic)
    movies = dic_df.name.to_list()  # 打过标签的电影名字

    # 读长评文件夹
    # 形成{名字：路径}字典
    year_dir = [os.path.join(i) for i in os.listdir()]
    long_comments_table = {}    #{名字：路径}字典
    name_list = []
    file_path_lsit = []
    for one_year in year_dir:
        names = [i.split('_')[-1].split('.')[0] for i in os.listdir(one_year)]
        file_paths = [os.path.join(one_year, i) for i in os.listdir(one_year)]
        name_list.extend(names)
        file_path_lsit.extend(file_paths)
    for x, y in zip(name_list, file_path_lsit):
        if x in movies:     # 如果这部电影打过标签
            long_comments_table[x] = y

    return long_comments_table


# 将所有打过标签的长评写一张表
def long_comments_tables2one( dic='电影名单.csv', save_path='maoyan_long_comments.csv'):
    """
    将所有打过标签的长评写一张表
    :param save_path:
    :param dic:
    :param comments_dir:
    :return:
    """
    long_comments_table = dic_movie2comments(dic)     # 电影名字，长评 字典
    dic_df = pd.read_csv(dic)
    movies = dic_df.name.to_list()  # 所有的电影名字

    # 将打过标签的电影的长评统一写在一张大表里面
    df_list = []
    for name in movies:
        path = long_comments_table.get(name)
        try:
            df = pd.read_csv(path, index_col=0)
        except ValueError as e:
            print(path, name)
            raise e
        df_list.append(df)
    all_df = pd.concat(df_list, ignore_index=True) # 将所有表拼在一起
    all_df.to_csv(save_path, encoding="utf_8_sig")


# 图构建
# 同一类型电影双向连接;评论该电影的长评双向连接；长评单向连接内容里面提到的电影
class MyGraph():
    def __init__(self):
        self.G = nx.DiGraph()
        self.node_type_dic = {}  # 这个字典记录 电影节点和类型 对应

    def add_movie_node(self, movie_index: int, types: list):
        """
        同一类型的电影相互双向连接
        :param movie_index: 电影对应的序号
        :param types: 一部电影的分类
        :return:
        """
        # 添加点
        self.G.add_node(movie_index)
        # 添加边
        for i in types:
            for key, val in self.node_type_dic.items():
                for x in val:
                    if i == x:  # 如果是同一个类型
                        # 双向连接
                        self.G.add_edge(key, movie_index)
                        self.G.add_edge(movie_index, key)
        self.node_type_dic[movie_index] = types

    def add_long_comment_node(self, comment_index: int, movie_index: int, referer_movies: list):
        """
        添加长评节点
        评论该电影的长评和电影双向连接
        与文中提到的电影单向连接
        :param comment_index: 评论的index
        :param movie_index: 评论的电影index
        :param referer_movies: 文中提到电影的index
        :return:
        """
        # 双向链接评论电影
        self.G.add_edge(comment_index, movie_index)
        self.G.add_edge(movie_index, comment_index)
        for i in referer_movies:
            # 单向链接提到电影
            self.G.add_edge(comment_index, i)

    def draw_graph(self):
        nx.draw_networkx(self.G)
        plt.show()

    def print_adjacency_matrix(self):
        A = np.array(nx.adjacency_matrix(self.G).todense())
        print(A)

    def get_adjacency_matrix(self):
        A = np.array(nx.adjacency_matrix(self.G).todense())
        return A


# 获得长评电影的类型
# 长评内容提到的其他电影
class LongComments():
    def __init__(self, movie_index_dic: dict, movie_type_dic: dict, root=None, comment_count=20, raw_data=None):
        """

        :param movie_index_dic: {名字：序号}
        :param movie_type_dic: {类型：序号}
        :param root: 长评文件
        :param raw_data: 读了长评文件后的原始数据，和root参数二选一
        """
        self.id = ''
        self.name = ''
        self.df = None
        self.movie_index_dic = movie_index_dic
        self.movie_type_dic = movie_type_dic
        self.comment_count = comment_count  # 采用一部电影长评数
        if root:
            try:
                self.df = pd.read_csv(root)
                if len(self.df) > self.comment_count:
                    self.df = self.df.iloc[:self.comment_count, :]
                self.id = self.df.电影id[0]
                self.name = self.df.电影名称[0]
            except:
                raise
        if raw_data:
            self.df = raw_data
            if len(self.df) > self.comment_count:
                self.df = self.df.iloc[:self.comment_count, :]

    def get_include_name(self):
        """
        获取评论里面电影名字
        :return:
        """
        comments = self.df.评论内容.to_list()
        pattern = re.compile('《(.*?)》')
        include_list = []  # 长评里面的电影名字列表
        include_index_list = []  # 将电影名字换成序号
        for comment in comments:
            include_movies = pattern.findall(comment)
            include_list.append(set(include_movies))

        # 转成对应的id
        for movie_names in include_list:
            index_lsit = []  # 一条长评里面的电影名字序号
            for i in movie_names:
                index = self.movie_index_dic.get(i, 'have not')
                if index != "have not":
                    index_lsit.append(index)
            include_index_list.append(index_lsit)

        return include_index_list

    def get_movie_type(self, movie_dir='../../../爬虫/maoyan/五年电影名单'):
        """
        获取当前电影类型
        :param movie_dir:
        :return:
        """
        year_list = [os.path.join(movie_dir, i) for i in os.listdir(movie_dir)]
        all_data = []  # 将五张表数据合起来
        for one_year_data in year_list:
            df = pd.read_csv(one_year_data)
            all_data.append(df)
        all_df = pd.concat(all_data)
        # 获取一个位置的值
        movie_types = all_df.loc[all_df['名字'] == self.name].类型
        movie_types_list = movie_types.to_list()[0].split(',')  # todo

        #
        # movie_types_index_list = [self.movie_type_dic.get(i) for i in movie_types_list] # 类型对应序号
        movie_types_index_list = []
        for type_one in movie_types_list:
            index = self.movie_type_dic.get(type_one, 'have not')
            if index != "have not":
                movie_types_index_list.append(index)

        # 没有这个名字电影
        # if movie_types_index_list[0] == None:
        #     raise '没有这部电影'
        if len(movie_types_index_list) == 0:
            raise '没有这部电影'

        return movie_types_index_list

    def add2table(self, save_path):
        """
        将当前数据添加到一张表中
        :param save_path:
        :return:
        """
        print(self.name)
        self.df.to_csv(save_path, mode='a', encoding="utf_8_sig")

    def __len__(self):
        return len(self.df)



"""
获得邻接矩阵
"""
class MyAtrycx():
    def __init__(self, movie_dic_path, type_dic_path, long_comment_dir):
        self.my_graph = MyGraph()
        self.movie_dic = movie_dic_path
        self.type_dic = type_dic_path
        self.long_comment_dir = long_comment_dir
        self.movie_long_comment_dic = dic_movie2comments(movie_dic_path)    # 映射打过标签的电影{电影名：path}

    def add_node_eadge(self):
        movie_index_dic = {}    # {电影名：序号}
        type_index_dic = {}
        movie_dic = pd.read_csv(self.movie_dic)
        # for i in movie_dic.itertuples():
        #     movie_index_dic[i.name] = i.index
        for i in range(len(movie_dic)):
            movie_index_dic[movie_dic.name[i]] = movie_dic.index[i]
        type_dic = pd.read_csv(self.type_dic)
        # for i in type_dic.itertuples():
        #     type_index_dic[i.类型] = i[0]
        for i in range(len(type_dic)):
            type_index_dic[type_dic.类型[i]] = type_dic.index[i]

        movie_count = len(movie_index_dic)
        long_comments_count = 0
        for movie_name, path in self.movie_long_comment_dic.items():
            one_movie_long_comments = LongComments(movie_index_dic=movie_index_dic, movie_type_dic=type_index_dic, root=path)
            this_index = movie_index_dic.get(one_movie_long_comments.name)  # 当前电影序号
            include_name_list = one_movie_long_comments.get_include_name()  # 评论中包含的电影名字的序号
            movie_type_list = one_movie_long_comments.get_movie_type()  # 评论电影的类型

            # 把这张表里面的长评，集中写到另外一张总表
            one_movie_long_comments.add2table('猫眼长评总表.csv')

            self.my_graph.add_movie_node(this_index, movie_type_list)  # 同类型电影进行连接
            for names in include_name_list:
                comment_index = movie_count + long_comments_count
                long_comments_count += 1
                self.my_graph.add_long_comment_node(comment_index, this_index, names)   # 评论和电影连接；评论中提到的电影进行连接

        self.my_graph.draw_graph()


if __name__ == '__main__':
    # maoyan_label_movie_to_dic('../maoyan/data/总表.csv', '电影名单.csv')

    # # 测试图构建
    # movie_dic_path = '电影名单.csv'
    # type_dic_path = '电影类型.csv'
    # long_comment_dir = '../../../爬虫/douban/data/电影长评'
    #
    # my_test = MyAtrycx(movie_dic_path, type_dic_path, long_comment_dir)
    # my_test.add_node_eadge()
    # A = my_test.my_graph.get_adjacency_matrix()
    # print(A)


    ## 测试 画图
    # my_graph = MyGraph()
    # my_graph.add_movie_node(1, [1, 2, 3])
    # my_graph.add_movie_node(2, [1, 2])
    # my_graph.add_movie_node(3, [1, 3])
    # my_graph.add_movie_node(2, [2, 3])
    # my_graph.add_movie_node(4, [4, 5])
    # my_graph.add_long_comment_node(6, 1, [2, 3, 4])
    # my_graph.draw_graph()
    # my_graph.print_adjacency_matrix()

    # path = '../../爬虫/maoyan/data/原始数据'
    # path = '../../爬虫/maoyan/五年电影名单'
    # movie_name_to_dic(path, 'xx.csv')
    # get_movie_type(path)


    ## 测试 LongComment
    # movie_dic = pd.read_csv('./xx.csv')
    # movie_index_dic = {}
    # movie_type_dic = {}
    # for i in movie_dic.itertuples():
    #     movie_index_dic[i.name] = i.id
    # type_dic = pd.read_csv('./电影类型.csv')
    # for i in type_dic.itertuples():
    #     movie_type_dic[i.类型] = i[0]
    #
    # test_long_comment = LongComments(movie_index_dic, movie_type_dic,
    #                                  root='..\\..\\爬虫\douban\\data\\电影长评\\2020电影\\26754233_八佰.csv')
    # print(test_long_comment.get_include_name())
    # print(test_long_comment.get_movie_type())

    # 测试把长评放一起
    long_comments_tables2one()

    #
    # df = pd.read_csv('xx.csv')
    # for i in range(len(df)):
    #     print(df.index[i])
    #     print(df.name[i])
    # print(df.index)
