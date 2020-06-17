import pyprind
import time
from random import seed
from random import randint
from csv import reader

# 建立一棵CART树
'''试探分枝'''
def data_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

'''计算基尼指数'''
def calc_gini(groups, class_values):
    gini = 0.0
    total_size = 0
    for group in groups:
        total_size += len(group)
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        for class_value in class_values:
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (size / float(total_size)) * (proportion * (1.0 - proportion))
    return gini

'''找最佳分叉点'''
def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randint(0, len(dataset[0]) - 2)  # 往features添加n_features个特征（n_feature等于特征数的根号），特征索引从dataset中随机取
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            groups = data_split(index, row[index], dataset)
            gini = calc_gini(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}  # 每个节点由字典组成

'''多数表决'''
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

'''分枝'''
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del (node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)  # 叶节点不好理解
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)

'''建立一棵树'''
def build_one_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root

'''用一棵树来预测'''
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# 随机森林类
class randomForest:
    def __init__(self,trees_num, max_depth, leaf_min_size, sample_ratio, feature_ratio):
        self.trees_num = trees_num                # 森林的树的数目
        self.max_depth = max_depth                # 树深
        self.leaf_min_size = leaf_min_size        # 建立树时，停止的分枝样本最小数目
        self.samples_split_ratio = sample_ratio   # 采样，创建子集的比例（行采样）
        self.feature_ratio = feature_ratio        # 特征比例（列采样）
        self.trees = list()                       # 森林

    '''有放回的采样，创建数据子集'''
    def sample_split(self, dataset):
        sample = list()
        n_sample = round(len(dataset) * self.samples_split_ratio)
        while len(sample) < n_sample:
            index = randint(0, len(dataset) - 2)
            sample.append(dataset[index])
        return sample

    '''建立随机森林'''
    def build_randomforest(self, train):
        max_depth = self.max_depth
        min_size = self.leaf_min_size
        n_trees = self.trees_num
        n_features = int(self.feature_ratio * (len(train[0])-1))#列采样，从M个feature中，选择m个(m<<M)
        for i in range(n_trees):
            sample = self.sample_split(train)
            tree = build_one_tree(sample, max_depth, min_size, n_features)
            self.trees.append(tree)
        return self.trees

    '''随机森林预测的多数表决'''
    def bagging_predict(self, onetestdata):
        predictions = [predict(tree, onetestdata) for tree in self.trees]
        return max(set(predictions), key=predictions.count)

    '''计算建立的森林的精确度'''
    def accuracy_metric(self, testdata):
        correct = 0
        for i in range(len(testdata)):
            predicted = self.bagging_predict(testdata[i])
            if testdata[i][-1] == predicted:
                correct += 1
        return correct / float(len(testdata)) * 100.0


# 数据处理
'''导入数据'''
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

'''划分训练数据与测试数据'''
def split_train_test(dataset, ratio=0.2):
    #ratio = 0.2  # 取百分之二十的数据当做测试数据
    num = len(dataset)
    train_num = int((1-ratio) * num)
    dataset_copy = list(dataset)
    traindata = list()
    while len(traindata) < train_num:
        index = randint(0,len(dataset_copy)-1)
        traindata.append(dataset_copy.pop(index))
    testdata = dataset_copy
    return traindata, testdata



# 测试


if __name__ == '__main__':
    seed(1)  # 每一次执行本文件时都能产生同一个随机数
    filename = '/Users/frank/Desktop/ssshuju1111.csv'
    dataset = load_csv(filename)
    traindata, testdata = split_train_test(dataset, ratio=0.2)
    max_depth = 20  # 调参（自己修改） #决策树深度不能太深，不然容易导致过拟合
    min_size = 1
    sample_ratio = 1
    trees_num = 20
    feature_ratio = 0.3
    myRF = randomForest(trees_num, max_depth, min_size, sample_ratio, feature_ratio)
    myRF.build_randomforest(traindata)
    acc = myRF.accuracy_metric(testdata[:-1])
    print('模型准确率：', acc, '%')
