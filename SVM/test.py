import pandas as pd
import numpy as np
import math
"""
# 读取整个csv文件
csv_data = pd.read_csv("/Users/frank/Desktop/2016.csv",encoding = "GB2312",low_memory=False)

# 读取指定列索引字段的数据
csv_data = pd.read_csv("/Users/frank/Desktop/2016.csv", usecols=['学号', '毕业去向'],encoding = "GB2312",low_memory=False)

# 将修改完的csv的文件保存到新的路径下
csv_data.to_csv('/Users/frank/Desktop/demo.csv',encoding='utf_8_sig')
"""

# csv_data = pd.read_csv("/Users/frank/Desktop/grants.csv",encoding="GB2312",low_memory=False)
# csv_data = pd.read_csv("/Users/frank/Desktop/grants.csv",usecols=["学号","奖助学金类别"],encoding="GB2312",low_memory=False)
# csv_data.to_csv("/Users/frank/Desktop/demo.csv",encoding="utf_8_sig")

# dd = pd.read_csv("/Users/frank/Desktop/demo.csv", usecols=['学号', '毕业去向'],encoding = "utf_8_sig",low_memory=False)
# df = pd.read_csv("/Users/frank/Desktop/2016.csv",usecols=["学号","奖助学金类别"],encoding="utf_8_sig",low_memory=False)
# dd['学号'] = pd.to_numeric(dd['学号'], errors='coerce')
# df['学号'] = pd.to_numeric(df['学号'], errors='coerce')
# data=pd.merge(dd,df,how="left")
# data.to_csv("/Users/frank/Desktop/all.csv",encoding="utf_8_sig")
"""
dd=pd.read_csv("/Users/frank/Desktop/2016zl.csv",usecols=["学号","就业状况"],encoding="GB2312",low_memory=False)
df=pd.read_csv("/Users/frank/Desktop/all.csv",encoding="utf_8_sig",low_memory=False)
dd["学号"]=pd.to_numeric(dd["学号"],errors="coerce")
df['学号']=pd.to_numeric(df['学号'], errors='coerce')
data=pd.merge(df,dd,how="left")
data.to_csv("/Users/frank/Desktop/all.csv",encoding="utf_8_sig")
"""



class SVM():
    def __init__(self, C, kernel, kernel_arg, e=0.001):
        '''
                                    kernel_arg
        kernel的类型: 'linear':     1
                    'poly':        d(d>1且为整数)
                    'gaussian':    σ(σ>0)
                    'lapras':      σ(σ>0)
                    'sigmoid':     beta,theta(beta>0,theta<0)
        kernel_arg若不符合要求将按照默认参数进行计算
        C为目标函数非线性部分的权重
        e为误差

        '''
        self.kernel = kernel
        self.kernel_arg = kernel_arg
        self.C = C
        self.e = e
        self.bias = 0

    def kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            if isinstance(self.kernel_arg, int) == False:
                self.kernel_arg = 2
            return np.dot(x1, x2)**self.kernel_arg
        elif self.kernel == 'gaussian':
            if isinstance(self.kernel_arg, float) == False:
                self.kernel_arg = 0.5
            return math.exp(-np.linalg.norm(x1 - x2)**2 / (2 * self.kernel_arg**2))
        elif self.kernel == 'lapras':
            if isinstance(self.kernel_arg, float) == False:
                self.kernel_arg = 0.5
            return math.exp(-np.linalg.norm(x1 - x2) / self.kernel_arg)
        elif self.kernel == 'sigmoid':
            if len(self.kernel_arg) != 2:
                self.kernel_arg = [0.5, -0.5]
            if self.kernel_arg[0] <= 0:
                self.kernel_arg[0] = 0.5
            if self.kernel_arg[1] >= 0:
                self.kernel_arg[1] = 0.5
            return math.tanh(self.kernel_arg[0] * np.dot(x1, x2) + self.kernel_arg[1])

    def fit(self, train_x, train_y, max_iter=1000):
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        self.alpha = np.zeros(train_x.shape[0])
        iter = 0

        while(iter < max_iter):
            print('iter = {}'.format(iter))
            index1, index2 = self.SMO_get_alpha()
            if index1 == -1:
                print('结束迭代, iter = {}'.format(iter))
                break
            train_result = self.SMO_train(index1, index2)
            if train_result == True:
                print('结束迭代, iter = {}'.format(iter))
                break
            iter += 1

    def SMO_get_alpha(self):
        for i in range(self.alpha.shape[0]):
            if 0 < self.alpha[i] < self.C:
                if self.train_y[i] * self.f(self.train_x[i]) != 1:
                    index2 = self.choose_another_alpha(i)
                    return i, index2
        for i in range(self.alpha.shape[0]):
            if self.alpha[i] == 0:
                if self.train_y[i] * self.f(self.train_x[i]) < 1:
                    index2 = self.choose_another_alpha(i)
                    return i, index2
            elif self.alpha[i] == self.C:
                if self.train_y[i] * self.f(self.train_x[i]) > 1:
                    index2 = self.choose_another_alpha(i)
                    return i, index2
        return -1, -1

    def f(self, x):
        result = 0
        for i in range(self.alpha.shape[0]):
            result += self.alpha[i] * self.train_y[i] * \
                self.kernel_function(self.train_x[i], x)
        return result + self.bias

    def error(self, index):
        return self.f(self.train_x[index]) - self.train_y[index]

    def choose_another_alpha(self, index):
        result_index = 0
        temp_diff_error = 0
        for i in range(self.alpha.shape[0]):
            diff_error = np.abs(self.error(index) - self.error(i))
            if diff_error > temp_diff_error:
                temp_diff_error = diff_error
                result_index = i
        return result_index

    def SMO_train(self, index1, index2):
        old_alpha = self.alpha.copy()
        x1 = self.train_x[index1]
        y1 = self.train_y[index1]
        x2 = self.train_x[index2]
        y2 = self.train_y[index2]

        eta = self.kernel_function(
            x1, x1) + self.kernel_function(x2, x2) - 2 * self.kernel_function(x1, x2)
        alpha2 = old_alpha[index2] + y2 * \
            (self.error(index1) - self.error(index2)) / eta

        if y1 != y2:
            L = max(0, old_alpha[index2] - old_alpha[index1])
            H = min(self.C, self.C + old_alpha[index2] - old_alpha[index1])
        else:
            L = max(0, old_alpha[index1] + old_alpha[index2] - self.C)
            H = min(self.C, old_alpha[index1] + old_alpha[index2])

        if alpha2 > H:
            alpha2 = H
        elif alpha2 < L:
            alpha2 = L

        alpha1 = old_alpha[index1] + y1 * y2 * (old_alpha[index2] - alpha2)

        self.alpha[index1] = alpha1
        self.alpha[index2] = alpha2

        b1 = -self.error(index1) \
            - y1 * self.kernel_function(x1, x1) * (alpha1 - old_alpha[index1]) \
            - y2 * self.kernel_function(x1, x2) * (alpha2 - old_alpha[index2]) \
            + self.bias

        b2 = -self.error(index2) \
            - y1 * self.kernel_function(x1, x2) * (alpha1 - old_alpha[index1]) \
            - y2 * self.kernel_function(x2, x2) * (alpha2 - old_alpha[index2]) \
            + self.bias
        if 0 < alpha1 < self.C:
            self.bias = b1
        elif 0 < alpha2 < self.C:
            self.bias = b2
        else:
            self.bias = (b1 + b2) / 2

        print('E = {}'.format(np.linalg.norm(old_alpha - self.alpha)))
        if np.linalg.norm(old_alpha - self.alpha) < self.e:
            return True
        else:
            return False

    def predict_one(self, x):
        if self.f(x) > 0:
            return 1
        else:
            return -1

    def predict(self, x_group):
        return np.array([self.predict_one(x) for x in x_group])


from sklearn import svm               # svm函数需要的
import numpy as np                    # numpy科学计算库
from sklearn import model_selection
import matplotlib.pyplot as plt       # 画图的库
from sklearn import metrics

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn import svm
import matplotlib as mpl
import matplotlib.pyplot as plt


def standardize(X):
    """特征标准化处理
    Args:
        X: 样本集
    Returns:
        标准后的样本集
    """
    m, n = X.shape
    # 归一化每一个特征
    for j in range(n):
        features = X[:,j]
        meanVal = features.mean(axis=0)
        std = features.std(axis=0)
        if std != 0:
            X[:, j] = (features-meanVal)/std
        else:
            X[:, j] = 0
    return X
def normalize(X):
    """Min-Max normalization     sklearn.preprocess 的MaxMinScalar
    Args:
        X: 样本集
    Returns:
        归一化后的样本集
    """
    m, n = X.shape
    # 归一化每一个特征
    for j in range(n):
        features = X[:,j]
        minVal = features.min(axis=0)
        maxVal = features.max(axis=0)
        diff = maxVal - minVal
        if diff != 0:
           X[:,j] = (features-minVal)/diff
        else:
           X[:,j] = 0
    return X


#def iris_type(s):
     #it={b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
     #return it[s]
path="/Users/frank/Desktop/shujuyuan.csv"
data=np.loadtxt(path,
                 dtype=float,
                 delimiter=",",
                 )
X, y = np.split(data, (7,), axis=1)
x = X[:, 0:7]
#normalize(x)
standardize(x)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.2)
# def read_data(test_data='/Users/frank/Desktop/shuju8.csv', n=0, label=1):
#     '''
#     加载数据的功能
#     n:特征数据起始位
#     label：是否是监督样本数据
#     '''
#     csv_reader = csv.reader(open(test_data))
#     data_list = []
#     for one_line in csv_reader:
#         data_list.append(one_line)
#     x_list = []
#     y_list = []
#     for one_line in data_list[1:]:
#         if label == 1:
#             y_list.append(int(one_line[-1]))  # 标志位
#             one_list = [float(o) for o in one_line[n:-1]]
#             x_list.append(one_list)
#         else:
#             one_list = [float(o) for o in one_line[n:]]
#             x_list.append(one_list)
#     return x_list, y_list
#
#
# def split_data(data_list, y_list, ratio=0.30):
#     '''
#     按照指定的比例，划分样本数据集
#     ratio: 测试数据的比率
#     '''
#     X_train, X_test, y_train, y_test = train_test_split(data_list, y_list, test_size=ratio, random_state=50)
#     print
#     '--------------------------------split_data shape-----------------------------------'
#     print
#     len(X_train), len(y_train)
#     print
#     len(X_test), len(y_test)
#     return X_train, X_test, y_train, y_test
#
#
# x_list, y_list=read_data()
# """划分为训练集和测试集及label文件"""
# x_train, x_test, y_train, y_test=split_data(x_list,y_list)
clf = svm.SVC(kernel='rbf',                      # 核函数
               gamma=4.255964812896852,
             decision_function_shape='ovo',     # one vs one 分类问题
              C=9.76411594501443)
clf.fit(x_train, y_train)                        # 训练
print("训练集准确率：",clf.score(x_train, y_train))
y_train_hat=clf.predict(x_train)
y_train_1d=y_train.reshape((-1))
recall = recall_score(y_train, y_train_hat)
print('训练集Recall:\t', recall)
print('训练集f1 score:\t', f1_score(y_train, y_train_hat))
comp=zip(y_train_1d,y_train_hat)
print(list(comp))

print("预测准确率：",clf.score(x_test,y_test))
y_test_hat=clf.predict(x_test)
y_test_1d=y_test.reshape((-1))
recall = recall_score(y_test, y_test_hat)
print('测试集Recall:\t', recall)
print('测试集f1 score:\t', f1_score(y_test, y_test_hat))
comp=zip(y_test_1d,y_test_hat)
print(list(comp))
plt.figure()
plt.subplot(121)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train.reshape((-1)), edgecolors='k',s=50)
plt.subplot(122)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train_hat.reshape((-1)), edgecolors='k',s=50)
plt.show()


