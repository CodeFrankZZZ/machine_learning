# iris_keras_dnn.py
# Python 3.5.1, TensorFlow 1.6.0, Keras 2.1.5
# ========================================================
# 导入模块
import os
import numpy as np
import keras as K
import tensorflow as tf
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 读取CSV数据集，并拆分为训练集和测试集
# 该函数的传入参数为CSV_FILE_PATH: csv文件路径
def load_data(CSV_FILE_PATH):
    IRIS = pd.read_csv(CSV_FILE_PATH)
    target_var = 'name'  # 目标变量
    # 数据集的特征
    features = list(IRIS.columns)
    features.remove(target_var)
    # 目标变量的类别
    Class = IRIS[target_var].unique()
    # 目标变量的类别字典
    Class_dict = dict(zip(Class, range(len(Class))))
    # 增加一列target, 将目标变量进行编码
    IRIS['target'] = IRIS[target_var].apply(lambda x: Class_dict[x])
    # 对目标变量进行0-1编码(One-hot Encoding)
    lb = LabelBinarizer()
    lb.fit(list(Class_dict.values()))
    transformed_labels = lb.transform(IRIS['target'])
    y_bin_labels = []  # 对多分类进行0-1编码的变量
    for i in range(transformed_labels.shape[1]):
        y_bin_labels.append('y' + str(i))
        IRIS['y' + str(i)] = transformed_labels[:, i]
    # 将数据集分为训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(IRIS[features], IRIS[y_bin_labels], \
                                                        train_size=0.7, test_size=0.3, random_state=0)
    return train_x, test_x, train_y, test_y, Class_dict
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
def main():

    # 0. 开始
    print("\nIris dataset using Keras/TensorFlow ")
    np.random.seed(4)
    tf.set_random_seed(13)

    # 1. 读取CSV数据集
    print("Loading Iris data into memory")
    # CSV_FILE_PATH = '/Users/frank/Desktop/shuju2222.csv'
    # train_x, test_x, train_y, test_y, Class_dict = load_data(CSV_FILE_PATH)
    path = "/Users/frank/Desktop/ssshuju1111.csv" #训练用数据
    IRIS = pd.read_csv("/Users/frank/Desktop/shuju2222.csv") #匹配label用数据
    target_var = 'name'  # 目标变量
    # 数据集的特征
    features = list(IRIS.columns)
    features.remove(target_var)
    # 目标变量的类别
    Class = IRIS[target_var].unique()
    # 目标变量的类别字典
    Class_dict = dict(zip(Class, range(len(Class))))
    # 增加一列target, 将目标变量进行编码
    IRIS['target'] = IRIS[target_var].apply(lambda x: Class_dict[x])
    data = np.loadtxt(path,
                      dtype=float,
                      delimiter=",",
                      )
    X, y = np.split(data, (7,), axis=1)
    x = X[:, 0:7]
    normalize(x)
    train_x, test_x, train_y, test_y = model_selection.train_test_split(x, y, random_state=1, test_size=0.2)
    #2. 定义模型
    init = K.initializers.glorot_uniform(seed=1)
    simple_adam = K.optimizers.Adam()
    model = K.models.Sequential()
    model.add(K.layers.Dense(units=5, input_dim=7, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=6, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=4, kernel_initializer=init, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])

    # 3. 训练模型
    b_size = 1
    max_epochs = 100
    print("Starting training ")
    h = model.fit(train_x, train_y, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)
    print("Training finished \n")

    # 4. 评估模型
    eval = model.evaluate(test_x, test_y, verbose=0)
    print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" \
          % (eval[0], eval[1] * 100) )

    #5. 使用模型进行预测
    np.set_printoptions(precision=4)
    unknown = np.array([[2.3, 0, 1, 0,0,1,0]], dtype=np.float32)
    predicted = model.predict(unknown)
    print("Using model to predict species for features: ")
    print(unknown)
    print("\nPredicted softmax vector is: ")
    print(predicted)
    species_dict = {v:k for k,v in  Class_dict.items()}
    print("\nPredicted species is: ")
    if species_dict[np.argmax(predicted)]==0:
        print("出国")
    elif  species_dict[np.argmax(predicted)]==1:
        print("工作")
    elif species_dict[np.argmax(predicted)]==2:
        print("推免")
    elif species_dict[np.argmax(predicted)]==3:
        print("考研")


main()
