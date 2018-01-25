# -*- coding:utf-8 -*-
#coding:utf-8
""""
PYSPARK_PYTHON=C:\Python27\python.exe;
PYTHONPATH=D:\spark\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7\python;
PYTHONUNBUFFERED=1;
SPARK_HOME=D:\spark\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7
"""

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.rdd import *
from time import time
from elm import ELMClassifier
from random_hidden_layer import SimpleRandomHiddenLayer
import numpy as np
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


def iris_label_transfor(s):
    if s == "Iris-setosa":
        return 0
    elif s=="Iris-versicolor":
        return 1
    elif s=="Iris-virginica":
        return 2


float_feature=lambda x:[float(i) for i in x]
def yuchuli_data(x):
    x[:feature_num]=[float(i) for i in x[0:feature_num]]
    x[class_index]=iris_label_transfor(x[class_index])# iris 使用
    # x[class_index]=int(x[class_index])  #pokerhand 使用
    return x

if __name__=="__main__":
    conf = SparkConf().setMaster("local[4]").setAppName("ActiveLearn")#spark://master:7077
    sc = SparkContext(conf=conf)
    all_data_path = "D:\\pysparktest\\ELM-Spark-ActiveLearning\\irisdata"
    #all_data_path = "hdfs://master:9000/Alspark/data/irisdata"

    #all_data_path = "D:\\pysparktest\\ELM-Spark-ActiveLearning\\pokerhandtesting.data"
    # all_data_path = "hdfs://master:9000/Alspark/data/pokerhandtesting.data"
    #save_file= ""
    hiddenLayer_num = 20  # ELM隐含层层数
    activation = "sigmoid"  #激活函数sigmoid hardlim  tribas  tanh  sine
    select_num = 5  #选择最小样例个数   iris选择10个  pokerhand 选择500
    feature_num =4  #数据集中属性个数 从0列到  iris4  pokerhand 10
    class_index = 5-1 #类标属性在第几列   iris 4  pokerhand 11

    frist_filter =0.7  # 初步过滤的概率值 小frist_filter值得将被选出
    #读入整个数据
    all_data = sc.textFile(all_data_path)
    print "all_data.first ：",all_data.first()
    print "feature_num:%d"%feature_num

    #划分数据集 将属性转化为float值  将类标转化为0 1 2   #[5.1, 3.5, 1.4, 0.2, 0]
    split_data=all_data.map(lambda x:x.split(",")).map(lambda x:yuchuli_data(x))#.map(lambda x:[float_feature(x[0:4]),iris_label_transfor(x[4])])
    test_data = split_data.sample(False, 0.1, 87)
    test_array = test_data.map(lambda x:x[0:feature_num]).collect()
    test_label = test_data.map(lambda x:x[class_index]).collect()
    print "test_data 1%：",sum(test_label)
    # 将数据集划分为训练集 与 无类标的数据集  3比7
    train_data,unlabel_data= split_data.randomSplit([0.2,0.8],7)# train_data.first() [4.6, 3.1, 1.5, 0.2, 0]   unlabel_data.first() #[5.1, 3.5, 1.4, 0.2, 0]
    # 创建隐含层
    srh = SimpleRandomHiddenLayer(activation_args=activation, n_hidden=hiddenLayer_num)
    # 创建ELM分类器
    elmc = ELMClassifier(hidden_layer=srh)
    for i in range(1):
        print "-"*20+"%d train"%(i+1)+"-"*20
        # 训练集属性值不含类标 用于训练
        train_ = train_data.map(lambda x:x[0:feature_num]) #[4.6, 3.1, 1.5, 0.2]
        #将无类标数据属性集合 不含类标用于预测
        unlabel_ = unlabel_data.map(lambda x:x[0:feature_num]) #[5.1, 3.5, 1.4, 0.2]

        # 训练集 类标集合
        train_label =train_data.map(lambda x:x[class_index]).collect()
        ###############  ELM 训练  #############

        train_array = train_.collect()
        print "train_array_num:",len(train_array)
        #训练分类器
        elmc.fit(train_array,train_label)
        pred_class=elmc.predict_class(test_array)

        #分类精度
        soc = accuracy_score(pred_class,test_label)
        print "test_soc:",soc
        #软最大化函数
        ruanzuida = lambda x: np.exp(x)/sum((np.exp(x)))
        def enry(x):
            tmp=ruanzuida(x)
            result = -sum(ruanzuida(x)*np.log(ruanzuida(x)))
            return result
        #对无类标的数据集进行预测 每个样例的到一个向量  然后进行软最大化处理
        def partitionfun(it):
            a= []
            try:
                while True:
                    a.append(it.next())
            except StopIteration:
                print len(a)
                return elmc.predict(a)
        print unlabel_.count()
        ruanhua_ = unlabel_.mapPartitions(lambda x:partitionfun(x))  #partitionfun 写的函数有错 只得到6个结果
        print ruanhua_.count()
        ruanhua_result =ruanhua_.map(lambda x:enry(x))
        print ruanhua_result.collect()
        # ruanhua_result =unlabel_.flatMap(lambda x:elmc.predict(x)).map(lambda x:enry(x))
        #[1.9240548462966107e-09, 0.041402756752319046, 0.77651983843494121, 0.42876278929411704, 6.9036877387924139e-09,
        #对软最大化处理后的数据进行过滤 为了使排序的样例变少  所以先进行初步过滤
        filter_result = ruanhua_result.filter(lambda x:x>frist_filter).collect()
        #对过滤后的样例排序 得到第10小的数值作为阈值
        try:
            yuzhi = sorted(filter_result,reverse=True)[select_num]
        except IndexError:
            print "select_num"
            yuzhi = sorted(filter_result,reverse=True)[select_num//2]
            print "break this for"
            print "yuzhi",yuzhi
        else:
            print "yuzhi:", yuzhi

        #将大数据集与软最大化后的数据连接 使之变为([4.6, 3.1, 1.5, 0.2,0],0.78780751565075124)
        #再进行以阈值过滤出大数据中所有满足小于阈值的子数据集之后转化为[4.6, 3.1, 1.5, 0.2,0]样式数据集
        add_rdd = unlabel_data.zip(ruanhua_result).filter(lambda a:a[1]>yuzhi).map(lambda x:x[0])
        print add_rdd.collect()
        # print "每次选出的样例:"
        # print add_rdd.collect()
        #将新的数据子集 合并到 训练集中 在循环进行训练
        train_data=add_rdd.union(train_data)

    sc.stop()




