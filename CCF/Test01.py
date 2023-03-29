import numpy as np
import json
def load_data():
    datafile='D:/python-make/paddle/work/housing.data'
    data = np.fromfile(datafile,sep=' ')
    # 读入之后的数据被转化成1维array，其中array的第0-13项是第一条数据，第14-27项是第二条数据，以此类推....
    # 这里对原始数据做reshape，变成N x 14的形式,
    feature_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIX','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
    feature_num = len(feature_names)
    data = data.reshape([data.shape[0]//feature_num,feature_num])
    #查看数据
    # x=data[0]
    # print(x.shape)
    # print(x)
    ratio = 0.8
    offset = int(data.shape[0]*ratio)
    training_data=data[:offset]
    # 80%的数据作为训练集，20%作为测试集，共404个样本，每个样本13个特征，1个预测值
    # 计算train训练集的最大最小值
    maximums,minimums = training_data.max(axis=0),training_data.min(axis=0)
    for i in range(feature_num):
        data[:,i]=(data[:,i]-minimums[i])/(maximums[i]-minimums[i])
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data,test_data
training_data,test_data=load_data()
x=training_data[:,:-1]
y=training_data[:,-1:]
print(x[0])
print(y[0])