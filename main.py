import pandas as pd
import numpy as np

#data = pd.read_csv('/Users/aquarius/Desktop/data.csv', header=False) 不要表头，默认是有的，如果是true的话第0行为表格
data = pd.read_csv('pre_0.csv').values #要加values转换为传统数组，因为是数组所以不算表头
# print(data[0,0:3]) #第0行，0-3列（不包括第三列）
df = data[:,2:3] #第三列切片（左闭右开）这样也是二维的
df2= data[:12:13]
print(df)
print(df.shape)
# print(np.mean(df)) #读取用pandas处理用numpy
# print(np.std(df)) #方差
# print(data.shape) #维度，（34353，4）表示行数列数
# print(len(data))

# #取前百分之80作为训练集，后百分之20作为测试集
# split = int(int(len(data) * 0.8)) #乘0.08之后不一定为整所以int一下
# trainSet = data[0:split,3:4] #取出切片
# testSet = data[split:,3:4]
# print(trainSet.shape)
# print(testSet.shape)
# trainMean=np.mean(trainSet)
# trainStd=np.std(trainSet)
# print(trainMean)
# print(trainStd)
#
# # 做归一化
# # tensor = (trainSet-trainMean)/(trainStd+0.10001) #除数不为0，一般给方差加很小的数
# # print(tensor)
#
# #写个归一化函数进行封装
# def z_score(df, mean, std):
#     out = (df-mean)/(std+0.00001)
#     return out
#
# tensor = z_score(trainSet,trainMean,trainStd)
# print(tensor)
#
# # 取样本，12个做训练12个做预测，每个样本是个一维的24个的数组，现在加一个维度，每一行代表一个样本
# #先计算一共能切分出多少个样本（总长度34535-一个样本的长度24+1）
# # 因此创建一个（样本总数，24）的数组代表共num个样本，每个样本24个数据
# sampleNum = 24
# num = len(trainSet)-sampleNum+1 #这里要用trainset而不是data的长度，因为只考虑训练集的，不然会多循环
# print(num)
# test = np.zeros([num, 24], dtype=np.float32)
# print(test.shape)
# print(test)
# #进行样本划分切片，放入刚新建的空数组中，要几个样本就循环多少次
# #但是一般不这么写
# # for i in range(num):
# #     x = trainSet[i:i+24,0] #x应该是一维的否则下一步维数对不上，test是二维数组，但test[i]是二维数组中的一维的数，逗号后的0就是取第一行的意思
# #     test[i] = x
#
# #划分样本的方法二
# df_train = [] #先建立一个空的数组，不用规定维数
# for i in range(num): #勿忘冒号
#     x = trainSet[i:i+24] #不用管维数
#     df_train.append(x) #这里是加到列表里，不是数组形式
# df_train = np.array(df_train) #缩进出循环，把列表转为数组
# print(df_train.shape)
