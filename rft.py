import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score



#数据输入与划分
# data = np.loadtxt(r'data2.txt')
# X = data[:,0:2]
# Y = data[:,2]
# xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.3,random_state=0)

# #数据存储
# train = np.c_[xTrain,yTrain]
# np.savetxt(r"train.txt",train)
# test = np.c_[xTest,yTest]
# np.savetxt(r"test.txt",test)
print('开始训练...')
train = np.loadtxt(r'.\data\train.txt')
xTrain = train[:,0:2]
yTrain = train[:,2]
#模型训练
s = time.time()
TRF = ensemble.RandomForestRegressor(
    n_estimators=1000,
    max_features=2,
    oob_score=True,
    random_state=0,
    # min_samples_split=3,
    n_jobs=200)
TRF.fit(xTrain,yTrain)
e = time.time()
print('训练完毕，花费时间：'+str(e-s))

test = np.loadtxt(r'.\data\test.txt')
xTest = test[:,:2]
yTest = test[:,2]
print(xTest[:,0])
#模型预测
pre = TRF.predict(xTest)

print('合并预测结果为xyz形式')
preData=np.column_stack((xTest[:,:2],pre))
preData=np.column_stack((preData,pre - yTest))
# print(preData)
np.savetxt(r".\data\pre.txt",preData)

#验证
print('开始计算误差指标...')
m = np.average(pre - yTest)
am = mean_absolute_error(yTest, pre)
mse = mean_squared_error(yTest,pre)
std = np.std(pre - yTest)
r2 = r2_score(yTest, pre)
print('ME，MAE，MSE，STD，R2')
print(m,am,mse,std,r2)

