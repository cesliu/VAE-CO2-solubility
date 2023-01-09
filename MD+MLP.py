# from rdkit import Chem as chem      #主要是负责常用的化学功能
# from rdkit import DataStructs as datastructs
# from rdkit.Chem import AllChem
# from rdkit.Chem import Descriptors
# from rdkit.Chem import Draw
import numpy as np   #  是一种可以处理大型的多维矩阵库
import xlrd     #导入excal文件
import matplotlib.pyplot as plt
import math
import pandas as pd
from rdkit.Chem.Draw import SimilarityMaps

data0=np.loadtxt(r'C:\Users\asud\Desktop\co2\co5.dat')
x=data0.tolist()
workbook=xlrd.open_workbook(r'C:\Users\asud\Desktop\co2\卷积自编码\C1.xlsx')
data=workbook.sheets()[0]
solve=data.col_values(7)
y=solve[0:9224]


from pandas.core.frame import DataFrame
x=DataFrame(x)         #共有345个特征

#特征工程
from feature_selector import FeatureSelector
fs=FeatureSelector(data=x,labels=y)
fs.identify_missing(missing_threshold=0.6)
fs.missing_stats.head()
fs.identify_single_unique()    #有151个特征只有单一特征值
fs.identify_collinear(correlation_threshold=0.98)  #有59个特征值相关性大于0.98

train_no_missing=fs.remove(methods=['missing','single_unique','collinear'],keep_one_hot=False)
train_no_missing=np.array(train_no_missing)
x1=train_no_missing.tolist()
x1=pd.DataFrame(x1)
x1=DataFrame.drop(self=x1,columns=6)
x1=DataFrame.drop(self=x1,columns=7)
x1=DataFrame.drop(self=x1,columns=8)
x1=DataFrame.drop(self=x1,columns=9)

from sklearn.preprocessing import StandardScaler    #预处理数据
x1=StandardScaler().fit_transform(x1)
x1=DataFrame(x1)


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.3)

model = MLPRegressor(activation="tanh", alpha=0.01, batch_size='auto',  # Batch size一次训练所选取的样本数
                     beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-08,
                     hidden_layer_sizes=(64,26), learning_rate="constant",
                     learning_rate_init=0.001, max_iter=100000000, momentum=0.9,
                     nesterovs_momentum=True, power_t=0.5, random_state=None,
                     shuffle=True, solver="lbfgs", tol=1e-08, validation_fraction=0.1,
                     verbose=False, warm_start=False)
model.fit(x_train, y_train)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
print('训练R2得分',r2_score(y_train,model.predict(x_train)))
print('训练MAE得分',mean_absolute_error(y_train,model.predict(x_train)))
print('训练MSE得分',mean_squared_error(y_train,model.predict(x_train)))
print('训练RMSE得分',sqrt(mean_squared_error(y_train,model.predict(x_train))))
print('测试R2得分',r2_score(y_test,model.predict(x_test)))
print('测试MAE得分',mean_absolute_error(y_test,model.predict(x_test)))
print('测试MSE得分',mean_squared_error(y_test,model.predict(x_test)))
print('测试RMSE得分',sqrt(mean_squared_error(y_test,model.predict(x_test))))
print('总R2得分',r2_score(y,model.predict(x1)))
print('总MAE得分',mean_absolute_error(y,model.predict(x1)))
print('总MSE得分',mean_squared_error(y,model.predict(x1)))
print('总RMSE得分',sqrt(mean_squared_error(y,model.predict(x1))))

import tensorflow as tf
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
print(y_train)
print(y_test)
print(model.predict(x_train),end="")

#绘图
from matplotlib.font_manager import FontProperties
font=FontProperties(fname=r'C:\Windows\Fonts\Times New Roman')

fig=plt.figure(figsize=(10,4), dpi=300)

ax=fig.add_subplot(121)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 12}
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.scatter(y_test,model.predict(x_test),marker="*",s=20,c=  '#FF0000')
ax.set_xlabel('Experimental value',font2)
ax.set_ylabel('Test set results',font2)
#x=np.linspace(0,10,100)
#ax.plot(x,x,'-b')
ax2=fig.add_subplot(122)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
ax2.scatter(y_train,model.predict(x_train),marker="*",s=20,c=  '#FF0000')
ax2.set_xlabel('Experimental value',font2)
ax2.set_ylabel('Train set results',font2)
#ax2.plot(x,x,'-b')
plt.savefig('jieguosuijisenlin.tif',figsize=[10,4])
plt.show()

###########################################
from sklearn.model_selection import validation_curve
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
kfold=KFold(n_splits=10,shuffle=True)
shufile=ShuffleSplit(n_splits=5,train_size=0.8,test_size=0.2)
param_range=[0.4,0.45,0.48,0.5,0.55,0.58,0.6]
train_loss, test_loss= validation_curve(
        model, x1, y,param_name='alpha', param_range=param_range, cv=shufile, scoring='r2')

train_loss_mean = np.mean(train_loss, axis=1)
test_loss_mean = np.mean(test_loss, axis=1)

#可视化图形
from matplotlib.font_manager import FontProperties
fig=plt.figure(figsize=(10,10), dpi=300)
font=FontProperties(fname=r'C:\Windows\Fonts\Times New Roman')
ax=fig.add_subplot(111)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 10}
ax.plot(param_range, train_loss_mean, 'o-', color="r",
         label="Training")
ax.plot(param_range, test_loss_mean, 'o-', color="g",
        label="Cross-validation")

ax.set_xlabel("ALPHA",font2)
ax.set_ylabel("Determination coefficient",font2)
labelss = plt.legend(loc="best").get_texts()
[label.set_fontname('Times New Roman') for label in labelss]
plt.show()