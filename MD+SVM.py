from rdkit import Chem as chem      #主要是负责常用的化学功能
from rdkit import DataStructs as datastructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
import numpy as np   #  是一种可以处理大型的多维矩阵库
import xlrd     #导入excal文件
import pandas as pd

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
# fs=FeatureSelector(data=x,labels=y)
# fs.identify_missing(missing_threshold=0.6)
# fs.missing_stats.head()
# fs.identify_single_unique()    #有151个特征只有单一特征值
# fs.identify_collinear(correlation_threshold=0.98)  #有59个特征值相关性大于0.98
#
# train_no_missing=fs.remove(methods=['missing','single_unique','collinear'],keep_one_hot=False)
# train_no_missing=np.array(train_no_missing)
# x1=train_no_missing.tolist()
# x1=pd.DataFrame(x1)
# x1=DataFrame.drop(self=x1,columns=6)
# x1=DataFrame.drop(self=x1,columns=7)
# x1=DataFrame.drop(self=x1,columns=8)
# x1=DataFrame.drop(self=x1,columns=9)

from sklearn.preprocessing import StandardScaler    #预处理数据
x1=StandardScaler().fit_transform(x)
x1=DataFrame(x1)

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.2)
# x_train=x1[0:7000]
# x_test=x1[7000:9435]
# y_train=y[0:7000]
# y_test=y[7000:9435]
model=SVR(kernel='rbf',C=100,degree=5)
model.fit(x_train,y_train)

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