import xlrd
import numpy as np

workbook=xlrd.open_workbook(r'C:\Users\asud\Desktop\co2\卷积自编码\C2.xlsx')
data=workbook.sheets()[0]
solve=data.col_values(2)
temperature=data.col_values(0)
persure=data.col_values(1)
y=solve[0:9225]
T=temperature[0:9225]
T=np.array(T)
T2=np.expand_dims(T,axis=1)
print(T2.shape)
p=persure[0:9225]
p=np.array(p)
p=np.expand_dims(p,axis=1)
print(y)

from keras.models import load_model
auto=load_model("autoencoder.h985", compile=False)
en=load_model("encoder.h985", compile=False)
de=load_model("decoder.h985", compile=False)


workbook2=xlrd.open_workbook(r'C:\Users\asud\Desktop\co2\卷积自编码\C3.xlsx')
data2=workbook2.sheets()[0]
i=0
smiles=[]
while i<9224:
    smiles1=data2.row_values(i)
    smiles.append(smiles1)
    i+=1
print("分子总数",len(smiles))
print("总原子数",len(smiles[0]))
print("最后一个离子液体原子",smiles[9223])
#########################################################################################################数据提取
i=0
a=sum(smiles,[])
feature=[]
for i in a:
    if i not in feature:
        feature.append(i)
print("独立原子个数",len(feature))
print("独立原子",feature)
##########################################################################################################特征提取
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(feature)
i=0
ionic=[]
for i in smiles:
    a=le.transform(i)
    ionic.append(a)
print("特征编码后原子数",len(ionic[0]))
print("特征编码后原子编码",ionic[0])
##############################################################################################################特征编码
from sklearn.preprocessing import OneHotEncoder
feature2=le.transform(feature)
feature2=feature2.tolist()
feature1=[]
for i in feature2:
    c=[]
    c.append(i)
    feature1.append(c)
print("特征编码后独立原子编码",feature1)
dict_label = dict(zip(feature,feature1))#标签对应字典
print("标签对应字典",dict_label)
yuanzi=[]
for i in ionic:
    yuanzii=i.reshape(-1,1)
    yuanzi.append(yuanzii)
enc=OneHotEncoder(handle_unknown="error")
enc.fit(feature1)
ionicoh=[]
for i in yuanzi:
    ionicoh.append(enc.transform(i))
ionic1=ionicoh[0].toarray()
print("独热编码后第一个离子液体编码",ionic1)
ionic2=ionic1.tolist()
print("独热编码后每个离子液体原子数",len(ionic2))
print("独热编码后第一个离子液体编码",ionic2)
feature3=enc.transform(feature1)
dict_one_hot = dict(zip(feature,feature3))#标签对应字典

#print("标签对应字典",dict_one_hot)
################################################################################################################离子液体编码
ioniconehot=[]
for i in ionicoh:
    i=i.toarray()
    ioniconehot.append(i)
# 训练模型
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras.callbacks import EarlyStopping
ioniconehot4=np.expand_dims(ioniconehot,axis=3)
####################################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from datetime import datetime
from keras.models import load_model
###################################################################读取自编码解码
auto=load_model("autoencoder.h985", compile=False)
en=load_model("encoder.h985", compile=False)
de=load_model("decoder.h985", compile=False)
#######################################################################读取生成对抗模型
print(len(ioniconehot4[0]))
print(len(ioniconehot4[0][0]))
print(len(ioniconehot4[0][0][0]))
print(len(ioniconehot4))
iii=en.predict(ioniconehot4)
###################################################################################
def flatten(input_list):
    output_list = []
    while True:
        if input_list == []:
            break
        for index, i in enumerate(input_list):

            if type(i)== list:
                input_list = i + input_list[index+1:]
                break
            else:
                output_list.append(i)
                input_list.pop(index)
                break

    return output_list
lll=[]
iiii=iii.tolist()
for iii in iiii:
    ll=flatten(iii)
    lll.append(ll)
print("lll:",len(lll))
print("lll[]:",len(lll[0]))
########################################################################################
y=np.array(y)
lll=np.array(lll)
llll=tf.concat([lll,T2,p],axis=1)
print(llll.shape)
print(llll)
from sklearn.preprocessing import StandardScaler    #预处理数据
from pandas.core.frame import DataFrame
x1=StandardScaler().fit_transform(llll)
x1=DataFrame(x1)
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.2,random_state=0)

##########################################################################################
#模型训练
from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor(n_estimators=60,max_depth=40)
model.fit(x_train,y_train)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
print('训练R2得分',r2_score(y_train,model.predict(x_train)))
print('训练MAE得分',mean_absolute_error(y_train,model.predict(x_train)))
print('训练MSE得分',mean_squared_error(y_train,model.predict(x_train)))
print('测试R2得分',r2_score(y_test,model.predict(x_test)))
print('测试MAE得分',mean_absolute_error(y_test,model.predict(x_test)))
print('测试MSE得分',mean_squared_error(y_test,model.predict(x_test)))
print(model.predict(x_test))
print(len(model.predict(x_test)))

