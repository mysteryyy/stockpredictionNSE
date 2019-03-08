# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 19:53:47 2019

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 19:23:02 2019

@author: admin
"""
import pandas as pd
import numpy as np
import warnings
from datetime import date
import datetime
from keras import backend as K

warnings.filterwarnings("ignore")
from nsepy import get_history
from keras.models import load_model

from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.optimizers import SGD
from keras import initializers
from keras.layers import Dense, Activation,LSTM,Dropout
import os
import time
path = 'C:\\Users\\admin\\Documents'
os.chdir(path)
k=pd.read_csv('11ystocks.csv')
k['date'] = pd.to_datetime(k.Date)
def attr(p): #function for transformation of prices into required inputs
 k = p.dropna()
 
 k['mp'] = (k.High+k.Low)/2
 k['im'] = 1*(k.mp-(pd.rolling_min(k.Low,window = 20)))/((pd.rolling_max(k.High,window=20))-(pd.rolling_min(k.Low,window = 20)))
 k['ft'] = 0.5*np.log((1+k.im)/(1-k.im))
 k['delta'] = k.Close.shift(-1)-k.Close
 k['du'] = ((k.delta/abs(k.delta)+1)/2)*k.delta#keeping only the positive k.delta
 k['dd'] = ((abs((k.delta/abs(k.delta))-1)/2))*abs(k.delta)#keeping only the negative delta
 k['rs'] = pd.rolling_mean(k.du,14)/pd.rolling_mean(k.dd,14)
 k['rsi'] = 100.0 - (100.0 / (1.0 + k.rs))
 k['rsi'] = k.rsi/100
 k['ft']= (k.ft-k.ft.min())/(k.ft.max()-k.ft.min())#fisher transform
 k = k.dropna()
 return k
def norm1(l):#normalizing finction
  std1 = l.mean()+4*l.std()
  std2 = l.mean()-4*l.std()
  
  range1 = std1-std2
  print(range1)
  l = (l/abs(l))*(l -std2)/range1
  return l
def norm2(l):
  std1 = l.mean()+4*l.std()
  std2 = l.mean()-4*l.std()
  
  range1 = std1-std2
  print(range1)
  l = (l/abs(l))*(l -std2)/range1
  return std2,std1
l1 =[]
for i in k.Symbol.unique():
    
    k1 = k[k.Symbol==i]
    k1['ret10'] = (k1.Close.shift(-10)-k1.Close)#returns 10 days ahead 
    k1['ret'] =  (k1.Close.shift(-1)-k1.Close)#returns 1 day ahead
    
    k1['ret10'] = (k1.ret10/k1.Close)*100
    k1['ret'] = (k1.ret/k1.Close)*100
    
    if('Trades' in k1.columns):
        k1= k1.drop('Trades',1)
    k1 = k1.dropna()
    k1['nret10'] = norm1(k1.ret10)
    k1['nret'] = norm1(k1.ret)
    k1 = k1.dropna()
    k1 = attr(k1)
    if(np.isnan(list(k1.nret)[0])):
        print(k1.ret)
        time.sleep(20)
    
    
    l1.append(k1)
    print(i+"done")
k =pd.concat(l1)
x=[]
y = []
x1 = []
y1= []
x2=[]
y2 = []
yy=[]
yy1 = []
yy2 = []
valdf = []
testdf = []
for i in k.Symbol.unique():
    k1 = k[k.Symbol==i]
    k11= k1[k1.date<=datetime.date(2017,1,1)]
    k12 = k1[(k1.date>=datetime.date(2017,1,1)) & (k1.date<=datetime.date(2018,1,1))]
    k13 = k1[k1.date>=datetime.date(2018,1,1)]
   
    l= k11
    
    for j in range(len(l)-20):
       l1= l.iloc[j:j+20]
       l1= l1.fillna(method='ffill')
       if(len(l1)<20):
           continue
       else:
           
        x.append(np.array(l1[['rsi','ft']]))
        yy.append(np.array(list(l1.nret10)[-1]))
        print(j)
    l =k12
    for j in range(len(l)-22):
        l1= l.iloc[j:j+20]
        l1 = l1.fillna(method = 'ffill')
        if(len(l1)<20):
          continue
        else:
         
         yy1.append(np.array(list(l1.nret10)[-1]))
         valdf.append(l.iloc[j+20:j+21])
         print(j)
    l = k13
    for j in range(len(l)-21):
        l1= l.iloc[j:j+20]
        l1 = l1.fillna(method = 'ffill')
        if(len(l1)<20):
            continue
        else:
            
         x2.append(np.array(l1[['rsi','ft']]))
         yy2.append(np.array(list(l1.nret10)[-1]))
         testdf.append(l.iloc[j+20:j+21])
         print(j)
    print(i+" done")
valdf = pd.concat(valdf)
testdf = pd.concat(testdf)
x = np.array(x)
x1 = np.array(x1)
y1 = np.array(y1)
y2 = np.array(y2)
yy = np.array(yy)
yy1 = np.array(yy1)
yy2 = np.array(yy2)
np.save('x2_ind1.npy',x2)
np.save('x_ind1.npy',x)
np.save('x1_ind1.npy',x1)
np.save('y1_ind1.npy',y1)
np.save('y2_ind1.npy',y2)
np.save('yy_ind1.npy',yy)
np.save('yy1_ind1.npy',yy1)
np.save('yy2_ind1.npy',yy2)
def sin1(x):
    return K.sin(x)
model = Sequential()
model.add(LSTM(32, input_shape=(20, 2),return_sequences = True,activation= sin1))
model.add(Activation('tanh'))
model.add(Dropout(0.2))

model.add(LSTM(8))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(4))
model.add(Activation('tanh'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer = 'adam')
model.fit(x,yy, epochs =100 ,verbose = 1,batch_size= 100)

def acc(lta,ltb):
 y1 = []
 y2= []
 pr1= model.predict(lta)
 for i in pr1:
        y1.append(i[0])
 for i in ltb:
    y2.append(i)
 df = pd.DataFrame({'real':y2,'predicted':y1})
 df['check'] = df.real*df.predicted#positive for correct directional prediction and negative otherwise
 print(len(df[df.check>0])/(len(df)))#printing accuracy
 return df
print(acc(x1,yy1))
print(acc(x2,yy2))

        