# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 23:47:26 2018

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 30 05:48:35 2018

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 27 02:28:14 2018

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
os.chdir('C:\\Users\\admin\\Documents')
k['open'] =k.Open
k['close'] =k.Close
k['high'] =k.High
k['low'] =k.Low
k['date'] = k.Date
def sin1(x):
    return K.sin(x)

from datetime import datetime
from datetime import date
k=pd.read_pickle('sinbacktest.pkl')
model=load_model('sin4.h5',custom_objects = {'sin1':sin1})
s1 = []
count= 0
test1 = []
for lk in k.Symbol.unique():
    k1=k[k.Symbol==lk]
    for i in range(len(k)-20):
        if(model.predict(np.array(k[['rsi','ft']].iloc[i:i+20]).reshape(1,20,2))>0):#prediction
            test1.append(k.iloc[i+20:i+21])
        print(str(i)+'complete')
       
k = pd.concat(test1)
b = 0.6
k['open'] =k.Open
k['close'] =k.Close
k['high'] =k.High
k['low'] =k.Low

k['nret']=((k.open-k.low)/(k.open))*100

k['ret']=((k.close-k.open)/k.open)*100
k['dir1'] = k.ret
k['ret'] = k.ret*1
k['nret'] = k.nret*1
k['Date'] = k.index
t1 = []
k3 = k
pk = []





m1= k3
r1=4000
pos4 = 4000
pos2 = 0
pos = []
c2 = 0
c = 0

recs= []
#backtesting for checking out profits
for i in sorted(list(m1.date.unique())):

 m = m1[m1.date == i]
 m = m.sort_values(['Volume'],ascending = False)
 if(len(m)>4):
       
  m = m[0:4]
 m =m.sort_values(['open'])

 m = m.reset_index()
 
 
 tq = 0
 
 
 
    
 for j in range(len(m)): 
  k3 = m.iloc[j:j+1]
  k3['a'] = 2/(len(m))
  div= len(m)
 
  q = round((2*r1)/(0.6*div*float(k3.open)))
  tq = tq+ q*float(k3.open)
  if(tq>10*r1):
        q = 0
  k3['nret'] = (q*(k3.open - k3.low)/(r1))*100
  k3['ret'] = (q*(k3.close - k3.open)/(r1))*100
  k3['a'] = 2/(len(m))
  
  recs.append(q)
  
 
  pos1 =  k3[(k3.ret>0) & (k3.nret<k3.a)].ret.sum()-k3[(k3.ret>0) & (k3.nret>k3.a)].a.sum()-k3[k3.ret<-k3.a].a.sum() + k3[(k3.ret<0)&(k3.ret>-k3.a)].ret.sum()
  pos2=pos2+pos1
  r1 = r1 + (pos1/100)*r1
  if(pos1>0):
      c2 = c2+1
  c=c+1
  pos.append(((r1-4000)/4000)*100)
  print("Open:" +str(float(k3.open)))
  print("qauntity:" +str(q))
  print("date:"+str(i))
  print(pos[-1])
print("no of correct trades"+ str(c2/c))

