from cudf import DataFrame
import pandas as pd
import numpy as np
import warnings
from datetime import date
import datetime
warnings.filterwarnings("ignore")
from keras.models import load_model
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.optimizers import SGD
from keras import initializers
from keras.layers import Dense, Activation,LSTM,Dropout
import os
import time
k=pd.read_csv('11ystocks.csv')
k['Date'] = pd.to_datetime(k.Date)
k = k[(k.Date>=datetime.date(2008,1,1)) & (k.Date<=datetime.date(2018,12,31))]
l1 = []
k['date'] = k.Date
def norm1(l):
  std1 = l.mean()+4*l.std()
  std2 = l.mean()-4*l.std()
  
  range = std1-std2
  print(range)
  l = (l/abs(l))*(l -std2)/range
  return l
def norm2(l):
  std1 = l.mean()+4*l.std()
  std2 = l.mean()-4*l.std()
  
  range = std1-std2
  print(range)
  l = (l/abs(l))*(l -std2)/range
  return std2,std1

for i in k.Symbol.unique():
    k1 = k[k.Symbol==i]
    k1['vol'] = k1.Volume/k1.Volume.shift(1)
    k1['ret10'] = (k1.Close.shift(-10)-k1.Close)
    k1['ret'] =  (k1.Close.shift(-1)-k1.Close)
    k1['ret10'] = (k1.ret/k1.Close)*100
    k1['ret'] = (k1.ret/k1.Close)*100
    
    if('Trades' in k1.columns):
        k1= k1.drop('Trades',1)
    k1 = k1.dropna()
    k1['nret10'] = norm1(k1.ret10)
    k1['nret'] = norm1(k1.ret)
    if(np.isnan(list(k1.nret)[0])):
        print(k1.ret)
        time.sleep(20)
    
    l1.append(k1)
k =pd.concat(l1)





lg = []
c=0
lt = []
lt1 = []
lti =[]
ltia=[]
lta = []
ltb = []
nret = []
stck = []
stck1 = []
stck2 = []
x=[]
y = []
yy = []
stcktot = []
stcktot1 = []
for i in k.Symbol.unique():
    k1 = k[k.Symbol==i]
    k11= k1[k1.date<=datetime.date(2017,1,1)]
    k12 = k1[(k1.date>=datetime.date(2017,1,1)) & (k1.date<=datetime.date(2018,1,1))]
    k13 = k1[k1.date>=datetime.date(2018,1,1)]
   
    l= k11
    no = []
    nc=[]
    nh=[]
    nl=[]
    
    for j in range(len(l)-20):
       
        l1 = l.iloc[j:j+20]
        l1= l1.fillna(method = 'ffill')

        var = list(l1.Close)[-1]
        date1 = l1.Date
        nret.append(list(l1.nret)[-1])
        no = pd.Series(data = l1.Open/var)#normalization procedure of taking the ratio of previous 20 OCHL data to the current OCHL data
        nc = pd.Series(data = l1.Close/var)
        nh = pd.Series(data = l1.High/var)
        nl = pd.Series(data =l1.Low/var)
        df = pd.DataFrame()
        df['o'] = list(no)
        df['h'] = list(nh)
        df['l'] = list(nl)
        df['c'] = list(nc)
        df['date'] = list(date1)
        df['nret']= list(l1.nret)
        df['nret10']= list(l1.nret10)
        df['Symbol'] = list(l1.Symbol)
        print(j)
        stck.append(df)
        
    l= k12
    no = []
    nc=[]
    nh=[]
    nl=[]
    
    for j in range(len(l)-20):
       
        l1 = l.iloc[j:j+20]
        l1= l1.fillna(method = 'ffill')

        var = list(l1.Close)[-1]
        date1 = l1.Date
        nret.append(list(l1.nret)[-1])
        no = pd.Series(data = l1.Open/var)
        nc = pd.Series(data = l1.Close/var)
        nh = pd.Series(data = l1.High/var)
        nl = pd.Series(data =l1.Low/var)
        df = pd.DataFrame()
        df['o'] = list(no)
        df['h'] = list(nh)
        df['l'] = list(nl)
        df['c'] = list(nc)
        df['date'] = list(date1)
        df['nret']= list(l1.nret)
        df['nret10']= list(l1.nret10)
        print(j)
        stck1.append(df)
    l= k13
    no = []
    nc=[]
    nh=[]
    nl=[]
    
    for j in range(len(l)-20):
       
        l1 = l.iloc[j:j+20]
        l1= l1.fillna(method = 'ffill')

        var = list(l1.Close)[-1]
        date1 = l1.Date
        nret.append(list(l1.nret)[-1])
        no = pd.Series(data = l1.Open/var)
        nc = pd.Series(data = l1.Close/var)
        nh = pd.Series(data = l1.High/var)
        nl = pd.Series(data =l1.Low/var)
        df = pd.DataFrame()
        df['o'] = list(no)
        df['h'] = list(nh)
        df['l'] = list(nl)
        df['c'] = list(nc)
        df['date'] = list(date1)
        df['Open'] = list(l1.Open)
        df['Close'] = list(l1.Close)
        df['High'] = list(l1.High)
        df['Low']= list(l1.Low)
        df['nret']= list(l1.nret)
        df['nret10']= list(l1.nret10)
        print(j)
        stck2.append(df)
      
      
stck=pd.concat(stck)
stck1=pd.concat(stck1)
stck2=pd.concat(stck2)

for v in range(10):
  temp = stck[v*round(len(stck)/10):(v+1)*round(len(stck)/10)]
  temp = temp[len(temp)-len(temp)%20]
  temp['o'] = norm1(temp.o)
  temp['c'] = norm1(temp.c)
  temp['h'] = norm1(temp.h)
  temp['l'] = norm1(temp.l)
  for i in range(len(temp)/20):
    x.append(np.array(temp['o','c','h','l'].iloc[i*20:(i+1)*20]))
    y.append(np.array(temp.nret.iloc[((i+1)*20)-1]))
    yy.append(np.array(temp.nret10.iloc[((i+1)*20)-1]))
    print(str(i)+" done")
    

model = Sequential()
model.add(LSTM(64, input_shape=(20, 4),return_sequences = True))
model.add(Activation('tanh'))
model.add(Dropout(0.2))

model.add(LSTM(32))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Activation('tanh'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer = 'adam')
x= np.array(x)
y = np.array(y)
yy = np.array(yy)
def test(stck1):
    t1 = stck
    x1= []
    y1 = []
    y2 = []
    
    bct = []
    i= 0
    t1 = stck1.copy()
    t1 = t1[round(0.9*len(t1)):]      

    while i<len(stck1):
       if(i == 0):
           t1 = t1.iloc[20:]
       else:
           t1=t1.iloc[1:]
       temp1 = stck1.iloc[i:i+20]
       t2 = t1.copy()
       t1= temp1.copy()
       cmax = t1.High.max()
       cmin = t1.Low.min()        
       t1['c'] = (t1.c-cmin)/(cmax-cmin)
       t1['h'] = (t1.h-cmin)/(cmax-cmin)
       t1['l'] = (t1.l-cmin)/(cmax-cmin)
       
       t1['o'] = (t1.o-cmin)/(cmax-cmin)
       t2 = t2.append(t1)
       t1= t2
       t1['o'] = norm1(t1.o)
       t1['c'] = norm1(t1.c)
       t1['h'] = norm1(t1.h)
       t1['l'] = norm1(t1.l)

       x1.append(np.array(t1[['o','c','h','l']].tail(20)).reshape(1,20,4))
       y2.append(np.array(model.predict(x1[-1])[0]))
       y1.append(np.array(list(temp1.nret)[-1]))
       
       
       i  = i+20
       

       if(y2[-1]*y1[-1]>0):
           c_right = c_right+1
           print('correct')
       else:
           print('incorrect')
       if(y1[-1]>0):
           temp1['pred'] = 1
       else:
           temp1['pred'] = -1
       bct.append(temp1.tail(1))
       i = i+20
       c_tot = c_tot+1
       f =open('results.txt','w+')
       f.write('accuracy:'+str(c_right/c_tot)+'\r\n')
       f.close()
       print('accuracy:'+str(c_right/c_tot))
       print('renorm' + str(i) + 'completed')
accdf = pd.DataFrame(columns= ['model name','accuracy','acc_improv'])
val = True
def train():
  history = model.fit(x,y, epochs =4 ,verbose = 1,batch_size= 200)
  return history
c= 0
while(val):
  
 history = train()
 if(history.history['loss'][-1]<0.2430):
    dd=stck1.copy()
    c_right = 0
    c_tot = 0
    test(stck1[0:round(0.2*len(dd))])
    
    if((c_right/c_tot)>.53):
      val1 = True
      while(val1):
       if((len(accdf)>=30) or (list(accdf.acc_improv)[-1]<0)):
        val1= False
        val= False
       c_right = 0
       c_tot = 0
       test(dd[round(0.2*len(dd)):])
       acc = c_right/c_tot
       c= c+1
       name= 'stockmodelpartnorm_it_'+str(c)+'.h5'
       model.save(name)
       if(len(accdf)>=2):
        accdf.append({'model name':name,'accuracy':acc,'acc_improv':list(accdf.accuracy)[-1]-list(accdf.accuracy)[-2]})
       else:
        accdf.append({'model name':name,'accuracy':acc})
       if((len(accdf)>=30) or (list(accdf.acc_improv)[-1]<0)):
        val1= False
        val= False
        accdf.to_csv('modelname.csv')
       else:
        history = train()
        
          
    
 else:
       train()
test(stck2)
      
