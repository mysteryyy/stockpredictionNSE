  
    
from keras.layers import Input, Dense
from keras.models import Model
import pandas as pd
import numpy as np
import warnings
from datetime import date
import datetime
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from joblib import dump, load
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
warnings.filterwarnings("ignore")
from keras.models import load_model
from numpy.lib.stride_tricks import as_strided
from keras.utils import to_categorical
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.optimizers import SGD
from keras import initializers,regularizers
from keras.layers import Dense, Activation,LSTM,Dropout
from sklearn.svm import SVC

from scipy.stats import chisquare
import os
import time
import os
class autoencode:
    def __init__(self,xc,yc,xc1,yc1,ch):
        self.xc=xc
        self.yc =yc
        self.xc1=xc1
        self.yc1=yc1
        self.ch=ch
        
        
        # this is our input placeholder
        def ancd(xc,input_dim,encoding_dim):
            
            
            
            input_img = Input(shape=(input_dim,))
            dr=Dropout(0.2)(input_img)
            
        
        
        
            # add a Dense layer with a L1 activity regularizer

            encoded1 = Dense(encoding_dim, activation='tanh',
                            activity_regularizer=regularizers.l1(10e-4))(dr)
            bn3= BatchNormalization()(encoded1)
            
            
            
            
            
            decoded = Dense(input_dim, activation='tanh')(bn3)
            
            autoencoder = Model(input_img, decoded)
            
            # this model maps an input to its encoded representation
            encoder = Model(input_img, encoded1)
            
            # create a placeholder for an encoded (32-dimensional) input
            encoded_input = Input(shape=(encoding_dim,))
            # retrieve the last layer of the autoencoder model
            decoder_layer = autoencoder.layers[-1]
            # create the decoder model
            #decoder = Model(encoded_input, decoder_layer(encoded_input))
            
            autoencoder.compile(optimizer='adam', loss='mean_squared_error')
            autoencoder.fit(xc, xc,
                            epochs=15,
                            batch_size=500,
                            shuffle=True,verbose=1)
            return encoder.predict(xc),encoder
        xct = self.xc
        print(xct.shape)
        encoder = []
        def encpred(xc,encoder):
             for i in encoder[0:len(encoder)-1]:
                xc = i.predict(xc)
             return xc
        def acc(labels_unique,encoder,ms,xpe,xc1=self.xc1,yc1=self.yc1):
            os.chdir('C:\\Users\\admin\\Documents')
            s = pd.DataFrame()
            sg =pd.DataFrame()
            
            xz = xpe
            xz1 = encpred(self.xc,encoder                                                                                                                                                                                  )
            
           
            s['label'] = pd.Series(ms.predict((xz)))
            s['ret']= pd.Series(yc1)/abs(pd.Series(yc1))
            sg['label'] = pd.Series(ms.predict((xz1)))
            sg['ret']= pd.Series(self.yc)/abs(pd.Series(self.yc))
            def check(s,i):
                cnt = []
                s1 = s[s.label==i]
                print(len(s1))
                cnt.append(len(s1))
                if(len(s1)==0):
                    return 0,0
                print(len(s1[s1.ret==1])/(len(s1)))
                return len(s1[s1.ret==1])/(len(s1)),len(s1)
            r1 = 'result '+str(int(self.xc.shape[1]/4))+'.pkl'
            r2 = pd.DataFrame()
            nm=[]
            tracc=[]
            teacc =[]
            cnt =[]
            cnt1=[]
            for i in labels_unique:
                print(str(i)+' train')
                nm.append(str((i)))
                fa,fb = check(sg,i)
                tracc.append(fa)
                cnt.append(fb)
                
                print(str(i)+ 'test')
                fa,fb = check(s,i)
                teacc.append(fa)
                cnt1.append(fb)
            r2['name'] = nm
            r2['training'] = tracc
            r2['testing'] = teacc
            r2['training_count'] = cnt
            r2['testing_count'] = cnt1
            print(tracc)
            print(teacc)
            r2.to_pickle(r1)
            os.chdir('C:\\Users\\admin\\test')
            return r2
            
        def encobj(xct=self.xc):#for encoding data into a lower dimension
             while(xct.shape[1]>3):
                obj = ancd(xct,xct.shape[1],int(xct.shape[1]/2))
                
                xct = obj[0]
                encoder.append(obj[1])
             return encoder
        
        def ms(xct=self.xc):
            encoder = encobj()
              
            xpe = encpred(xct,encoder)
            bandwidth = estimate_bandwidth(xpe, quantile=0.2, n_samples=1500)
            
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True,cluster_all=False)
            ms.fit(xpe)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_
            
            labels_unique = np.unique(labels)
            dump(ms, 'meanshift4.joblib') 
            ms = load('meanshift4.joblib')
            print(cluster_centers)
            r2 =acc(labels_unique,encoder,ms,encpred(self.xc1,encoder))
            return r2,ms
        
        def svc(xc=self.xc,yc=self.yc,xc1=self.xc1,yc1=self.yc1):
         encoder = encobj()
         cl = SVC(kernel='rbf',probability=True)
        
         yc = (((yc/abs(yc))+1)/2).astype(int)
         yc1 = (((yc1/abs(yc1))+1)/2).astype(int)

         cl.fit(encpred(xc,encoder),yc)
         pyc1 = cl.predict_proba(encpred(xc1,encoder))
         check = pd.DataFrame()
         check['real'] = yc1
         check['probs'] = pyc1[:,[1]]
         check['err'] = abs(check.real-check.probs)
         check.to_pickle('result.pkl')
         print('err ',str(check.err.mean()))
        if(self.ch=='ms'):
            r = ms()
            print(r[0])
            r[0].to_pickle('resultscluster_'+str(20)+str(10)+'.pkl')
            dump(r[1], 'meanshiftinit.joblib') 
        else:
            svc()
         
            
        