# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:25:26 2021

@author: lov
"""
import numpy as np
import itertools
def pairscal(m,pb):
    iv=np.zeros((m))
    for i in range(0,m):
        iv[i]=i
    np.random.shuffle(iv)
    m=int(m*pb/(2*100))
    print(m)
    pairs=np.zeros((m,2))
    l=0;
    while l<m:
        print(iv[2*(l):2*l+2])
        pairs[l,0:2]=iv[2*(l):2*l+2]
        l=l+1
    return pairs  
def pairscaln(m,pb):
    iv=np.zeros((m))
    for i in range(0,m):
        iv[i]=i
    pairs = list(itertools.combinations(iv, 2))
    np.random.shuffle(pairs)
    l=int(len(pairs)*pb/100)
    return pairs[0:l]
def crossoverv(x,lb,up,pb):
    pb=pairscaln(np.shape(x)[0],pb)
    y=np.zeros((np.shape(pb)[0],x.shape[1]))
    xn=np.zeros((np.shape(pb)[0],np.shape(x)[1]))
    for i in range(0,np.shape(pb)[0]):
        a=int(pb[i][0]);
        b=int(pb[i][1]);
        x1=x[a,lb:up]
        xn[i,0:lb]=x[a,0:lb]
        xn[i,lb:up]=x[b,lb:up]
        xn[i,up:np.shape(x)[1]]=x[a,up:np.shape(x)[1]]
    return xn   

def mutationv(x,pm):
    m=int(x.shape[1]*pm/100)
    for i in range(0,x.shape[0]):
        n=np.random.randint(0,x.shape[1],m)
        for j in range(0,n.shape[0]):
            x[i,n[j]]=49-x[i,n[j]]
    return x 

def originaldata(data,values):
    scaled_dataset1 = scaler.fit_transform(values[:,0].reshape(-1,1))
    d=scaler.inverse_transform(data[:,0].reshape(-1,1))
    data[:,0]=d.reshape(1,-1)
    d=scaler.inverse_transform(data[:,1].reshape(-1,1))
    data[:,1]=d.reshape(1,-1)
    return data

v1=[9,10,1,10,10,1,10,10,1,1,1]
def fitnesscal(a,m1,y_test,X_test,X):
    c=a[0].copy()
    msec=np.zeros((100,1))
    for i in range(0,100):
        b=X[i,:]
        k=0;
        for j in range(0,11):
            if v1[j]>1:
                for j1 in range(0,v1[j]):
                    c[j][j1]=a[int(b[k])][j][j1]
                    k=k+1
            else:
                c[j]=a[int(b[k])][j]
                k=k+1
        m1.set_weights(c)
        Yp = m1.predict(X_test)
        yp1=np.concatenate((Yp.reshape(-1,1),y_test), axis=1)
        ypred1=originaldata(yp1,values)
        msec[i]=mean_squared_error(ypred1[:,1],ypred1[:,0])
    return msec   


import random
X5 = [random.randint(0, 49) for i in range(0, 64*100)]
x = np.reshape(X5, (100, 64))
bfit=100000
itern=0
sam=0
while itern<100:
    msn=fitnesscal(a,m1,y_test,X_test,x)
    print(np.min(msn))
    bf=np.min(msn)
    bf1=np.argmin(msn)
    if bfit==bf:
        sam=sam+1
    else:
        sam=0
    if bfit>bf:
        bfit=bf
        bff=x[bf1,:]
    x=crossoverv(x,20,40,0.5)
    x=mutationv(x,0.1)
    non=100-np.shape(x)[0]
    X5 = [random.randint(0, 49) for i in range(0, 64*non)]
    xnew = np.reshape(X5, (non, 64)) 
    x=np.concatenate((x,xnew),axis=0)
    if sam==2:
        X5 = [random.randint(0, 49) for i in range(0, 64*100)]
        x = np.reshape(X5, (100, 64))   
    itern=itern+1
    print(bfit)