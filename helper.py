#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:13:12 2022

@author: kosmas
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.io import loadmat

#participant ids
PARTICIPANTS = ["16","17","19","21","23","25","26","28","30","34","37","39","41","42","43","45","46","48","56","58","62","64","65"]


def calc_baselines(y1):
    """
    Parameters
    ----------

    y : array
        Label set
  
    """
    b1 = np.mean(y1)
    if b1<=0.5:
        b1=1-b1
   
    return b1


def keep_dims(X,y,batch_size,z=None,t=None):
    """
    Parameters
    ----------
    X : array
        Feature set
    y : array
        Label set
    batch_size: int
        Number of samples per data batch
    z: array or NoneType
        Change labels
    t: array or NoneType
        Trend labels
    """
    if X.shape[0]%batch_size==1:
        X=X[:-1]
        y=y[:-1]
        if z!=None:
            z=z[:-1]
        if t!=None:
            t=t[:-1]
        
    return X,y,z,t

def feature_operation(window_features,operation='skip',sample_skip=5):
    """
    Parameters
    ----------
    window_features : array
        fratures within a time window
    operation : string
        Reduction operation to perform
   sample_skip: int
        Number of samples to skip (optional)

    """
    
    if operation=='skip':
        window_features[[idx for idx in range(0,window_features.shape[0],sample_skip)]]
        window_features = window_features.flatten()
    elif operation=='mean':
        window_features=np.mean(window_features,axis=0)
    elif operation=='grad':
        window_features=np.mean(window_features[1:,:]-window_features[:-1,:],axis=0)
    
    return window_features

def load_set(fold, label,feature_space,operation='concat',thresh=None,tw=1):
    """
    Parameters
    ----------
    fold : list
        Participant ids 
    label : string
        Affect dimension
   feature_space: string
        The features used during training 
    operation : string
        Reduction operation to perform
    thres: float
        Separate affect values
    tw: int
       Rime window length in sec
    """
    #lists to save data
    X_set = [] 
    y_set = []
    z_set=[]
    t_set=[]
    X_f=[]
    y_f=[]
    z_f=[]
    t_f=[]
    
    #iterate over player ids
    for pid in fold:
        #load and save participants data
        X,y,z,t= load_data(pid, label,feature_space,operation,tw)
        X_set.append(X)
        y_set.append(y)
        z_set.append(z)
        t_set.append(t)
        
    #concatenate the individualarrays to form the dataset    
    X_set = np.concatenate(X_set,axis=0)
    y_set = np.concatenate(y_set,axis=0)
    z_set = np.concatenate(z_set,axis=0)
    t_set = np.concatenate(t_set,axis=0)
    
    #set thresholds
    thres=np.median(y_set) if thresh==None else thresh
    ct=0.0017 if label=='x' else 0.0013
    tt=0
    epsilon=0.1
    
    #create labels
    for iid in range(X_set.shape[0]):
        window_features=X_set[iid]
        window_trace = y_set[iid]
        window_change = z_set[iid]
        window_trend=t_set[iid]
       
        change = 1 if window_change>ct else 0
        trend =1 if window_trend>tt else 0
        if window_trace>thres+epsilon:
            X_f.append(window_features)
            y_f.append(1)
            z_f.append(change)
            t_f.append(trend) 
        elif window_trace<thres-epsilon:
            X_f.append(window_features)
            y_f.append(0)
            z_f.append(change)
            t_f.append(trend)
    return thres,np.array(X_f),np.array(y_f),np.array(z_f),np.array(t_f)



def load_data(pid,label, feature_space ='AU',operation='skip',tw=1):
    """
    Parameters
    ----------
    pid : string
        Participant id
    label : string
        Affect dimension
   feature_space: string
        The features used during training 
    operation : string
        Reduction operation to perform
    tw: int
       Rime window length in sec
    """
    #lists to save data
    X=[]
    y=[]
    z=[]
    tr=[]
    
    frame_skip=5
    #skip and window length hyperparameters
    frame_step=10
    window_size=25*tw
    
    #read participants data and obtain features and affect trace
    df = pd.read_csv('./PROCESSED-DATA/P'+str(pid)+'.csv')
    label_df = df.iloc[:,-12:]
    trace= label_df[[k for k in label_df.keys() if '_'+label in k]].values
    trace = np.median(trace,axis=1)
 
    #keep features according to the feature_space parameter
    df_keys=[]
    if feature_space =='AU':
        for key in df.keys():
            if 'VIDEO_40_LLD_AU' in key and 'delta' not in key:
                df_keys.append(key)
        features =df[df_keys].values
    elif feature_space =='ECG':
        for key in df.keys():
            if 'ECG_54_LLD_' in key :
                df_keys.append(key)
        features =df[df_keys].values
        features[np.isnan(features)]=0
    elif feature_space =='ECGEDA':
        for key in df.keys():
            if 'EDA_62_LLD_' in key or 'ECG_54_LLD_' in key:
                df_keys.append(key)
        features =df[df_keys].values
        features[np.isnan(features)]=0
    elif feature_space =='EDA':
        for key in df.keys():
            if 'EDA_62_LLD_' in key :
                df_keys.append(key)
        features =df[df_keys].values
        features[np.isnan(features)]=0
    elif feature_space =='AUDIO':
        for key in df.keys():
            if 'ComParE13_LLD_' in key :
                df_keys.append(key)
        features =df[df_keys].values
    elif feature_space  =='VIDEO':
        for key in df.keys():
            if 'VIDEO_' in key:
                df_keys.append(key)
        features =df[df_keys].values
    elif feature_space=="MULTIMODAL":
        for key in df.keys():
            if 'ComParE13_LLD_' in key or "EDA_62_LLD_" in key or "ECG_54_LLD_" in key or "VIDEO_" in key:
                df_keys.append(key)
        features =df[df_keys].values 
        features[np.isnan(features)]=0
    elif feature_space=="AUDIOVISUAL":
        for key in df.keys():
            if 'ComParE13_LLD_' in key or "VIDEO_" in key:
                df_keys.append(key)
        features =df[df_keys].values 
        # features[np.isnan(features)]=0
   
        
    for fid in range(0,features.shape[0],frame_step):
        if fid+window_size>features.shape[0]:
            continue
        #perform windowing & reduction
        window_features=features[fid:window_size+fid,:]
        window_trace = trace[fid:window_size+fid]
        window_features = feature_operation(window_features,operation=operation,sample_skip=frame_skip)
        
        #calculate the mean, change and trend traces
        window_change =np.mean(np.abs(window_trace[1:]-window_trace[:-1]))
        window_trend = np.mean(window_trace[1:]-window_trace[:-1])
        window_trace= np.mean(window_trace)
        X.append(window_features)
        y.append(window_trace)
        z.append(window_change)
        tr.append(window_trend)

        
    
    return np.array(X), np.array(y),np.array(z),np.array(tr)

def create_folds(lst,num_folds=5):
    """
    Parameters
    ----------
    lst : list
        Elements to put in folds
    num_folds : int        
        Number of folds 

    """
    folds=[[] for f in range(num_folds)]
    for idf in range(len(lst)):
        fid = idf%num_folds
        folds[fid].append(lst[idf])
    return folds




