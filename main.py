#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:14:59 2022

@author: kosmas
"""

import sklearn

import tensorflow as tf
import helper
import numpy as np
from helper import PARTICIPANTS
from itertools import chain
from scl_helper import SupervisedContrastiveLoss,add_probe
from scipy.io import savemat
from sklearn.preprocessing import StandardScaler as Scaler
import random
import os




def train_classifier(X,y,feature_space,num_epochs,batch_size):
    """
    Parameters
    ----------
    X : touple
        Train & Test feature sets
    y : touple
        Train & Test label sets
   feature_space: string
        The features used during training 
    num_epochs: int
        Number of training epochs
    batch_size: int
        Number of samples per data batch
    """
   
    X_train = X[0]
    X_test=X[1]
    y_train=y[0]
    y_test=y[1]
    #create classifier model
    es=tf.keras.callbacks.EarlyStopping( monitor="loss",min_delta=0,patience=10)
    model_enc = tf.keras.Sequential(
                        [
                            tf.keras.Input(shape=X_train[0].shape),
                            tf.keras.layers.Dense(30, activation='sigmoid'),
                            
                        ],
                        name="model_enc",
                    )
    model = tf.keras.Sequential([model_enc,
                                    tf.keras.layers.Dense(2),],name='model')
    
    
    model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                )
    
    print(feature_space.capitalize()+" classification")
    # Train and evaluate classifier on data.
    model.fit(X_train, y_train, epochs=num_epochs,batch_size=batch_size,verbose=0,callbacks=[es])
    ev=model.evaluate(X_test, y_test)
    
    return ev

    
def train_scl(X,y,feature_space,num_epochs,batch_size): 
    """
    Parameters
    ----------
    X : touple
        Train & Test feature sets
    y : touple
        Train, Contrastive & Test label sets
   feature_space: string
        The features used during training 
    num_epochs: int
        Number of training epochs
    batch_size: int
        Number of samples per data batch
    """
   
    X_train = X[0]
    X_test=X[1]
    y_train=y[0]
    y_test=y[1]
    y_contr=y[2]
    #create encoder model
    ese=tf.keras.callbacks.EarlyStopping( monitor="loss",min_delta=0,patience=10)
    model_enc = tf.keras.Sequential(
                        [
                            tf.keras.Input(shape=X_train[0].shape),
                            tf.keras.layers.Dense(30, activation='sigmoid'),
                            
                        ],
                        name="model_enc",
                    )
    model_enc.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss=SupervisedContrastiveLoss(0.1),
                )
    
    #train encoder
    model_enc.fit(X_train, y_contr, epochs=num_epochs,batch_size=batch_size,verbose=0, callbacks=[ese])
    #add probe
    model = add_probe(X_train[0].shape,model_enc)
    
    
    model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'],
                )
    esp=tf.keras.callbacks.EarlyStopping( monitor="loss",min_delta=0,patience=10)
    print(feature_space.capitalize()+" contrastive")
    #train and evaluate probe model
    model.fit(X_train, y_train, epochs=num_epochs,batch_size=batch_size,verbose=0,callbacks=[esp])
    ev=model.evaluate(X_test, y_test)
    return ev
    
def process_data(train_fold,test_fold,feature_space,tw, thres, batch_size, label='x',operation='mean'):
    """
    Parameters
    ----------
    train_fold : list
        Training set participants
    test_fold : list
        Test set participants
   feature_space: string
        The features used during training 
    tw: string
        Time window lebgth in seconds
    thres: float
        Separate High/Low affect values

    batch_size: int
        Number of samples per data batch
    label: string
        Affect dimenssion 
    operation: string
        Data reduction method
    """
   
    
    
    #load training and test sets
    _,X_train,y_train,z_train,t_train = helper.load_set(train_fold,label,feature_space,operation,tw=int(tw),thresh=thres)
    _,X_test,y_test,z_test,_ = helper.load_set(test_fold, label,feature_space,operation,tw=int(tw),thresh=thres)
    
    #perform scaling
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    
    #acoid shape misnmtch during SCL training
    X_train,y_train,z_train,t_train = helper.keep_dims(X_train,y_train,batch_size,z_train,t_train)
    X_test,y_test,z_test,_ = helper.keep_dims(X_test,y_test,batch_size,z_test)
    return X_train,y_train,z_train,t_train,X_test,y_test

def train_modality(X,y,feature_space,batch_size,num_epochs):
    """
    Parameters
    ----------
    X : touple
        Train & Test feature sets
    y : touple
        Train, Contrastive & Test label sets
   feature_space: string
        The features used during training 
    num_epochs: int
        Number of training epochs
    batch_size: int
        Number of samples per data batch
    """
   
    #training all models on a specific modality 
    fss = feature_space
    ev = train_classifier(X, [y[0],y[1]], fss, num_epochs, batch_size)
    cev = train_scl(X, [y[0],y[1],y[0]], fss, num_epochs, batch_size)
    cevc = train_scl(X, [y[0],y[1],y[2]], fss+' change', num_epochs, batch_size)
    cevt=train_scl(X, [y[0],y[1],y[3]], fss+' trend', num_epochs, batch_size)
    return [ev,cev,cevc,cevt]
    
SEED = 42
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(SEED)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(SEED)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(SEED)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(SEED)


#set hyperparemeters
num_trials=5
num_folds=5
learning_rate=0.001
batch_size = 250
num_epochs=300
num_classes  = 2
patience =10
paradigm='classification'
label='x'
operation='mean'
alp = '42'
thres = 0.05 if label=='x' else 0.09


#create folds
folds = helper.create_folds(PARTICIPANTS,num_folds)


for tw in['4']:
    baselines=[]
    save_dict={}
    cbaselines=[]
    acc_m=[]
    acc_mc=[]
    acc_mcc=[]
    acc_mclc=[]
    
    acc_v=[]
    acc_vc=[]
    acc_vcc=[]
    acc_vclc=[]
    acc_a=[]
    acc_ac=[]
    acc_acc=[]
    acc_aclc=[]
    acc_e=[]
    acc_ec=[]
    acc_ecc=[]
    acc_eclc=[]
    acc_av=[]
    acc_avc=[]
    acc_avcc=[]
    acc_avclc=[]
    dum=[]
    
    P_list=PARTICIPANTS
    for trial in range(num_trials):
        
        random.shuffle(P_list)
        #create folds
        folds = helper.create_folds(P_list,num_folds)
        for fid in range(num_folds):
            print('=== Trial '+str(trial+1)+', Fold '+str(fid+1)+ ' === arousal classification_'+operation+'_'+tw+'_'+alp)
            test_fold = folds[fid]
            train_fold = [folds[ii] for ii in range(num_folds) if ii!=fid]
            train_fold = list(chain.from_iterable(train_fold))
            
            
            
            #create datasets
            X_m_train,y_m_train,z_m_train,t_m_train,X_m_test, y_m_test=process_data(train_fold,test_fold,"MULTIMODAL",tw, thres, batch_size,label=label)
            
            X_v_train,y_v_train,z_v_train,t_v_train,X_v_test, y_v_test=process_data(train_fold,test_fold,"VIDEO",tw, thres, batch_size,label=label)
            
            X_a_train,y_a_train,z_a_train,t_a_train,X_a_test, y_a_test=process_data(train_fold,test_fold,"AUDIO",tw, thres, batch_size,label=label)
    
            X_e_train,y_e_train,z_e_train,t_e_train,X_e_test, y_e_test=process_data(train_fold,test_fold,"ECGEDA",tw, thres, batch_size,label=label)
            
            X_av_train,y_av_train,z_av_train,t_av_train,X_av_test, y_av_test=process_data(train_fold,test_fold,"AUDIOVISUAL",tw, thres, batch_size,label=label)
            
            
            #calculate baselines
            b1=helper.calc_baselines(y_m_train)
            b3=helper.calc_baselines(y_m_test)
            
            bl = [b1,b3]
            baselines.append(bl)
            print(bl)
            

         
            
#=================================MULTIMODAL================================================
            fss = 'multimodal'
            ret_list = train_modality([X_m_train,X_m_test],[y_m_train,y_m_test,z_m_train,t_m_train],fss,batch_size,num_epochs) 
            
            # ret_list.append(ev)
            if fss in save_dict.keys():
                save_dict[fss].append(ret_list)
            else:
                save_dict[fss]=[]
                save_dict[fss].append(ret_list)

#=================================VIDEO================================================            
            fss = 'video'
            ret_list = train_modality([X_v_train,X_v_test],[y_v_train,y_v_test,z_v_train,t_v_train],fss,batch_size,num_epochs) 
            if fss in save_dict.keys():
                save_dict[fss].append(ret_list)
            else:
                save_dict[fss]=[]
                save_dict[fss].append(ret_list)

#=================================AUDIO================================================
            fss = 'audio'
            ret_list = train_modality([X_a_train,X_a_test],[y_a_train,y_a_test,z_a_train,t_a_train],fss,batch_size,num_epochs) 
            if fss in save_dict.keys():
                save_dict[fss].append(ret_list)
            else:
                save_dict[fss]=[]
                save_dict[fss].append(ret_list)
                
#=================================AUDIOVISUAL================================================            
            fss = 'audiovis'
            ret_list = train_modality([X_av_train,X_av_test],[y_av_train,y_av_test,z_av_train,t_av_train],fss,batch_size,num_epochs) 
            if fss in save_dict.keys():
                save_dict[fss].append(ret_list)
            else:
                save_dict[fss]=[]
                save_dict[fss].append(ret_list)

#=================================PHYSIOLOGICAL===============================================
            fss = 'physiological'
            ret_list = train_modality([X_e_train,X_e_test],[y_e_train,y_e_test,z_e_train,t_e_train],fss,batch_size,num_epochs) 
            if fss in save_dict.keys():
                save_dict[fss].append(ret_list)
            else:
                save_dict[fss]=[]
                save_dict[fss].append(ret_list)


#=================================MAJORITY VOTER================================================            
            
            # # acc_11.append(ff)
            dummy = 1 if np.mean(y_e_train)>0.5 else 0
            dummy = dummy*np.ones(shape=y_e_test.shape)
            dacc = np.mean(dummy==y_e_test)
            dum.append(dacc)

    save_dict['dummy']=dum
    save_dict['baselines']=baselines
# save_dict ={'dummy':dum,'baselines':np.array(baselines),'mult':np.array(acc_m),'scl_mult':np.array(acc_mc),'scl_mult_change':np.array(acc_mcc),'mult_change':np.array(acc_mclc),'vid':np.array(acc_v),'scl_vid':np.array(acc_vc),'scl_vid_change':np.array(acc_vcc),'vid_change':np.array(acc_vclc),'aud':np.array(acc_a),'scl_aud':np.array(acc_ac),'scl_aud_change':np.array(acc_acc),'aud_change':np.array(acc_aclc),'audv':np.array(acc_av),'scl_audv':np.array(acc_avc),'scl_audv_change':np.array(acc_avcc),'audv_change':np.array(acc_avclc),'phys':np.array(acc_e),'scl_phys':np.array(acc_ec),'scl_phys_change':np.array(acc_ecc),'phys_change':np.array(acc_eclc)}
    savemat('./results/sclf/'+tw+'_arousal_'+str(trial), save_dict)
