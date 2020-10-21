#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:45:25 2020

@author: kunwang
"""

# Imports

from skmultiflow.data import SEAGenerator
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import LearnPPNSEClassifier
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.meta import LeveragingBaggingClassifier
from skmultiflow.meta import OnlineBoostingClassifier
from skmultiflow.meta import OnlineRUSBoostClassifier
from skmultiflow.evaluation import EvaluatePrequential
import numpy as np
import arff
import pandas as pd
from skmultiflow.data.data_stream import DataStream
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve



# load .arff dataset
def load_arff(path, dataset_name):
    file_path = path + dataset_name + '/'+ dataset_name + '.arff'
    dataset = arff.load(open(file_path), encode_nominal=True)
    return pd.DataFrame(dataset["data"])


# datasets list
#datasets = [
    #'usenet1', 'usenet2','weather','spam_corpus_x2_feature_selected',
    #'elecNorm','airline'] 
datasets = ['usenet2'] 
#batch = [40,40,365,100,100,100]
batch = [40]
path = 'Realworld/'

# Set the model
ARF = AdaptiveRandomForestClassifier()
NSE = LearnPPNSEClassifier()
AWE = AccuracyWeightedEnsembleClassifier()
LEV = LeveragingBaggingClassifier()
OBC = OnlineBoostingClassifier()
RUS = OnlineRUSBoostClassifier()
model=[ARF,NSE,AWE,LEV,OBC,RUS]
model_names=['ARF','NSE','AWE','LEV','OBC','RUS']


for i in range(0, len(datasets)):
    for j in range(0,len(model)):
        data = load_arff(path, str(datasets[i]))
        print('dataset',str(datasets[i]),'batch_size',batch[i],model_names[j])

        # data transform
        stream = DataStream(data)
    
        # Setup variables to control loop and track performance
        n_samples = 0
        correct_cnt = 0
        max_samples = data.shape[0]

        # Train the classifier with the samples provided by the data stream
        pred = np.empty(0)
        np.random.seed(0)
        while n_samples < max_samples and stream.has_more_samples():
            X, y = stream.next_sample(batch[i])
            y_pred = model[j].predict(X)
            pred = np.hstack((pred,y_pred))
            model[j].partial_fit(X, y,stream.target_values)
            n_samples += batch[i]
    
        # evaluate
        data = data.values
        Y = data[:,-1]
        acc = accuracy_score(Y[batch[i]:], pred[batch[i]:])
        f1 = f1_score(Y[batch[i]:], pred[batch[i]:], average='macro')
        print (Y[batch[i]:].shape, pred[batch[i]:].shape)
        print("acc:",acc)
        print("f1:",f1)
        
        # save results
        result = np.zeros([pred[batch[i]:].shape[0], 2])
        result[:, 0] = pred[batch[i]:]
        result[:, 1] = Y[batch[i]:]
        np.savetxt(str(datasets[i])+'_'+str(model_names[j])+'.out', result, delimiter=',')
    
    