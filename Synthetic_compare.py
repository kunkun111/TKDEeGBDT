#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:51:26 2020

@author: kunwang
"""

# Imports
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import LearnPPNSEClassifier
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.meta import LeveragingBaggingClassifier
from skmultiflow.meta import OnlineBoostingClassifier
from skmultiflow.meta import OnlineRUSBoostClassifier
import numpy as np
import arff
import pandas as pd
from skmultiflow.data.data_stream import DataStream
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve



# load .arff dataset
def load_arff(path, dataset_name, num_copy):
    file_path = path + dataset_name + '/'+ dataset_name + str(num_copy) + '.arff'
    dataset = arff.load(open(file_path), encode_nominal=True)
    return pd.DataFrame(dataset["data"])

#datasets list
#datasets = ['AGRa', 'HYP','RBF','RBFr','RTG','SEAa'] 
datasets = ['AGRa'] 
path = 'Synthetic/'
batch = 100
num_run = 15

# Set the model
ARF = AdaptiveRandomForestClassifier()
NSE = LearnPPNSEClassifier()
AWE = AccuracyWeightedEnsembleClassifier()
LEV = LeveragingBaggingClassifier()
OBC = OnlineBoostingClassifier()
RUS = OnlineRUSBoostClassifier()
model=[AWE,ARF,NSE,LEV,OBC,RUS]
model_names=['ARF','NSE','AWE','LEV','OBC','RUS']
#model_names=['NSE']


for i in range(len(datasets)):
    for j in range(len(model)):
        for num_copy in range(num_run):
            print(num_copy, '/', num_run)
            data = load_arff(path, str(datasets[i]),num_copy)
            print('dataset',str(datasets[i]),num_copy,'batch_size',batch, model_names[j])

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
                X, y = stream.next_sample(batch)
                y_pred = model[j].predict(X)
                pred = np.hstack((pred,y_pred))
                model[j].partial_fit(X, y,stream.target_values)
                n_samples += batch
    
            # evaluate
            data = data.values
            Y = data[:,-1]
            acc = accuracy_score(Y[batch:], pred[batch:])
            f1 = f1_score(Y[batch:], pred[batch:], average='macro')
            print("acc:",acc)
            print("f1:",f1)
        
            # save results
            result = np.zeros([pred[batch:].shape[0], 2])
            result[:, 0] = pred[batch:]
            result[:, 1] = Y[batch:]
            np.savetxt(str(datasets[i])+str(num_copy)+'_'+str(model_names[j])+'.out', result, delimiter=',')
    
    