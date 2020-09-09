#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 14:32:07 2020

@author: kunwang
"""
# Imports

from skmultiflow.data import SEAGenerator
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import LearnPPNSEClassifier
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.meta import LeveragingBaggingClassifier
from skmultiflow.meta import OnlineBoostingClassifier
from skmultiflow.evaluation import EvaluatePrequential
import numpy as np
import arff
import pandas as pd
from skmultiflow.data.data_stream import DataStream


# load .arff dataset
def load_arff(path, dataset_name):
    file_path = path + dataset_name + '/'+ dataset_name + '.arff'
    dataset = arff.load(open(file_path), encode_nominal=True)
    return pd.DataFrame(dataset["data"])


# datasets list
datasets = [
    'usenet1', 'usenet2','weather','spam_corpus_x2_feature_selected',
    'elecNorm','airline'] 

# batch list
batch = [40,40,365,100,100,100]
path = 'Realworld/'

for i in range(0, len(datasets)):
    data = load_arff(path, str(datasets[i]))
    print('dataset',str(datasets[i]),'batch_size',batch[i])

    # data transform
    stream = DataStream(data)
    
    # Set the model
    np.random.seed(0)
    ARF = AdaptiveRandomForestClassifier()
    NSE = LearnPPNSEClassifier()
    AWE = AccuracyWeightedEnsembleClassifier()
    LEV = LeveragingBaggingClassifier()
    OBC = OnlineBoostingClassifier()

    # Set the evaluator
    evaluator = EvaluatePrequential(n_wait=batch[i],
                                    max_samples=data.shape[0],
                                    batch_size=batch[i],
                                    pretrain_size=batch[i],
                                    metrics=['accuracy', 'f1','true_vs_predicted'],
                                    output_file=str(datasets[i])+'.csv')

    # Run evaluation
    evaluator.evaluate(stream=stream, model=[ARF,NSE,AWE,LEV,OBC], model_names=['ARF','NSE','AWE','LEV','OBC'])

