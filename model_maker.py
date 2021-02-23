import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from model_tools import loadDataFrame,extractTrainingSet,createTrainingFile
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


createTrainingFile('July 20, 2016','csv_files/2018-2021DaysTrainingSet.csv')
binance_frame = pd.read_csv('csv_files/2018-2021DaysTrainingSet.csv')
binance_frame = binance_frame.iloc[:-3]

X_train,Y_train,training_price_changes,preprocessors = extractTrainingSet(binance_frame,0.01,csv=True)
lr = LogisticRegression(C=0.1,solver='sag',class_weight={0:0.5,1:0.5},max_iter=2000).fit(X_train,Y_train)

pipe_line = Pipeline([('scaler',preprocessors['scaler']),('pca',preprocessors['pca']),('lda',preprocessors['lda']),('model',lr)])
pickle.dump(pipe_line,open('pickle_files/pipe_line.pkl','wb'))