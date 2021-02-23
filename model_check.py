import pickle
import pandas as pd
import numpy as np
from model_tools import loadDataFrame, processFrame, testModel, plotResults

validation_frame = loadDataFrame(pull_date='January 10, 2021')
validation_time = validation_frame['closeTime']
validation_frame,price_changes = processFrame(validation_frame,csv=False)

X_set = np.array(validation_frame)
pipe_line = pickle.load(open('pickle_files/pipe_line.pkl', 'rb'))
predictions = list(pipe_line.predict_proba(X_set))
accuracy_intervals,principal_intervals,control_prinicpal_intervals = testModel(
    price_changes=price_changes,
    probability_cutoff=0.6,
    predictions=predictions
    )

validation_frame['prediction'],validation_frame['priceChanges'],validation_frame['time'] = predictions,price_changes,validation_time
validation_frame.to_csv('csv_files/check_file.csv')
plotResults(principal_intervals,control_prinicpal_intervals,accuracy_intervals)