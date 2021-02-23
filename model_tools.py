import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import csv
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from binance.client import Client
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

api_key = "<>"
api_secret = "<>"
client = Client(api_key, api_secret)

def getBinanceData(pull_date):
    binance_data = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_1DAY, pull_date)
    return binance_data

def processData(candle_data):
    column_names = [
    "openTime","open","high","low","close","volume","closeTime","quoteAssetVolume",
    "trades","takerBaseAssetVolume","takerQuoteAssetVolume","ignored"]
    column_values = {}
    for name in column_names:
        column = {name:[]}
        column_values.update(column)
    for row in candle_data:
        counter = 0
        for item in row:
            column_values[column_names[counter]].append(float(item))
            counter += 1
    return column_values

def loadDataFrame(pull_date):
    binance_data = getBinanceData(pull_date)
    processed_binance_data = processData(binance_data)
    return pd.DataFrame(data=processed_binance_data)

def addFeatures(binance_frame):
    binance_frame['volumeTradeRatio'] = [volume/trade for volume,trade in zip(binance_frame['volume'],binance_frame['trades'])]
    binance_frame['differenceVolumeRatio'] = [((closing_value-opening_value)/opening_value)/trade_ratio for opening_value,closing_value,trade_ratio in zip(binance_frame['open'],binance_frame['close'],binance_frame['volumeTradeRatio'])]
    binance_frame['highLowVolumeTradeRatio'] = [((high-low)/low)/trade_ratio for high,low,trade_ratio in zip(binance_frame['high'],binance_frame['low'],binance_frame['volumeTradeRatio'])]
    return binance_frame

def processFrame(binance_frame,csv):
    price_changes = [
    (closing_value-opening_value)/opening_value for opening_value,closing_value in zip(binance_frame['open'],binance_frame['close'])
    ]
    price_changes.pop(0)
    binance_frame = binance_frame.drop(len(binance_frame)-1,axis=0)
    if csv:
        binance_frame = binance_frame.drop(['ignored','openTime','closeTime','Unnamed: 0'],axis=1)
    else:
        binance_frame = binance_frame.drop(['ignored','openTime','closeTime'],axis=1)
    binance_frame = addFeatures(binance_frame)
    return binance_frame,price_changes

def createTrainingFile(pull_date,csv_name):
    binance_frame = loadDataFrame(pull_date)
    binance_frame.to_csv(csv_name)

def extractTrainingSet(binance_frame,percent_cut_off,csv):
    binance_frame, price_changes = processFrame(binance_frame,csv)
    X_set = np.array(binance_frame)
    X_train,Y_train = [], []
    training_price_changes = []
    desired_vectors,contrast_vectors = 0,0
    for price_change,feature in zip(price_changes,X_set):
        if price_change > percent_cut_off:
            training_price_changes.append(price_change)
            X_train.append(feature)
            Y_train.append(1)
            desired_vectors += 1
        if price_change < percent_cut_off and contrast_vectors < desired_vectors:
            training_price_changes.append(price_change)
            X_train.append(feature)
            Y_train.append(0)
            contrast_vectors += 1
    X_train,Y_train,training_price_changes,preprocessors = preProcess(X_train,Y_train,training_price_changes)
    return X_train,Y_train,training_price_changes,preprocessors

def preProcess(X_train,Y_train,training_price_changes):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    pca = PCA(n_components=7)
    X_train = pca.fit_transform(X_train,Y_train)
    lda = LinearDiscriminantAnalysis()
    X_train = lda.fit_transform(X_train,Y_train)
    preprocessors = {'scaler':scaler,'pca':pca,'lda':lda}
    return np.array(X_train),np.array(Y_train),np.array(training_price_changes),preprocessors

def testModel(price_changes,probability_cutoff,predictions):
    accuracy_intervals,principal_intervals,control_prinicpal_intervals = [],[],[]
    positive_intervals,prediction_count = 0,0
    principal,control_principal = 96.28899,96.28899
    transaction_fee= 0.0001
    # principal,control_principal = 10000,10000
    # transaction_fee= 0.0005
    prior_prediction = 0
    for prediction,price_change in zip(predictions,price_changes):
        positive_probability = prediction[1]
        control_principal = (control_principal * price_change) + control_principal
        control_prinicpal_intervals.append(control_principal)
        if positive_probability > probability_cutoff:
            prediction = 1
        else:
            prediction = 0
        if prediction == 1:
            prediction_count += 1
            if price_change > 0:
                positive_intervals += 1
            if prior_prediction == 0:
                principal = principal - (principal * transaction_fee) + (principal - (principal * transaction_fee)) * price_change
            else:
                principal = (principal * price_change) + principal
        else:
            if prior_prediction == 1:
                principal = principal - (principal * transaction_fee)
        principal_intervals.append(principal)
        if prediction_count != 0:
            accuracy_intervals.append(positive_intervals/prediction_count)
        prior_prediction = prediction
    print("Accuracy: " + str(positive_intervals/prediction_count))
    print("Positive Days: " + str(positive_intervals))
    print("Predicted Days: " + str(prediction_count))
    print("Principal: " + str(principal))
    print("Control Princiapl: " + str(control_principal))
    return accuracy_intervals,principal_intervals,control_prinicpal_intervals

def recordLiveResults(latest_entry_info,prediction):
    with open('csv_files/result_file2.csv', mode='a') as result_file:
        result_writer = csv.writer(result_file, delimiter=',')
        result_writer.writerow([list(latest_entry_info['closeTime'])[0],(list(latest_entry_info['close'])[0]-list(latest_entry_info['open'])[0])/list(latest_entry_info['open'])[0],prediction])

def plotResults(principal_intervals,control_prinicpal_intervals,accuracy_intervals):
    plt.plot(range(0,len(principal_intervals)),principal_intervals,color='green')
    plt.plot(range(0,len(control_prinicpal_intervals)),control_prinicpal_intervals,color='red')
    plt.title('Binance Model')
    plt.xlabel('Days')
    plt.ylabel('Principal and Control Principal')
    plt.show()
    plt.plot(range(0,len(accuracy_intervals)),accuracy_intervals,color='green')
    plt.title('Binance Model')
    plt.xlabel('Days')
    plt.ylabel('Accuracy')
    plt.show()

def plot_decision_regions(X,y,classifier,resoltuion=0.02):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resoltuion),np.arange(x2_min,x2_max,resoltuion))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],
                    y=X[y==cl,1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black'
                    )
        