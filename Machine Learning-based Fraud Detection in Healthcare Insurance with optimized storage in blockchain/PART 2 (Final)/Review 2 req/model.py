import re
import os
import math
import pickle
import sklearn
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from datetime import date
from tqdm.notebook import tqdm
from scipy.sparse import hstack, vstack
from scipy.stats import randint as sp_randint, uniform as sp_uniform
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline as sklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, confusion_matrix, f1_score, roc_curve, auc, balanced_accuracy_score, matthews_corrcoef
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from flask import Flask, request , jsonify
# from sklearn_json import sklearn_to_json
import pickle

pd.options.display.max_columns = None # To display all the columns of a Dataframe.
warnings.filterwarnings('ignore')

data = pd.read_csv('C:\\Users\\Dell\\Desktop\\pw2\\fraud-model\\ds1\\featureEngg_data.csv')
dataY = data['PotentialFraud'] # Class Label
dataX = data.drop(columns='PotentialFraud') # Features
xTrain, xTest, yTrain, yTest = train_test_split(dataX, dataY, test_size=0.2, stratify=dataY)
# print('Shape of the Train Dataset (Features): ', xTrain.shape)
# print('Shape of the Train Dataset (Class Label)', yTrain.shape)
# print('Shape of the Test Dataset (Features): ', xTest.shape)
# print('Shape of the Test Dataset (Class Label)', yTest.shape)



# rndmUndrSampler = RandomUnderSampler(sampling_strategy=1)
# Create Random Undersampled Dataset by calling the 'fit_resample' method
# xTrainRndmUS, yTrainRndmUS = rndmUndrSampler.fit_resample(xTrain, yTrain)
# print('Shape of the Train Dataset (Features) after Random Undersampling: ', xTrainRndmUS.shape)
# print('Shape of the Train Dataset (Class Label) Random Undersampling', yTrainRndmUS.shape)
# print('After Random Undersampling, count of unique labels:')
# yTrainRndmUS.value_counts()



if os.path.isfile('C:\\Users\\Dell\\Desktop\\pw2\\fraud-model\\ds1\\xTrainSMOTE.csv') and os.path.isfile('C:\\Users\\Dell\\Desktop\\pw2\\fraud-model\\ds1\\yTrainSMOTE.csv'):

    xTrainSMOTE = pd.read_csv('C:\\Users\\Dell\\Desktop\\pw2\\fraud-model\\ds1\\xTrainSMOTE.csv')
    yTrainSMOTE = pd.read_csv('C:\\Users\\Dell\\Desktop\\pw2\\fraud-model\\ds1\\yTrainSMOTE.csv')

else:

    # Create an SMOTE object
    smote = SMOTE(sampling_strategy=1)

    # Create Oversampled Dataset by calling the 'fit_resample' method
    xTrainSMOTE, yTrainSMOTE = smote.fit_resample(xTrain, yTrain)

    ## Save the Dataset
    xTrainSMOTE.to_csv('C:\\Users\\Dell\\Desktop\\pw2\\fraud-model\\ds1\\xTrainSMOTE.csv', index=False)
    yTrainSMOTE.to_csv('C:\\Users\\Dell\\Desktop\\pw2\\fraud-model\\ds1\\yTrainSMOTE.csv', index=False)

# print('Shape of the Train Dataset (Features) after SMOTE Oversampling: ', xTrainSMOTE.shape)
# print('Shape of the Train Dataset (Class Label) SMOTE Oversampling', yTrainSMOTE.shape)
# yTrainSMOTE = yTrainSMOTE.squeeze(axis=1) # Convert the DataFrame to Series for Class Label
# print('After SMOTE Oversampling, count of unique labels:')
# yTrainSMOTE.value_counts()



class ResponseEncoder(BaseEstimator, TransformerMixin):
    '''
    Class to do Response Encoding for the Categorical features.
    This class can be used in the sklearn's Pipeline to avoid data leakdage issues
    '''
    def __init__(self, categoricalFeatures, className):
        '''
        Function to initialize the class members

        Parameter(s):
        ------------
        categoricalFeatures: list
            List of features for which the response encoding has to be done to generate new features.
        className: str
            Name of the Class
        '''
        self.categoricalFeatures = categoricalFeatures # Categorical Features for which Response Encoding has to be done.
        self.responseTable = dict() # Dictionary to store the key:value pair with the 'key' being the categorical feature
        # name and its 'value' as the dataFrame containing the Response Table.
        self.className = className
        self.classCount = 0 # Number of unique class labels. For binary classification, it will be 2.

    def fit(self, X, y):
        '''
        Function called on a Dataset (usually Train Dataset) and Class Label to generate Response Encoded Table.
        This function is called only for the train dataset and not for any cv/test dataset to avoid data leakage.

        Parameters:
        ----------
        X: pandas.core.frame.DataFrame
            DataFrame on which the Response Encoding has to be carried out.
        y: pandas.core.series.Series
            Class Labels of the DataFrame
        '''

        # Prepare a DataFrame based on the given the dataset and class label, containing just the input features and class labels
        data = pd.DataFrame()

        # Add the features to the dataframe
        for col in self.categoricalFeatures:
            data[col] = X[col]

        # Add the Class Label to the dataframe
        data['PotentialFraud'] = y

        # Store the count of total unique class labels in the class variable 'classCount'
        self.classCount = len(data['PotentialFraud'].unique())

        # Iterate through each of the categorical features for which Response Encoding has to be done
        for feature in self.categoricalFeatures:

            # Dictionary to store the unique categorical feature names and encoded feature name as keys and their values as
            # list of their corresponding values, class counts and count probabilities
            dictResponseTable = dict()

            uniqueFeatValues = np.sort(X[feature].unique()) # Array of unique feature values
            uniqueClassLabels = np.sort(data['PotentialFraud'].unique()) # Array of unique class labels

            # Iterate through each of the categorical feature values and generate the Response Table
            for featureVal in uniqueFeatValues:

                countClass = list() # List to store the count/frequency of a Class label for a particular feature.
                probClass = list() # List to store the probability of occurence of a class label for a particular feature.

                # Loop through the unique Class Labels and find the count of the feature value
                for label in uniqueClassLabels:

                    # Append the frequency of Class Label 'label' for the feature 'featureVal' to the list 'countClass'
                    countClass.append(data[(data[feature] == featureVal) & (data[self.className] == label)][self.className].count())


                # Loop through the unique Class Labels and find the likelihood probability of the feature value
                for label in uniqueClassLabels:

                    # Append the likelihood probability of the occurence of the class 'label' for the feature 'featureVal'
                    probClass.append(countClass[label]/sum(countClass))


                # Prepare a dictionary having keys as features (original and new features) and their values as
                # feature value (for original features), class counts and class probabilities

                # Check if the key already exist or not in the dictionary. If not, create it
                if (feature not in dictResponseTable.keys()):
                    dictResponseTable[feature] = []
                dictResponseTable[feature].append(featureVal) # Append the current iteration's feature value 'feature'

                # For each unique class label, add the class label and probability to the corresponding keys in the dictionary
                for label in uniqueClassLabels:

                    if (feature + 'Class' + str(label) not in dictResponseTable.keys()):
                        dictResponseTable[feature + 'Class' + str(label)] = []
                    if (feature + '_' + str(label) not in dictResponseTable.keys()):
                        dictResponseTable[feature + '_' + str(label)] = []
                    dictResponseTable[feature + 'Class' + str(label)].append(countClass[label])
                    dictResponseTable[feature + '_' + str(label)].append(probClass[label])

            # Prepare and store the Response Table in the dictionary 'self.responseTable'
            self.responseTable[feature] = pd.DataFrame(dictResponseTable)

        return self

    def transform(self, X, y= None):
        '''
        Function called on a Dataset (Train/Test Dataset) and/or Class Label to generate Response Encoded Features.
        This is called to avoid any data leakage. This uses the Response Table already prepared by the fit() method
        and does not consider the test dataset.

        Parameters:
        ----------
        X: pandas.core.frame.DataFrame
            DataFrame on which the Response Encoding has to be done.
        y: pandas.core.series.Series
            Class Labels of the DataFrame
        '''

        # Get a copy of the input dataframe such the input dataframe is not modified.
        xEncoded = X.copy()

        listResponseEncFeat = list() # List to store the names of the response encoded features.

        # Iterate through each of the categorical features for which Response Encoding has to be done
        for feature in self.categoricalFeatures:

            # Merge with the Response Table Dataframe
            xEncoded = pd.merge(left=xEncoded, right=self.responseTable[feature], how='left', on=feature)

            # Form the list of the features (original and class count) to be dropped from the Dataset
            listResponseEncFeat.extend([col for col in list(self.responseTable[feature].columns) if '_' not in col])

        # Fill the values for the datapoints which are not present in the Response Table with equal probabilities of the class
        xEncoded.fillna(1/self.classCount)

        # Drop the original features and class count features. Keep only the response encoded features having class prob.
        xEncoded.drop(columns=listResponseEncFeat, inplace=True)

        # Convert the values of the dataframe to numeric.
        xEncoded.apply(pd.to_numeric)

        # Fill the empty/missing values with 0.
        xEncoded.fillna(0, inplace=True)

        # Return this DataFrame with all the numerical features and the response encoded features for the categorical features
        return xEncoded

# responseEnc = ResponseEncoder(categoricalFeatures=['State', 'Country'], className='PotentialFraud')
# # Fit the 'ResponseEncoder' class object on only Train Data
# responseEnc.fit(xTrain, yTrain)
# xTrainRE = responseEnc.transform(xTrain) # Train Data
# xTestRE = responseEnc.transform(xTest) # Test Data
# print('Shape of the Train Data before doing response encoding of \'State\' and \'Country\' Features:',  xTrain.shape)
# print('Shape of the Test Data before doing response encoding of \'State\' and \'Country\' Features:',  xTest.shape)
# print('Shape of the Train Data after doing response encoding of \'State\' and \'Country\' Features:',  xTrainRE.shape)
# print('Shape of the Test Data after doing response encoding of \'State\' and \'Country\' Features:',  xTestRE.shape)
# xTrainRE[[col for col in xTrainRE.columns if re.match(r'(State|Country)[_]*[0-9]*', col)]].head()
    


class Standardize(BaseEstimator, TransformerMixin):
    '''
    Class to do standardization of the numerical features.
    This class can be used in the sklearn's Pipeline to avoid data leakage issues
    '''
    def __init__(self, numericalFeatures):
        '''
        Function to initialize the class members

        Parameter(s):
        ------------
        numericalFeatures: list
            List of numerical features to be standardized.
        '''
        self.numericalFeatures = numericalFeatures # Numerical Features to be standardized.
        self.standardScaler = StandardScaler() # Object of StandardScaler.

    def fit(self, X, y=None):
        '''
        Function called on a Dataset (usually Train Dataset) to fit the train dataset.
        This function is called only for the train dataset and not for any cv/test dataset to avoid data leakage.

        Parameters:
        ----------
        X: pandas.core.frame.DataFrame
            Dataset to be considered for standardization.
        '''

        # Fit the StandardScaler object only on the numerical features.
        self.standardScaler.fit(X[self.numericalFeatures])

        return self

    def transform(self, X, y=None):
        '''
        Function called on a Dataset (Train/Test Dataset) to standardize the data based on the Train Dataset.
        This is called to avoid any data leakage. This uses the standardScaler object already prepared by the fit() method
        and does not consider the test dataset.

        Parameters:
        ----------
        X: pandas.core.frame.DataFrame
            Dataset to be standardized.
        '''

        # Create a copy of the dataframe so that it does not modify the input dataframe
        xStandardized = X.copy()

        # Standardize the numerical features of the dataset.
        xStandardized[self.numericalFeatures] = self.standardScaler.transform(xStandardized[self.numericalFeatures])

        # Return the standardized dataset
        return xStandardized

featuresToStd = ['Race', 'ClaimSettlementDelay', 'TreatmentDuration', 'Age', 'TotalClaimAmount',
                 'IPTotalAmount', 'OPTotalAmount', 'UniquePhysCount', 'PhysRoleCount']

# Create an object of the 'Standardize' class
# std = Standardize(numericalFeatures=featuresToStd)
# # Fit the 'Standardize' class object on only Train Data
# std.fit(xTrain, yTrain)
# xTrainStd = std.transform(xTrain) # Train Data
# xTestStd = std.transform(xTest) # Test Data
# xTrainStd.head()



def getLogloss(yActual, yPredProb, datasetType='Test'):
    '''
    Finds and displays the Log Loss

    Parameters:
    ----------
    yActual: array-like
        Ground truth (correct) class labels for 'n' samples.
    yPredProb: array-like
        Predicted probabilities, as returned by a model's predict_proba method.
    datasetType: str
        Type of Dataset: Test or Train.
    '''

    logloss = log_loss(y_true=yActual, y_pred=yPredProb)

    print('Log-loss of the Model on ', datasetType ,' Dataset: ', logloss)

def getF1Score(yActual, yPred):
    '''
    Calculates and displays the F1-Score

    Parameters:
    ----------
    yActual: array-like
        Ground truth (correct) class labels for 'n' samples.
    yPred: array-like
        Predicted class labels for 'n' samples.
    '''
    f1Score = f1_score(y_true=yActual, y_pred=yPred)
    print('F1-Score of the Model on Test Data: ', f1Score)

def plotPerformanceMatrix(yActual, yPred):
    '''
    Function to compute Confusion, Precision and Recall Matrix and plot them.

    Parameters:
    ----------
    yActual: array-like
        Ground truth (correct) class labels for 'n' samples.
    yPred: array-like
        Predicted class labels for 'n' samples.
    '''

    # Get Confusion Matrix based on the input 'yActual' and 'yPred'.
    confusionMatrix = confusion_matrix(y_true=yActual, y_pred=yPred)

    # Compute the Precision Matrix
    precisionMatrix = (confusionMatrix/confusionMatrix.sum(axis=0))
    # Divide each element of the confusion matrix with the sum of the elements in that column (total predicted value)

    # Compute the Recall Matrix
    recallMatrix = (confusionMatrix.T/confusionMatrix.sum(axis=1)).T
    # Divide each element of the confusion matrix with the sum of the elements in that row (total actual values)

    plt.figure(figsize=(20,5))

    # Plot the Confusion Matrix.
    plt.subplot(131)
    sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Reds')
    plt.title('Confusion Matrix on Test Data', fontsize=20)
    plt.xlabel('Predicted Value', fontsize=15) # Label on the x-axis
    plt.ylabel('Actual Values', fontsize=15) # Label on the y-axis
    plt.xticks(ticks=[0.5, 1.5], labels=['Predicted: NO (Non-fraud)', 'Predicted: YES (Fraud)'])
    plt.yticks(ticks=[0.5, 1.5], labels=['Actual: NO (Non-fraud)', 'Actual: YES (Fraud)'], rotation=0)

    # Plot the Precision Matrix.
    plt.subplot(132)
    sns.heatmap(precisionMatrix, annot=True, fmt='.3f', cmap=sns.light_palette('green'))
    plt.title('Precision Matrix on Test Data', fontsize=20)
    plt.xlabel('Predicted Value', fontsize=15) # Label on the x-axis
    plt.xticks(ticks=[0.5, 1.5], labels=['Predicted: NO (Non-fraud)', 'Predicted: YES (Fraud)'])

    # Plot the Recall Matrix.
    plt.subplot(133)
    sns.heatmap(recallMatrix, annot=True, fmt='.3f', cmap='Blues')
    plt.title('Recall Matrix on Test Data', fontsize=20)
    plt.xlabel('Predicted Value', fontsize=15) # Label on the x-axis
    plt.xticks(ticks=[0.5, 1.5], labels=['Predicted: NO (Non-fraud)', 'Predicted: YES (Fraud)'])
    plt.show()

    totalDatapoints = len(yActual) # Total number of Datapoints
    print('Percentage of misclassified points (Test Data): ', ((totalDatapoints - np.trace(confusionMatrix))/totalDatapoints)*100, '%')
    print('Sum of columns in the Precision Matrix: ', precisionMatrix.sum(axis=0))
    print('Sum of rows in the Recall Matrix: ', recallMatrix.sum(axis=1))

    # Find Total Negatives and Total Positives (Actual/ ground truth)
    totalNegative, totalPositive = tuple(confusionMatrix.sum(axis=1))

    # Find True Positive, False Positive, False Negative and True Positive
    tn, fp, fn, tp = confusionMatrix.ravel()

    # Calculate the FPR and FNR
    fpr = fp / totalNegative
    fnr = fn / totalPositive
    print('False Positive Rate (FPR) on Test Data: ', fpr)
    print('False Negative Rate (FNR) on Test Data: ', fnr)

    # Get the BACC Score
    bacc = balanced_accuracy_score(y_true=yActual, y_pred=yPred)
    print('Balanced Accuracy Score (BACC) on Test Data: ', bacc)

    # Get the MCC Score
    mcc = matthews_corrcoef(y_true=yActual, y_pred=yPred)
    print('Matthew\'s Correlation Coefficient (MCC) on Test Data: ', mcc)

def getProbEstimates(yPredProb, positiveClass=True, batchSize=1000):
    '''
    Returns the Probability Estimates of the data as predicted by the model, for the given Class.
    Class can be negative (0) or positive (1).
    This function returns the Probability Estimates of the positive class (1) by default.

    Parameters:
    -----------
    yPredProb: array-like
        Contains the Predicted Probability Scores of both -ve (0) and +ve (1) class.
    positiveClass: bool
        Flag to decide whether to find the Probability Estimates for a positive class (True) or negative class (False).
    batchSize: int
        Size of the batch to precessed at a time.
    '''

    # Check if the input Predicted variable is a Probabilty Estimates or not
    if len(yPredProb.shape) == 1:
        print('Please pass the Probability Estimates')
        return

    probEstimates = list() # Variable to store the Predicted Probability Estimates
    classIndex = 1 # Index of the Class (For -ve class, it should be 0 and for +ve class, it should be 1)
    if positiveClass == False:
        classIndex = 0
    batchsize = batchSize # Process a batch of given size.
    totalCnt = yPredProb.shape[0] # Total number of Datapoints
    datapointCount = totalCnt - totalCnt%batchsize # Total no. of datapoints minus the last batch of datapoint

    # Loop through the dataset batch-wise
    for i in range(0, datapointCount, batchsize):
        probEstimates.extend(yPredProb[i : i + batchsize][:,classIndex]) # Add Probability Estimates of +ve class

    # Find the Probability Estimate for the remaining last batch
    if totalCnt%batchsize != 0:
        probEstimates.extend(yPredProb[datapointCount:][:,classIndex])

    return probEstimates

def plotROC(yTestActual, yTestPredProb, yTrainActual=None, yTrainPredProb=None):
    '''
    Plots the ROC and calculates the AUC

    Parameters:
    ----------
    yTestActual: array-like
        Ground truth (correct) class labels for 'n' samples on Test Dataset.
    yTestPredProb: array-like
        Predicted probabilities, as returned by a model's predict_proba method on Test Dataset.
    yTrainActual: array-like
        Ground truth (correct) class labels for 'n' samples on Train Dataset.
    yTrainPredProb: array-like
        Predicted probabilities, as returned by a model's predict_proba method on Train Dataset.
    '''

    # Check if the input Predicted variable is a Probabilty Estimates or not
    if len(yTestPredProb.shape) == 1:
        print('Please pass the correct Probability Estimates')
        return

    # Calculate the Predicted Probability Estimates for the positive class of Test Data
    yTestPredPE = getProbEstimates(yTestPredProb)

    # Find the FPR, TPR and Threshold values for the Test Data
    fprTest, tprTest, thresholdTest = roc_curve(yTestActual, yTestPredPE)

    # Find the AUC
    areaTest = auc(fprTest, tprTest)

    # If Train Dataset to also to be considered for plotting ROC
    if (type(yTrainActual) != type(None) and type(yTrainPredProb) != type(None)):

        # Check if the input Predicted variable is a Probabilty Estimates or not
        if len(yTrainPredProb.shape) == 1:
            print('Please pass the correct Probability Estimates')
            return

        # Calculate the Predicted Probability Estimates for the positive class of Train Data
        yTrainPredPE = getProbEstimates(yTrainPredProb)

        # Find the FPR, TPR and Threshold values for the Train Data
        fprTrain, tprTrain, thresholdTrain = roc_curve(yTrainActual, yTrainPredPE)

        # Find the AUC
        areaTrain = auc(fprTrain, tprTrain)

    # Plot the ROC Curve
    # If Train Dataset to also to be considered for plotting ROC
    if (type(yTrainActual) != type(None) and type(yTrainPredProb) != type(None)):

        plt.plot(fprTrain, tprTrain, label= 'Train AUC: ' + str(areaTrain))

    plt.plot(fprTest, tprTest, label= 'Test AUC: ' + str(areaTest))
    plt.legend()
    # If Train Dataset to also to be considered for plotting ROC
    if (type(yTrainActual) != type(None) and type(yTrainPredProb) != type(None)):

        plt.title('ROC Curve of Train and Test Data', fontsize=20)

    else:

        plt.title('ROC Curve of Test Data', fontsize=20)

    plt.xlabel('FPRs', fontsize=15)
    plt.ylabel('TPRs', fontsize=15)
    plt.grid()
    plt.show()

def showPerformanceMetrics(model, xTestData, yTestData, xTrainData=None, yTrainData=None):
    '''
    Calls all the above defined performance metrics functions 'getLogloss', 'getF1Score', 'plotPerformanceMatrix' and 'plotROC'.

    Parameters:
    ----------
    model: Classifier Model
        Classifier Model trained to do classification.
    xTestData: DataFrame
        Test Dataset containing the features.
    yTestData: Series
        Test Dataset containing only the Class Labels.
    xTrainData: DataFrame
        Train Dataset containing the features.
    yTrainData: Series
        Train Dataset containing only the Class Labels.
    '''

    # Predict the Class Labels of the Test Dataset using the given Model.
    yTestPred = model.predict(xTestData)

    # Predict the Probability Estimates of both the classes of the Test Dataset using the given Model.
    yTestPredProba = model.predict_proba(xTestData)

    # if (type(xTrainData) != type(None)):

    #     # Predict the Class Labels of the Train Dataset using the given Model.
    #     yTrainPred = model.predict(xTrainData)

    #     # Predict the Probability Estimates of both the classes of the Train Dataset using the given Model.
    #     yTrainPredProba = model.predict_proba(xTrainData)

        # Call the 'getLogloss' function to get the Log-loss of the Model on the given Train dataset.
        # getLogloss(yTrainData, yTrainPredProba, datasetType='Train')

    # Call the 'getLogloss' function to get the Log-loss of the Model on the given Test dataset.
    # getLogloss(yTestData, yTestPredProba)

    # Call the 'getF1Score' function to get the F1-Score of the Model on the given Test dataset.
    getF1Score(yTestData, yTestPred)

    # Call the 'plotPerformanceMatrix' function to plot Confusion, Precision and Recall Matrices
    # and display various metrics based on these matrices
    # plotPerformanceMatrix(yTestData, yTestPred)

    # if (type(xTrainData) != type(None)):

    #     # Call the 'plotROC' function to plot the ROC Curve and display the AUC value of the Model.
    #     plotROC(yTestData, yTestPredProba, yTrainData, yTrainPredProba)

    # else:

    #     # Call the 'plotROC' function to plot the ROC Curve and display the AUC value of the Model.
    #     plotROC(yTestData, yTestPredProba)


# Create Pipeline:
# 1. Standardization of numerical features.
# 2. Response Encoding of 'State' and 'Country' features.
# 3. RandomizedSearchCV using XGBoost Classifier
pipelineXGB4 = sklearnPipeline(steps=[
    ('standardization', Standardize(numericalFeatures=featuresToStd)),
    ('responseEncoding', ResponseEncoder(categoricalFeatures=['State', 'Country'], className='PotentialFraud')),
    ('xgboost', XGBClassifier(subsample=1, n_estimators=1000, max_depth=10, learning_rate=0.1, colsample_bytree=0.3))
])

# Fit the Pipeline on the Train Data with SMOTE Oversampling
pipelineXGB4.fit(xTrainSMOTE, yTrainSMOTE)
showPerformanceMetrics(model=pipelineXGB4, xTestData=xTest, yTestData=yTest, xTrainData=xTrainSMOTE, yTrainData=yTrainSMOTE)
df = pd.DataFrame()
final = pd.DataFrame()
df = xTest.head()
print(len(df))
print(pipelineXGB4.predict(xTest.iloc[0]))
# print(len)
final = pipelineXGB4.predict(xTest)
final = df.assign(result = pipelineXGB4.predict(xTest.head()))
print(final['result'])

with open('model.pkl', 'wb') as file:
    pickle.dump(pipelineXGB4, file)

# print(type(final))
# print(str(list(final)))

# print(xTest)
# model_json = sklearn_to_json(model)

# # Save the JSON to a file
# with open('model.json', 'w') as f:
#     f.write(model_json)

# app = Flask(__name__)
# @app.route("/flask", methods=['GET'])
# def index():
#     return str(final)

# @app.route('/flask-endpoint', methods=['GET','POST'])
# def flask_endpoint():
#     # data = request.json
#     # print(data)
#     # # Process the data and return a response
#     # return jsonify({'message': 'Data received by Flask'})
#     if request.method == 'GET':
#         # Handle GET request
#         return jsonify({'message': 'GET request received by Flask'})
#     elif request.method == 'POST':
#         # Handle POST request
#         data = request.json
#         print(data)
#         # Process the data and return a response
#         return jsonify({'message': 'POST request received by Flask'})

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)