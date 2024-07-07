import pandas as  pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from flask import Flask, request, jsonify


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



data = pd.read_csv('C:\\Users\\Dell\\Desktop\\pw2\\fraud-model\\ds1\\featureEngg_data.csv')
dataY = data['PotentialFraud'] # Class Label
dataX = data.drop(columns='PotentialFraud') # Features
xTrain, xTest, yTrain, yTest = train_test_split(dataX, dataY, test_size=0.2, stratify=dataY)
# xTrainSMOTE = pd.read_csv('C:\\Users\\Dell\\Desktop\\pw2\\fraud-model\\ds1\\xTrainSMOTE.csv')
# yTrainSMOTE = pd.read_csv('C:\\Users\\Dell\\Desktop\\pw2\\fraud-model\\ds1\\yTrainSMOTE.csv')

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

prediction = model.predict(xTest.head())
print(prediction)
# showPerformanceMetrics(model=model, xTestData=xTest, yTestData=yTest, xTrainData=xTrainSMOTE, yTrainData=yTrainSMOTE)

app = Flask(__name__)
@app.route("/flask", methods=['GET'])
def index():
    return str(prediction)

@app.route('/flask-endpoint', methods=['GET','POST'])
def flask_endpoint():
    # data = request.json
    # print(data)
    # # Process the data and return a response
    # return jsonify({'message': 'Data received by Flask'})
    if request.method == 'GET':
        # Handle GET request
        return jsonify({'message': 'GET request received by Flask'})
    elif request.method == 'POST':
        # Handle POST request
        data = request.json
        print(data)
        # Process the data and return a response
        return jsonify({'message': 'POST request received by Flask'})

if __name__ == '__main__':
    app.run(port=5000, debug=True)