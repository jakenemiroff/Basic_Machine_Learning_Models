import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

#function to construct a dataframe - read data from files
def buildDataFrame(listOfFiles):

    dataFrameList = [pd.read_csv(file, sep='\n', header=None) for file in listOfFiles]
    dataFrame = pd.concat(dataFrameList, axis=1)
    dataFrame.columns = ['X', 'Y']
    return dataFrame

#function to plot the inputted data as a scatter plot
def plotDataFrame(dataFrame):

    dataFrame.plot(kind='scatter', x='X', y='Y')
    plt.show(block=True)

#global variable for predicted values (the regression line/curve)
y_pred = []

#function to graph a linear regression
def linearRegression(xValue, yValue, b): 

    # plotting the actual points as scatter plot 
    plt.scatter(xValue, yValue) 

    global y_pred
    # predicted response vector 
    y_pred = b[0] * xValue + b[1]

    # plotting the regression line 
    plt.plot(xValue, y_pred, color = "r") 

    # putting labels 
    plt.xlabel('X') 
    plt.ylabel('Y') 

    # function to show plot 
    plt.show()

#function to graph a 2nd degree polynomial regression
def X2Regression(xValue, yValue, b):

    # plotting the actual points as scatter plot 
    plt.scatter(xValue, yValue) 

    x_line = np.linspace(xValue.min(), xValue.max(), 100)
    global y_pred
    y_pred = b[0] * np.square(x_line) + b[1] * x_line + b[2] 

    plt.plot(x_line, y_pred, color = "r") 

    y_pred = b[0] * np.square(xValue) + b[1] * xValue + b[2] 

    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y')

    # function to show plot 
    plt.show()

#function to graph a 3rd degree polynomial regression
def X3Regression(xValue, yValue, b):

    # plotting the actual points as scatter plot 
    plt.scatter(xValue, yValue) 

    x_line = np.linspace(xValue.min(), xValue.max(), 100)
    global y_pred
    y_pred = b[0] * np.power(x_line, 3) + b[1] * np.square(x_line) + b[2] * x_line + b[3] 

    plt.plot(x_line, y_pred, color = "r") 

    y_pred = b[0] * np.power(xValue, 3) + b[1] * np.square(xValue) + b[2] * xValue + b[3]

    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y')

    # function to show plot 
    plt.show()

#function to graph a 4th degree polynomial regression
def X4Regression(xValue, yValue, b):

    # plotting the actual points as scatter plot 
    plt.scatter(xValue, yValue) 

    x_line = np.linspace(xValue.min(), xValue.max(), 100)
    global y_pred
    y_pred = b[0] * np.power(x_line, 4) + b[1] * np.power(x_line, 3) + b[2] * np.square(x_line) + b[3] * x_line + b[4] 

    plt.plot(x_line, y_pred, color = "r") 

    y_pred = b[0] * np.power(xValue, 4) + b[1] * np.power(xValue, 3) + b[2] * np.square(xValue) + b[3] * xValue + b[4]

    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y')

    # function to show plot 
    plt.show()

#function to graph training and testing error as a function of lambda
def plotLambda(lambda_, averageTrainingErrorList, averageTestingErrorList):
    
    plt.scatter(lambda_, averageTrainingErrorList, label='training error')
    plt.scatter(lambda_, averageTestingErrorList, label='testing error')
    plt.xscale('log')
    plt.legend(loc='upper left')
    plt.xlabel('Lambda') 
    plt.ylabel('Error')
    plt.show()

#function to graph validation error as a function of lambda
def plotValidationError(lambda_, averageTrainingErrorList):

    plt.scatter(lambda_, averageTrainingErrorList)
    plt.xscale('log')
    plt.xlabel('Lambda') 
    plt.ylabel('Validation Error')
    plt.show()

#function to plot the different weights
def plotWeights(lambda_, weight1, weight2, weight3, weight4, weight5):

    plt.scatter(lambda_, weight1, label='weight 1')
    plt.scatter(lambda_, weight2, label='weight 2')
    plt.scatter(lambda_, weight3, label='weight 3')
    plt.scatter(lambda_, weight4, label='weight 4')
    plt.scatter(lambda_, weight5, label='weight 5')
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Lambda') 
    plt.ylabel('weights')
    plt.show()

#function to compute the weights using the equation from the lectures
def computeWeights(X, FeaturesList, OutputList, numberOfDataPoints, regularization, lambda_ = 0):

    if (regularization == True):
        # create identity matrix where bottom right element is 0
        identity_matrix = np.identity(5)
        identity_matrix[:1] = 0
        regularization = lambda_ * identity_matrix
    else:
        regularization = 0

    A = np.linalg.inv(np.dot(X.T, X) + regularization)
    B = np.dot(X.T, OutputList)
    weight = np.dot(A, B)

    return weight

#function to calculate the error of a given output data set
def calculateError(outputList, numberOfDataPoints):
    
    arrayOfErrors = np.square(y_pred - outputList) / numberOfDataPoints
    averageError = np.sum(arrayOfErrors)
    return averageError

#helper function used to split list of validation errors by their respective lambda
def chunkList(lst, n):
    return [ lst[i:i+n] for i in range(0, len(lst), n) ]

def main():

    #list the files
    trainingSet = glob.glob('*tr.dat')
    testingSet = glob.glob('*te.dat')

    #sort files (in the event that there are many testing sets)
    trainingSet.sort()
    testingSet.sort()

    
    #display the training and testing data
    trainingDataFrame = buildDataFrame(trainingSet)
    plotDataFrame(trainingDataFrame)

    testingDataFrame = buildDataFrame(testingSet)
    plotDataFrame(testingDataFrame)


    trainingFeaturesList = np.array(trainingDataFrame['X'].to_numpy())
    trainingOutputList = np.array(trainingDataFrame['Y'].to_numpy())

    testingFeaturesList = np.array(testingDataFrame['X'].to_numpy())
    testingOutputList = np.array(testingDataFrame['Y'].to_numpy())

    numberOfTrainingDataPoints = len(trainingFeaturesList)
    numberOfTestingDataPoints = len(testingFeaturesList)


    #add column vector of ones to features
    X = np.c_[trainingFeaturesList, np.ones(numberOfTrainingDataPoints)]

    #compute weights and plot linear regression for training and testing data sets
    weight = computeWeights(X, trainingFeaturesList, trainingOutputList, numberOfTrainingDataPoints, regularization = False)

    linearRegression(trainingFeaturesList, trainingOutputList, weight)

    averageTrainingError = calculateError(trainingOutputList, numberOfTrainingDataPoints)
    
    print("\nThe average training error for linear regression is: ", averageTrainingError)

    linearRegression(testingFeaturesList, testingOutputList, weight)

    averageTestingError = calculateError(testingOutputList, numberOfTestingDataPoints)

    print("\nThe average test error for linear regression is: ", averageTestingError)


    #compute weights and plot second degree polynomial regression for training and testing data sets
    X2 = np.c_[np.power(trainingFeaturesList, 2), trainingFeaturesList, np.ones(numberOfTrainingDataPoints)]

    weight = computeWeights(X2, trainingFeaturesList, trainingOutputList, numberOfTrainingDataPoints, regularization = False)

    X2Regression(trainingFeaturesList, trainingOutputList, weight)

    averageTrainingError = calculateError(trainingOutputList, numberOfTrainingDataPoints)

    print("\nThe average training error for second degree polynomial is: ", averageTrainingError)

    X2Regression(testingFeaturesList, testingOutputList, weight)

    averageTestingError = calculateError(testingOutputList, numberOfTestingDataPoints)

    print("\nThe average testing error for second degree polynomial is: ", averageTestingError)


    #compute weights and plot third degree polynomial regression for training and testing data sets
    X3 = np.c_[np.power(trainingFeaturesList, 3), np.power(trainingFeaturesList, 2), trainingFeaturesList, np.ones(numberOfTrainingDataPoints)]

    weight = computeWeights(X3, trainingFeaturesList, trainingOutputList, numberOfTrainingDataPoints, regularization = False)

    X3Regression(trainingFeaturesList, trainingOutputList, weight)

    averageTrainingError = calculateError(trainingOutputList, numberOfTrainingDataPoints)

    print("\nThe average training error for third degree polynomial is: ", averageTrainingError)

    X3Regression(testingFeaturesList, testingOutputList, weight)

    averageTestingError = calculateError(testingOutputList, numberOfTestingDataPoints)

    print("\nThe average testing error for third degree polynomial is: ", averageTestingError)



    #compute weights and plot fourth degree polynomial regression for training and testing data sets
    X4 = np.c_[np.power(trainingFeaturesList, 4), np.power(trainingFeaturesList, 3), np.power(trainingFeaturesList, 2), trainingFeaturesList, np.ones(numberOfTrainingDataPoints)]

    weight = computeWeights(X4, trainingFeaturesList, trainingOutputList, numberOfTrainingDataPoints, regularization = False)

    X4Regression(trainingFeaturesList, trainingOutputList, weight)

    averageTrainingError = calculateError(trainingOutputList, numberOfTrainingDataPoints)

    print("\nThe average training error for fourth degree polynomial is: ", averageTrainingError)

    X4Regression(testingFeaturesList, testingOutputList, weight)

    averageTestingError = calculateError(testingOutputList, numberOfTestingDataPoints)

    print("\nThe average testing error for fourth degree polynomial is: ", averageTestingError)


    print('\n------------------------------------------------------------------------------------------\n')

    #list of different lambdas to try as regularization terms
    lambda_ = [0.01, 0.1, 1, 10, 100, 1000, 10000]

    weight1 = []
    weight2 = []
    weight3 = []
    weight4 = []
    weight5 = []
    averageTrainingErrorList = []
    averageTestingErrorList = []
    
    for lam in lambda_:

        weight = computeWeights(X4, trainingFeaturesList, trainingOutputList, numberOfTrainingDataPoints, regularization = True, lambda_ = lam)
        
        weight1.append(weight[0])
        weight2.append(weight[1])
        weight3.append(weight[2])
        weight4.append(weight[3])
        weight5.append(weight[4])

        X4Regression(trainingFeaturesList, trainingOutputList, weight)

        averageTrainingErrorList.append(calculateError(trainingOutputList, numberOfTrainingDataPoints))

        X4Regression(testingFeaturesList, testingOutputList, weight)

        averageTestingErrorList.append(calculateError(testingOutputList, numberOfTestingDataPoints))


    #plot the average error as a function of lambda
    plotLambda(lambda_, averageTrainingErrorList, averageTestingErrorList)

    #plot each weight as a function of lambda
    plotWeights(lambda_, weight1, weight2, weight3, weight4, weight5)


    # prepare cross validation
    kfold = KFold(5, True, 1)

    validationError = []

    #loop through values of lambdas, each fold will have error corresponding to each lambda - add them all to a list
    for lam in lambda_:

        for trainingIndex, testingIndex in kfold.split(trainingFeaturesList):
            
            newX4DataSet = np.c_[[X4[i] for i in trainingIndex]]
            
            xTrain, xTest = trainingFeaturesList[trainingIndex], trainingFeaturesList[testingIndex]
            yTrain, yTest = trainingOutputList[trainingIndex], trainingOutputList[testingIndex]
        
            weight = computeWeights(newX4DataSet, xTrain, yTrain, len(yTrain), regularization = True, lambda_ = lam)

            X4Regression(xTrain, yTrain, weight)

            validationError.append(calculateError(yTrain, len(yTrain)))

    #use helper function to manipulate list into sublists sorted by lambda values
    averageValidationError = chunkList(validationError, 5)

    result = [] 
    for x in averageValidationError:
        result.append([sum(x)/len(x)])

    plotValidationError(lambda_, result)

    #use best lambda
    weight = computeWeights(X4, trainingFeaturesList, trainingOutputList, len(yTrain), regularization = True, lambda_ = 0.1)

    X4Regression(testingFeaturesList, testingOutputList, weight)

    
    
main()