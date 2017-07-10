# Logistic regression model on 'pima-indians-diabetes.csv'

from math import sqrt, exp
from random import seed
from random import randrange
from csv import reader
from matplotlib import pyplot

# Load a CSV file
def load_csv(filename):
    dataset = list() ;
    with open(filename, 'r') as file:
        csv_reader = reader(file) ;
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row) ;
        return dataset ;

# Convert string column to float
def str_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip()) ;

# Find min-max values for each column
def min_max(dataset):
    minmax = list() ;
    for i in range(len(dataset[0])):
        column = [row[i] for row in dataset] ;
        min_val = min(column) ;
        max_val = max(column) ;
        minmax.append([min_val, max_val]) ;
    return minmax ;

# Normalization of the dataset
def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0])/(minmax[i][1] - minmax[i][0]) ;

# Split dataset into k-folds
def dataset_k_fold_split(dataset, k):
    dataset_split = list() ;
    dataset_copy = list(dataset) ;
    fold_size = int(len(dataset)/k) ;
    for i in range(k):
        fold = list() ;
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy)) ;
            fold.append(dataset_copy.pop(index)) ;
        dataset_split.append(fold) ;
    return dataset_split ;

# Simple logistic regression with coefficients
def make_predictions(row, coefficients):
    y = coefficients[0] ;
    for i in range(len(row)-1):
        y += coefficients[i + 1] * row[i] ;
    return 1.0/(1.0 + exp(-y)) ;

# Calculate the coefficients using scholastic gradient descent
def coefficients(train, learning_rate, epochs):
    coeff = [0.0 for row in range(len(train[0]))] ;
    for epoch in range(epochs):
        sse = 0 ;
        for row in train:
            y = make_predictions(row, coeff) ;
            error = row[-1] - y ;
            sse += error**2 ;
            coeff[0] = coeff[0] + learning_rate * error * y * (1 - y) ;
            for i in range(len(row)-1):
                coeff[i + 1] = coeff[i + 1] + learning_rate * error * y * (1 - y) * row[i] ;
    return coeff ;

# Simple logistic regression algorithm
def logistic_regression(train, test, learning_rate, epochs):
    predictions = list() ;
    coeff = coefficients(train, learning_rate, epochs) ;
    for row in test:
        y = round(make_predictions(row, coeff)) ;
        predictions.append(y) ;
    return predictions ;

# Calculate the accuracy of prediction
def accuracy_calc(actual, predicted):
    correct_predicts = 0 ;
    for i in range(len(actual)):
        if(actual[i] == predicted[i]):
            correct_predicts += 1 ;
    return correct_predicts/float(len(actual)) * 100 ;

# Evaluate regression algorithm based on k-fold dataset
def evaluate_k_fold_split_algorithm(dataset, algorithm, k, *args):
    folds = dataset_k_fold_split(dataset, k) ;
    outcome = list() ;
    for fold in folds:
        train = list(folds) ;
        train.remove(fold) ;
        train = sum(train, []) ;
        test = list() ;
        test_x = list() ;
        for row in fold:
            row_copy = list(row) ;
            test_x.append(row_copy[0]) ;
            test.append(row_copy) ;
            row_copy[-1] = None ;
        predicted = algorithm(train, test, *args) ;
        actual = [row[-1] for row in fold] ;
        pyplot.title('Logistic Regression') ;
        pyplot.xlabel('Input') ;
        pyplot.ylabel('Predictions(blue)/Actual(green)') ;
        pyplot.scatter(test_x, predicted, color='blue') ;
        pyplot.scatter(test_x, actual, color='green') ;
        pyplot.show() ;
        accuracy = accuracy_calc(actual, predicted) ;
        outcome.append(accuracy) ;
    return outcome ;
    
# Load and prepare the datset
seed(1) ;
filename = raw_input('Filename: ') ;
dataset = load_csv(filename) ;
for i in range(len(dataset[0])):
    str_float(dataset, i) ;

# Normalize the dataset
normalize(dataset, min_max(dataset)) ;

# Test the algorithm with k-folds
k = int(raw_input('K-Folds: ')) ;
learning_rate = float(raw_input('Learning Rate: ')) ;
epochs = int(raw_input('Epochs: ')) ;
outcome = evaluate_k_fold_split_algorithm(dataset, logistic_regression, k, learning_rate, epochs) ;
print('Outcome: %r' % (outcome)) ;
print('Average accuracy: %f' % (sum(outcome)/float(len(outcome)))) ;
