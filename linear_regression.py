# Linear regression model on 'insurance.csv'

from math import sqrt
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

# Split dataset into training and test
def dataset_split(dataset, split):
    train = list() ;
    train_size = split * len(dataset) ;
    dataset_copy = list(dataset) ;
    while len(train) < train_size:
        index = randrange(len(dataset_copy)) ;
        train.append(dataset_copy.pop(index)) ;
    return train, dataset_copy ;

# Calculate RMSE
def rmse_calc(actual, predicted):
    sse = 0.0 ;
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i] ;
        sse += (prediction_error ** 2) ;
    rmse = sqrt(sse/float(len(actual))) ;
    return rmse ;

# Calculate mean of a list of numbers
def mean(values):
    return sum(values)/float(len(values)) ;

# Calculate variance of a list of numbers
def variance(values, mean):
    return sum([(i - mean)**2 for i in values]) ;

# Calculate co-variance between x, y
def covariance(X, mean_x, Y, mean_y):
    covar = 0.0 ;
    for i in range(len(X)):
        covar += (X[i] - mean_x) * (Y[i] - mean_y) ;
    return covar ;

# Calculate the coefficients for linear regression
def linear_coefficients(dataset):
    X = [row[0] for row in dataset]
    Y = [row[1] for row in dataset]
    mean_x, mean_y = mean(X), mean(Y) ;
    b1 = covariance(X, mean_x, Y, mean_y)/variance(X, mean_x) ;
    b0 = mean_y - b1 * mean_x ;
    return [b0, b1] ;

# Simple linear regression algorithm
def linear_regression(train, test):
    predictions = list() ;
    b0, b1 = linear_coefficients(train) ;
    for row in test:
        y = b0 + b1 * row[0] ;
        predictions.append(y) ;
    return predictions ;

# Evaluate regression algorithm on training set
def evaluate_algorithm(dataset, algorithm):
    test_set = list() ;
    for row in dataset:
        row_copy = list(row) ;
        row_copy[-1] = None ;
        test_set.append(row_copy) ;
    predicted = algorithm(dataset, test_set) ;
    print(predicted) ;
    actual = [row[-1] for row in dataset] ;
    rmse = rmse_calc(actual, predicted) ;
    return rmse ;

# Evaluate regression algorithm based on training and test set
def evaluate_split_algorithm(dataset, algorithm, split, *args):
    train, test = dataset_split(dataset, split) ;
    test_set = list() ;
    test_x = list() ;
    for row in test:
        row_copy = list(row) ;
        test_x.append(row_copy[0]) ;
        row_copy[-1] = None ;
        test_set.append(row_copy) ;
    predicted = algorithm(train, test_set, *args) ;
    # print(test_set) ;
    # print('Predicted: %r' % (predicted)) ;
    actual = [row[-1] for row in test] ;
    # print('Actual: %r' % (actual)) ;
    pyplot.title('Linear Regression') ;
    pyplot.xlabel('Input') ;
    pyplot.ylabel('Predictions(blue)/Actual(green)') ;
    pyplot.plot(test_x, predicted, 'b') ;
    pyplot.scatter(test_x, actual, color='green') ;
    pyplot.show() ;
    rmse = rmse_calc(actual, predicted) ;
    return rmse ;

# Load and prepare the datset
seed(1) ;
filename = raw_input('Filename: ') ;
dataset = load_csv(filename) ;
for i in range(len(dataset[0])):
               str_float(dataset, i) ;

# Test the algorithm
split = float(raw_input('Split: ')) ;
rmse = evaluate_split_algorithm(dataset, linear_regression, split) ;
print('RMSE: %f' % (rmse)) ;
