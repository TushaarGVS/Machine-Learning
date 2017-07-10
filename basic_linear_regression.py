# Linear regression model on a random dataset

from math import sqrt

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

# Calculate the coefficients
def coefficients(dataset):
    X = [row[0] for row in dataset]
    Y = [row[1] for row in dataset]
    mean_x, mean_y = mean(X), mean(Y) ;
    b1 = covariance(X, mean_x, Y, mean_y)/variance(X, mean_x) ;
    b0 = mean_y - b1 * mean_x ;
    return [b0, b1] ;

# Simple linear regression algorithm
def linear_regression(train, test):
    predictions = list() ;
    b0, b1 = coefficients(train) ;
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

# Test the algorithm
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]] ;
rmse = evaluate_algorithm(dataset, linear_regression) ;
print('RMSE: %f' % (rmse)) ;
