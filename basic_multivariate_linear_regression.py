# Multivariate linear regression on random dataset

from math import sqrt, exp

# Multivariate linear regression using coeffients
def make_predictions(row, coefficients):
    y = coefficients[0] ;
    for i in range(len(row)-1):
        y += coefficients[i + 1] * row[i] ;
    return y ;

# Estimating coefficients using scholastic gradient
def coefficients(train, learning_rate, epochs):
    coeff = [0.0 for i in range(len(train[0]))] ;
    for epoch in range(epochs):
        sse = 0 ;
        for row in train:
            y = make_predictions(row, coeff) ;
            error = y - row[-1] ;
            sse += error**2 ;
            coeff[0] = coeff[0] - learning_rate  * error ;
            for i in range(len(row)-1):
                coeff[i + 1] = coeff[i + 1] - learning_rate * error * row[i] ;
            print('Epoch: %d, Learning Rate: %f, SSE: %f' % (epoch, learning_rate, sse)) ;
    return coeff ;

# Test the algorithm
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]] ;
learning_rate = 0.001 ;
epochs = 50 ;
coeff = coefficients(dataset, learning_rate, epochs) ;
print(coeff) ;
