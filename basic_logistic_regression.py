# Logistic regression model on a random dataset

from math import sqrt, exp

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
        print('Epoch: %d, Learning Rate: %f, SSE: %f' % (epoch, learning_rate, sse)) ;
    return coeff ;

# Test the algorithm
dataset = [[2.7810836,2.550537003,0],
           [1.465489372,2.362125076,0],
           [3.396561688,4.400293529,0],
           [1.38807019,1.850220317,0],
           [3.06407232,3.005305973,0],
           [7.627531214,2.759262235,1],
           [5.332441248,2.088626775,1],
           [6.922596716,1.77106367,1],
           [8.675418651,-0.242068655,1],
           [7.673756466,3.508563011,1]] ;
# coeff = [-0.406605464, 0.852573316, -1.104746259] ;
learning_rate = 0.3 ; epoch = 100 ;
coeff = coefficients(dataset, learning_rate, epoch) ;
print('Coefficients: %r' % (coeff)) ;
for row in dataset:
    y = make_predictions(row, coeff) ;
    print('Expected: %f, Predicted: %f [%d]' % (row[-1], y, round(y))) ;

# Learning Rate:
# Used to limit the amount each coefficient is corrected each time it is updated.
# Epochs:
# The number of times to run through the training data while updating the coefficients.
