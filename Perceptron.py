# Perceptron model for IRIS.csv - Linear dataset: Gives accuracy of 100%
# Change 'IRIS-Setosa' to 'Yes'; Line 25; to use it on SPECTF_New.csv - Non Linear dataset: Gives varied accuracy

from csv import reader
from random import randrange, seed

def load_csv(filename):
    dataset = list()
    check = 0
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            if check != 0:
                dataset.append(row)
            check += 1
    change_dataset(dataset)
    return dataset

def change_dataset(dataset):
    for element in dataset:
        for i in range(len(element) - 1):
            element[i] = float(element[i].strip())
        element[-1] = 1.0 if element[-1] == 'Iris-setosa' else 0.0

def make_predictions(X, weights):
    activation = weights[0]
    for i in range(len(weights) - 1):
        activation += weights[i + 1] * X[i]
    return 1.0 if activation >= 0.0 else 0.0

def compute_weights(train_set, learning_rate, epochs):
    weights = [0.0 for i in range(len(train_set[0]))]
    for i in range(epochs):
        for X in train_set:
            predicted_value = make_predictions(X, weights)
            error = X[-1] - predicted_value
            weights[0] += learning_rate * error
            for j in range(len(weights) - 1):
                weights[j + 1] += learning_rate * error * X[j]
    return weights

def compute_accuracy(actual, predicted):
    correct_prediction = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct_prediction += 1
    return float(correct_prediction) / len(actual) * 100

def split_into_k_folds(dataset, k_folds):
    dataset_copy = list(dataset)
    split_dataset = list()
    fold_size = int(len(dataset) / k_folds)
    for i in range(k_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        split_dataset.append(fold)
    return split_dataset

def perceptron(train_set, test_set, learning_rate, epochs):
    weights = compute_weights(train_set, learning_rate, epochs)
    predicted = list()
    for X in test_set:
        predicted.append(make_predictions(X, weights))
    return predicted

def cross_validate(dataset, k_folds, algorithm, *args):
    split_dataset = split_into_k_folds(dataset, k_folds)
    accuracy_values = list()
    for chunk in split_dataset:
        train_set = list(split_dataset)
        train_set.remove(chunk)
        train_set = sum(train_set, [])
        test_set = list()
        for X in chunk:
            X_copy = list(X)
            X_copy[-1] = None
            test_set.append(X_copy)
        actual = [X[-1] for X in chunk]
        predicted = algorithm(train_set, test_set, *args)
        accuracy_values.append(compute_accuracy(actual, predicted))
    return accuracy_values

seed(1)

filename = raw_input("Enter the file name: ")
learning_rate = raw_input("Enter the learning rate: ")
epochs = raw_input("Enter the number of epochs: ")
k_folds = raw_input("Enter the number of folds: ")

dataset = load_csv(filename)
accuracy_values = cross_validate(dataset, int(k_folds), perceptron, float(learning_rate), int(epochs))
print("Accuracy: %s" % (accuracy_values))
