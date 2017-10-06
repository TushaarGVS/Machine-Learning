import random
from random import randrange
from csv import reader
from math import sqrt, exp, pi

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
        element[-1] = 1.0 if element[-1] == 'Yes' else 0.0

'''
def split_dataset(dataset, split):
    train_size = int(len(dataset) * split)
    train_set = []
    test_set = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))
    return [train_set, test_set]
'''

def split_dataset_k_folds(dataset, k_folds):
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

def cross_validate(dataset, k_folds):
    split_dataset = split_dataset_k_folds(dataset, k_folds)
    accuracy_values = list()
    for chunk in split_dataset:
        train_set = list(split_dataset)
        train_set.remove(chunk)
        train_set = sum(train_set, [])
        test_set = list()
        for X in chunk:
            X_copy = list(X)
            test_set.append(X_copy)
        accuracy_value = one_fold_naive_bayes(train_set, test_set)
        accuracy_values.append(accuracy_value)
    return mean(accuracy_values)

def class_label_split(dataset):
    split_set = {}
    for i in range(len(dataset)):
        data_point = dataset[i]
        if data_point[-1] not in split_set:
            split_set[data_point[-1]] = []
        split_set[data_point[-1]].append(data_point)
    return split_set

def mean(data_points):
    return sum(data_points)/ float(len(data_points))

def standard_deviation(data_points):
    if len(data_points) > 1:
        mean_val = mean(data_points)
        variance = 0.0
        for x in data_points:
            variance += ((x - mean_val) ** 2)
        variance = variance/ float(len(data_points) - 1)
    else:
        variance = 0
    return sqrt(variance)

def calculate_mean_standard_deviation(dataset):
    attributes = zip(*dataset)
    prior_calculations = [(mean(x), standard_deviation(x)) for x in attributes]
    del prior_calculations[-1]
    return prior_calculations

def prior_probability(dataset):
    split_set = class_label_split(dataset)
    prior_calculations = {}
    for class_label, data_points in split_set.iteritems():
        prior_calculations[class_label] = calculate_mean_standard_deviation(data_points)
    return prior_calculations

def gaussian_probability(Xi, mean_val, standard_deviation):
    if standard_deviation > 0:
        val = -1.0 * ((Xi - mean_val) ** 2)/ (2 * (standard_deviation ** 2))
        constant = 1.0 / sqrt(2 * pi)
        return (constant / standard_deviation) * exp(val)
    return 1

def class_wise_probability(prior_calculations, data_point):
    probabilities = {}
    for class_label, attribute_values in prior_calculations.iteritems():
        probabilities[class_label] = 1
        for i in range(len(attribute_values)):
            mean_val, standard_deviation = attribute_values[i]
            Xi = data_point[i]
            probabilities[class_label] *= gaussian_probability(Xi, mean_val, standard_deviation)
    return probabilities

def assign_class_label(prior_calculations, data_point):
    probabilities = class_wise_probability(prior_calculations, data_point)
    max_probability = -1
    class_assigned = None
    for class_label, probability in probabilities.iteritems():
        if probability > max_probability:
            max_probability = probability
            class_assigned = class_label
    return class_assigned

def make_predictions(prior_calculations, test_set):
    predicted = []
    for i in range(len(test_set)):
        class_assigned = assign_class_label(prior_calculations, test_set[i])
        predicted.append(class_assigned)
    return predicted

def accuracy(predicted, test_set):
    correct_predictions = 0
    for i in range(len(test_set)):
        if predicted[i] == test_set[i][-1]:
            correct_predictions += 1
    return (correct_predictions/ float(len(predicted))) * 100

def one_fold_naive_bayes(training_set, test_set):
    prior_calculations = prior_probability(training_set)
    predicted = make_predictions(prior_calculations, test_set)
    accuracy_value = accuracy(predicted, test_set)
    return accuracy_value

def naive_bayes(dataset, k_folds):
    accuracy = cross_validate(dataset, k_folds)
    return accuracy

'''
filename = raw_input("Enter the file name: ")
# split = float(raw_input("Enter the split ratio: "))
k_folds = int(raw_input("Enter the number of folds: "))

dataset = load_csv(filename)
# training_set, test_set = split_dataset(dataset, split)
# accuracy_value = one_fold_naive_bayes(training_set, test_set)
accuracy_value = naive_bayes(dataset, k_folds)
print("Accuracy: %s" % accuracy_value)
'''