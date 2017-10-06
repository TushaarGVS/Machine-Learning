import sys, csv
import Naive_Bayes as NB
import Genetic_Algorithm as GA
from time import time
import operator

def count_attributes(filename):
    f = open(filename, 'r')
    reader = csv.reader(f, delimiter=',')
    num_attributes = len(next(reader)) - 1
    return num_attributes

def feature_selection(population_size, chromosome_length, encoding, cross_over_rate, mutation_rate, num_iterations, dataset, k_folds):
    start = time()
    chromosomes = GA.genetic_algorithm(population_size, chromosome_length, encoding, cross_over_rate, mutation_rate, num_iterations, dataset, k_folds)
    fitness_values = GA.fitness_evaluation(chromosomes, dataset, k_folds)
    end = time()
    time_taken = end - start
    return [chromosomes, fitness_values, time_taken]

def selected_attributes(filename, best_chromosome):
    attributes = []
    best_attributes = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            attributes = row
            break
    for i in range(len(best_chromosome)):
        if best_chromosome[i] == 1:
            best_attributes.append(attributes[i])
    return best_attributes

filename = raw_input("Enter the file name: ")
# split = float(raw_input("Enter the split ratio: "))
k_folds = int(raw_input("Enter the number of folds: "))
population_size = int(raw_input("Enter the population size: "))
chromosome_length = count_attributes(filename)
encoding = 1
cross_over_rate = int(raw_input("Enter the cross-over rate (%): "))
mutation_rate = int(raw_input("Enter the mutation rate (%): "))
num_iterations = int(raw_input("Enter the number of iterations: "))

dataset = NB.load_csv(filename)
chromosomes, fitness_values, time_taken = feature_selection(population_size, chromosome_length, encoding, cross_over_rate, mutation_rate, num_iterations, dataset, k_folds)

print("---------------------------------------------------")
print("Time taken for Feature Selection: %s" % time_taken)
print("Final Chromosomes: %s" % chromosomes)
print("Final Fitness Values: %s" % fitness_values)
print("---------------------------------------------------\n")

index, max_fitness_value = max(enumerate(fitness_values), key = operator.itemgetter(1))
best_chromosome = chromosomes[index]
best_attributes = selected_attributes(filename, best_chromosome)
print("Selected features: %s" % best_attributes)
print("Accuracy with selected features: %s" % max_fitness_value)