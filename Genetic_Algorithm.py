from random import randint, random, choice
from time import time
import Naive_Bayes as NB

def delete_column(dataset, attribute):
    attributes = zip(*dataset)
    del(attributes[attribute])
    return zip(*attributes)

def format_dataset(dataset, chromosome):
    num_deletions = 0
    for attribute in range(len(chromosome)):
        if chromosome[attribute] == 0:
            index = attribute - num_deletions
            dataset = delete_column(dataset, index)
            num_deletions += 1
    return dataset

def initialize_population(population_size, chromosome_length, encoding):
    chromosomes = []
    for i in range(population_size):
        chromosome = []
        for j in range(chromosome_length):
            chromosome_val = randint(0, encoding)
            chromosome.append(chromosome_val)
        chromosomes.append(chromosome)
    print("\nInitialize Population: %s" %chromosomes)
    return chromosomes

def fitness_evaluation(chromosomes, dataset, k_folds):
    fitness_values = []
    for chromosome in chromosomes:
        '''
        val = chromosome[0] + 2*chromosome[1] + 3*chromosome[2] + 4*chromosome[3] - 30
        if val == -1:
            val = -1.0
        else:
            val = 1.0/(1+val)
        '''
        formatted_dataset = format_dataset(dataset, chromosome)
        '''
        training_set, test_set = NB.split_dataset(formatted_dataset, split)
        val = NB.naive_bayes(training_set, test_set)
        '''
        val = NB.naive_bayes(formatted_dataset, k_folds)
        fitness_values.append(val)
    print("Fitness Values: %s\n" %fitness_values)
    return fitness_values

def roulette_wheel_selection(chromosomes, fitness_values):
    probability_values = []
    cumulative_probabilities = []
    random_numbers = []
    assigned_chromosomes = []
    for i in range(len(fitness_values)):
        probability_values.append(1.0 * fitness_values[i] / sum(fitness_values))
        cumulative_probabilities.append(sum(probability_values))
        random_numbers.append(random())
    for i in range(len(random_numbers)):
        for j in range(len(cumulative_probabilities)):
            if cumulative_probabilities[j] >= random_numbers[i]:
                assigned_chromosomes.append(j)
                break
    chromosomes_copy = list(chromosomes)
    chromosomes = []
    for i in range(len(assigned_chromosomes)):
        chromosomes.append(chromosomes_copy[assigned_chromosomes[i]])
    print("Roulette Wheel Selection: %s" %chromosomes)
    return chromosomes

def cross_over(chromosomes, population_size, chromosome_length, cross_over_rate):
    num_chromosomes = int(cross_over_rate * population_size / 100)
    cross_over_chromosomes = []
    index = randint(0, population_size-1)
    while len(cross_over_chromosomes) < num_chromosomes:
        if index not in cross_over_chromosomes:
            cross_over_chromosomes.append(index)
        index = randint(0, population_size-1)
    chromosomes_copy = list(chromosomes)
    for i in range(len(cross_over_chromosomes)):
        cross_over_point = randint(0, chromosome_length-2)
        first_chromosome = [chromosomes_copy[cross_over_chromosomes[i]][j] for j in range(cross_over_point + 1)]
        second_chromosome = [chromosomes_copy[cross_over_chromosomes[(i+1) % num_chromosomes]][j] for j in range(cross_over_point + 1, chromosome_length)]
        chromosomes[cross_over_chromosomes[i]] = first_chromosome + second_chromosome
    print("Cross Over: %s" %chromosomes)
    return chromosomes

def mutation(chromosomes, population_size, chromosome_length, mutation_rate, encoding):
    num_mutation_bits = int(population_size * chromosome_length * mutation_rate / 100)
    mutation_chromosomes = []
    mutation_bits = []
    index = randint(0, population_size-1)
    bit = randint(0, chromosome_length-1)
    while len(mutation_bits) < num_mutation_bits:
        if index not in mutation_chromosomes or bit not in mutation_bits:
            mutation_chromosomes.append(index)
            mutation_bits.append(bit)
        index = randint(0, population_size - 1)
        bit = randint(0, chromosome_length - 1)
    for i in range(len(mutation_chromosomes)):
        val = chromosomes[mutation_chromosomes[i]][mutation_bits[i]]
        '''
        allowed_values = range(0, val) + range(val + 1, encoding)
        chromosomes[mutation_chromosomes[i]][mutation_bits[i]] = choice(allowed_values)
        '''
        chromosomes[mutation_chromosomes[i]][mutation_bits[i]] = 1 - val
    print("Mutation: %s" %chromosomes)
    return chromosomes

def check_equal(fitness_values):
    return all(value == fitness_values[0] for value in fitness_values)

def genetic_algorithm(population_size, chromosome_length, encoding, cross_over_rate, mutation_rate, num_iterations, dataset, k_folds):
    chromosomes = initialize_population(population_size, chromosome_length, encoding)
    fitness_values = []
    while num_iterations > 0 or not check_equal(fitness_values):
        fitness_values = fitness_evaluation(chromosomes, dataset, k_folds)
        chromosomes = roulette_wheel_selection(chromosomes, fitness_values)
        chromosomes = cross_over(chromosomes, population_size, chromosome_length, cross_over_rate)
        chromosomes = mutation(chromosomes, population_size, chromosome_length, mutation_rate, encoding)
        num_iterations -= 1
    return chromosomes

'''
population_size = int(raw_input("Enter the population size: "))
chromosome_length = int(raw_input("Enter the chromosome length: "))
encoding = int(raw_input("Enter the maximum encoding value: "))
cross_over_rate = int(raw_input("Enter the cross-over rate (%): "))
mutation_rate = int(raw_input("Enter the mutation rate (%): "))
num_iterations = int(raw_input("Enter the number of iterations: "))

start = time()
chromosomes = genetic_algorithm(population_size, chromosome_length, encoding, cross_over_rate, mutation_rate, num_iterations)
fitness_values = fitness_evaluation(chromosomes)
end = time()

print("---------------------------------------------------")
print("Time taken by Genetic Algorithm: %s" %(end - start))
print("Final Chromosomes: %s" %chromosomes)
print("Final Fitness Values: %s" %fitness_values)
print("---------------------------------------------------\n")
'''