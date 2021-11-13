from random import seed #seed method used random number generator
from random import randrange
from csv import reader
# from matplotlib import pyplot as plt
 
# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
 
# Convert string column to float
def convert_inputs_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
 
# Convert string column to integer
def convert_desired_outputs_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup
 
# Test the algorithm using a cross validation split, calculate the accuracy
def test(dataset, algorithm, number_of_folds, *args):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / number_of_folds)
    for i in range(number_of_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    folds = dataset_split
    scores = list()
    for fold in folds:
        training_set = list(folds)
        training_set.remove(fold)
        training_set = sum(training_set, [])
        testing_set = list()
        for row in fold:
            row_copy = list(row)
            testing_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(training_set, testing_set, *args)
        actual = [row[-1] for row in fold]
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        accuracy = correct / float(len(actual)) * 100.0
        scores.append(accuracy)
    return scores
 
# Make a prediction with synaptic_weights
def predict(row, synaptic_weights):
    induced_local_field = synaptic_weights[0]
    for i in range(len(row)-1):
        induced_local_field += synaptic_weights[i + 1] * row[i]
    return 1.0 if induced_local_field >= 0.0 else 0.0
 
# Perceptron Algorithm With Stochastic Gradient Descent
def train(train, test, learning_rate_parameter, number_of_epochs):
    predictions = list()
    synaptic_weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(number_of_epochs):
        for row in train:
            prediction = predict(row, synaptic_weights)
            error = row[-1] - prediction
            synaptic_weights[0] = synaptic_weights[0] + learning_rate_parameter * error
            for i in range(len(row)-1):
                synaptic_weights[i + 1] = synaptic_weights[i + 1] + learning_rate_parameter * error * row[i]
    for row in test:
        prediction = predict(row, synaptic_weights)
        predictions.append(prediction)
    return(predictions)
 
# Test the Perceptron algorithm on the sonar dataset
seed(1)
# load and prepare data
filename = r'C:\Users\megha\Documents\Neural Networks\sonar_all-data.csv'
dataset = load_csv(filename) #call load_csv method
#print(dataset)
for i in range(len(dataset[0])-1):
    convert_inputs_to_float(dataset, i) #go through each item in list
#print(dataset) #dataset is now a list of floats (not strings)
# convert string class to integers
convert_desired_outputs_to_int(dataset, len(dataset[0])-1)
#key R corresponds to value 0, key M corresponds to value 1
# evaluate algorithm
number_of_folds = 2 #number of iterations
learning_rate_parameter = float(input("Enter a value between 0 and 1 for the learning rater parameter: ")) #.2 and .8 both lead to 75% accuracy
number_of_epochs = int(input("Enter number of epochs: ")) #1000 yields 75% accuracy
scores = test(dataset, train, number_of_folds, learning_rate_parameter, number_of_epochs)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))