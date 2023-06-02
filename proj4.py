
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


import pandas as pd

from deap import base
from deap import creator
from deap import tools

from random import randint, random

import argparse

from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

import random

import math
from sklearn import metrics

#Fucntion that initializes and mutates the classifiers parameters
def paramInit(*paramBounradies):
    if isinstance(paramBounradies[0], str):
        return random.choice(paramBounradies)
    else:
        if isinstance(paramBounradies[0], int):
            return randint(*paramBounradies)
        else:
            return random.uniform(*paramBounradies)

#Parcing the argumants of our script
# Expection -c and then one of ['SVC', 'KNN', 'RForest', 'Tree', 'MLP']
parser = argparse.ArgumentParser(description="OE arguments",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--classifier', choices=['SVC', 'KNN', 'RForest', 'Tree', 'MLP'], help='Classifier')

args = vars(parser.parse_args())

#A parameters mapping dict for every classifier, so that we have the same interface for every model
parameters = {
    'SVC': {
        'class': SVC,
        'params': {
            'C': {
                'value': None,
                'values': [0.0, 1.0],
                'method': paramInit
            },
            'kernel': {
                'value': None,
                'values': ['linear', 'poly', 'rbf', 'sigmoid'],
                'method': paramInit
            },
            'degree': {
                'value': None,
                'values': [0, 5],
                'method': paramInit
            },
            'gamma': {
                'value': None,
                'values': ['scale', 'auto'],
                'method': paramInit
            },
            'coef0': {
                'value': None,
                'values': [0.0, 10.0],
                'method': paramInit
            }
        },
    },
    'KNN': {
        'class': KNeighborsClassifier,
        'params': {
            'n_neighbors': {
                'value': None,
                'values': [3, 10],
                'method': paramInit
            },
            'weights': {
                'value': None,
                'values': ['uniform', 'distance'],
                'method': paramInit
            },
            'algorithm': {
                'value': None,
                'values': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'method': paramInit
            },
            'leaf_size': {
                'value': None,
                'values': [10, 40],
                'method': paramInit
            },
            'p': {
                'value': None,
                'values': [1, 2],
                'method': paramInit
            }
        },
    },
    'RForest': {
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': {
                'value': None,
                'values': [10, 100],
                'method': paramInit
            },
            'criterion': {
                'value': None,
                'values': ['gini', 'entropy', 'log_loss'],
                'method': paramInit
            },
            'max_depth': {
                'value': None,
                'values': [10, 50],
                'method': paramInit
            },
            'min_samples_split': {
                'value': None,
                'values': [2, 8],
                'method': paramInit
            },
            'max_features': {
                'value': None,
                'values': ['sqrt', 'log2', None],
                'method': paramInit
            }
        },
    },
    'Tree': {
        'class': DecisionTreeClassifier,
        'params': {
            'splitter': {
                'value': None,
                'values': ['best', 'random'],
                'method': paramInit
            },
            'criterion': {
                'value': None,
                'values': ['gini', 'entropy', 'log_loss'],
                'method': paramInit
            },
            'max_depth': {
                'value': None,
                'values': [10, 50],
                'method': paramInit
            },
            'min_samples_split': {
                'value': None,
                'values': [2, 8],
                'method': paramInit
            },
            'max_features': {
                'value': None,
                'values': ['sqrt', 'log2', None],
                'method': paramInit
            }
        },
    },
    'MLP': {
        'class': MLPClassifier,
        'params': {
            'hidden_layer_sizes': {
                'value': None,
                'values': [5, 30],
                'method': paramInit
            },
            'activation': {
                'value': None,
                'values': ['identity', 'logistic', 'tanh', 'relu'],
                'method': paramInit
            },
            'solver': {
                'value': None,
                'values': ['lbfgs', 'sgd', 'adam'],
                'method': paramInit
            },
            'alpha': {
                'value': None,
                'values': [0.00001, 0.001],
                'method': paramInit
            },
            'learning_rate': {
                'value': None,
                'values': ['constant', 'invscaling', 'adaptive'],
                'method': paramInit
            },
            'max_iter': {
                'value': None,
                'values': [300, 500],
                'method': paramInit
            }
        },
    },
}

#Extract the classifier by name from args
clsf = parameters[args['classifier']]['class']
#Extract the parameters dict for the chosen classifier
clsf_params = parameters[args['classifier']]

#Read the data file
#Detect the target column and remove it from the dataframe
#Converto target to numpy
df = pd.read_csv('./heart.csv')
y = df['target'].to_numpy()
df.drop('target', axis = 1, inplace=True)

#Identify how many features there are in dataframe
n_features = len(df.columns)
print(n_features)

#Function to represent individual creation
#Icls inherits the dict behaviour
def ClsfParams(n_features, icls):
    #Create genome that would be a dict in its essence
    genome = clsf_params

    #Initialize the parameters of the chosen classifier
    for key, val in genome['params'].items():
        genome['params'][key]['value'] = genome['params'][key]['method'](*genome['params'][key]['values'])

    #Create the features list
    genome['features'] = []

    #Determine which features to select from the DF
    for i in range(0,n_features):
        select = random.randint(0, 1)
        #Id select value is 1 - add the index of the column to the list.
        #These features will be kept in the dataset later during model evaluation
        if select == 1:
            genome['features'].append(i)
    genome['features'] = sorted(genome['features'])

    return icls(genome)

#Function to determine the fitness of the individual
def ClsfFitness(y, df, individual):
    split=5
    cv = model_selection.StratifiedKFold(n_splits=split)
    #Normalize
    mms = MinMaxScaler()
    df_norm = pd.DataFrame(mms.fit_transform(df))
    #Leave only selected features (by their id)
    df_norm = df_norm.iloc[:, individual['features']].to_numpy()
    #Create the classifier by unpacjing the parameters into the class init method
    estimator = clsf(**{key: individual['params'][key]['value'] for key in individual['params'].keys()})
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected,
        predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn) 
        resultSum = resultSum + result 
    return resultSum / split

#Function to mutate the individuals
def ClsfMutation(individual):
    param_to_mutate = random.randint(0, len(individual['params'].keys()))
    #mutate features as it is the last possible parameter
    #If the chosen feature is in the features list already - remove it from there
    #Otherwise - append it and sort the list
    if param_to_mutate == len(individual['params'].keys()):
        feature_to_mutate = random.randint(0, n_features - 1)
        if feature_to_mutate in individual['features']:
            individual['features'].remove(feature_to_mutate)
        else:
            individual['features'].append(feature_to_mutate)
        individual['features'] = sorted(individual['features'])
        return

    #Mutate the chosen parameters of the classifier. Do it by reinitializing it
    key = list(individual['params'].keys())[param_to_mutate]
    #Mutate the parameter with the corresponding key for the index
    individual['params'][key]['value'] = individual['params'][key]['method'](*individual['params'][key]['values'])

#Crossover sunction
def ClsfMate(ind1, ind2):
    #Determine the crossover point. Valid for both parameters and feature list
    point = random.randint(0, len(ind1['params'].keys()) - 1)

    #Create 2 kids
    kid1 = {
        'params': {},
        'features': {}
    }

    kid2 = {
        'params': {},
        'features': {}
    }

    #mate with parameters
    #Iterate over the parameters of the parents
    #The first kid gets first part of the parameters of the first parent (up to a key wth determined id) and second part of the second parent
    # Second kid gets the same, but from reversed parents 
    for idx, key in enumerate(ind1['params'].keys()):
        if idx <= point:
            kid1['params'][key] = dict(ind1['params'][key])
            kid2['params'][key] = dict(ind2['params'][key])
        else:
            kid1['params'][key] = dict(ind2['params'][key])
            kid2['params'][key] = dict(ind1['params'][key])

    #mate with features
    #Similar logic for the features
    kid1['features'] = [i for i in ind1['features'] if i <= point] + [i for i in ind2['features'] if i > point]
    kid2['features'] = [i for i in ind2['features'] if i <= point] + [i for i in ind1['features'] if i > point]

    return (kid1, kid2)

import multiprocessing

# maksymalizacja
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", dict, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

if __name__ == "__main__":

    # generowanie nowych osobników
    toolbox.register('individual', ClsfParams, n_features, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # wskazanie funkcji celu
    toolbox.register("evaluate", ClsfFitness, y, df)
    toolbox.register("mutate", ClsfMutation)

    toolbox.register('select', tools.selBest)
    toolbox.register('mate', ClsfMate)


    sizePopulation = 50
    probabilityMutation = 0.2
    probabilityCrossover = 0.8
    numberIteration = 30

    # wygenerowanie początkowej populacji, obliczenie funkcji dopasowania
    pop = toolbox.population(n=sizePopulation)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = [fit]

    g = 0
    numberElitism = 1

    history = {
        "std": [],
        "mean": [],
        "best": [],
    }

    while g < numberIteration:
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        listElitism = []
        for x in range(0, numberElitism):
            listElitism.append(tools.selBest(pop, 1)[0])

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < probabilityCrossover:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < probabilityMutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = [fit]

        print("  Evaluated %i individuals" % len(invalid_ind))
        pop[:] = offspring + listElitism

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        best_ind = tools.selBest(pop, 1)[0]
        ind_repr = {key: best_ind['params'][key]['value'] for key in best_ind['params'].keys()}
        print(f"Best individual is {ind_repr}, {best_ind['features']} , {best_ind.fitness.values}")
        history['std'].append(std)
        history['mean'].append(mean)
        history['best'].append(best_ind.fitness.values)
    #
    print("-- End of (successful) evolution --")


#Remains : opis danych, krótki opis modeli (tabela z nazwami i wabranymi dla nich parametrami), wykresy 
# (zmienny history zawiera wszystko co jest potrzebne), analiza czasowa (uruchomić i zobaczyć który jest szybszy), 
# wyniki (max accuracy dla kazdego modelu i jakie są najlepsze osobniki), wnioski (że który lepiej i dlaczego. Accuracy prawie wszędzie jest taka sama, 
# więc tutaj raczej czasy są ważniejsze)

# Uruchomienie: pythin proj4.py -c i jedno z ['SVC', 'KNN', 'RForest', 'Tree', 'MLP']

