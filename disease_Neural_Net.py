import argparse
import json
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from pprint import pprint

def main():
	# set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="dsppd_xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="dsppd_yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="dsppd_xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="dsppd_yTest.csv",
                        help="filename for labels associated with the test data")
    parser.add_argument("bestParamOutput",
                         help="json filename for best parameter")
    parser.add_argument("-r", "--readParams",
                        action='store_true',
                        help="Use parameters stored in bestParamOutput")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()

    print("Tuning neural networks --------")
    nnName = "NN"
    nnGrid = {
            "hidden_layer_sizes": [(35, 30, 25), (25,), (25, 10), (25, 25), (10, 10, 10), (25, 10, 10), (50), (50, 10)],
            "alpha": [1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-3, 1e-2],
            "activation": ["relu", "tanh"],
        }
    nnClf = MLPClassifier(max_iter=10000)

    if args.readParams:
        with open(args.bestParamOutput, 'r') as f:
            nnGrid = json.load(f)
            for key, val in nnGrid.items():
                if not isinstance(val, list): nnGrid[key] = [val]

    resultDict, bestParams = eval_gridsearch(nnClf, nnGrid,
                                                   xTrain, yTrain, xTest, yTest)
    
    if not args.readParams:
        # store the best parameters
        with open(args.bestParamOutput, 'w') as f:
            json.dump(bestParams, f)

    #pprint(resultDict)
    print(f"Accuracy: {resultDict["Accuracy"]}")


def eval_gridsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):

    grid = GridSearchCV(
        estimator=clf,
        param_grid=pgrid,
        scoring="accuracy",
        cv=KFold(n_splits=5, shuffle=True),
        n_jobs = -1,
        refit=True #refits the final model using full train set
        );

    start = time.time()

    grid.fit(xTrain, yTrain)

    best_model = grid.best_estimator_

    yHat = best_model.predict(xTest)
    acc = accuracy_score(yTest, yHat)

    disease_labels = best_model.classes_
    f1_per_disease = f1_score(
        yTest,
        yHat,
        labels=disease_labels,
        average=None
    )

    timeElapsed = time.time() - start

    resultDict = {"Accuracy": float(acc), "Time": float(timeElapsed)}

    for label, score in zip(disease_labels, f1_per_disease):
        resultDict[label] = score

    print(classification_report(yTest, yHat, labels=disease_labels))

    return resultDict, grid.best_params_


if __name__ == "__main__":
    main()

