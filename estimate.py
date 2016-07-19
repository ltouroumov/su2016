import csv
import random
from sklearn import naive_bayes
from sklearn.ensemble.forest import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.linear_model.ridge import Ridge
import sklearn.metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.regression import KNeighborsRegressor
from sklearn import cross_validation
from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from pprint import pprint
import xgboost as xgb

def load_dataset(file):
    with open(file, "r") as fd:
        csv_reader = csv.reader(fd, delimiter=";")
        header = next(csv_reader)
        print("Loading dataset ...")
        return header, [[float(value) for value in row] for row in csv_reader]

do_validation = False

header, dataset = load_dataset(file='data/ech_apprentissage_clean.csv')

target_idx = header.index('prime_tot_ttc')
targets = [row[target_idx] for row in dataset]
print("Splitting dataset")

non_features = [
    header.index('id'),
    header.index('prime_tot_ttc')
]

# remove id column from the dataset we feed to the algo
dataset_filtered = [[val for idx, val in enumerate(row) if idx not in non_features] for row in dataset]

dataset_size = len(dataset_filtered)
train_dataset_size = int(dataset_size * 0.7)
test_dataset_size = dataset_size - train_dataset_size

train_dataset = dataset_filtered[:train_dataset_size]
test_dataset = dataset_filtered[train_dataset_size:]

train_target = targets[:train_dataset_size]
test_target = targets[train_dataset_size:]

print("train_dataset_size = {}\ntest_dataset_size = {}".format(len(train_dataset), len(test_dataset)))

model = xgb.XGBRegressor(max_depth=8, n_estimators=400, silent=False)
if do_validation:
    print("Fitting model")
    model.fit(train_dataset, train_target)

    ## Cross Validation ###
    print("Validating")
    ###scoring
    scores = cross_validation.cross_val_score(model, train_dataset, train_target, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #Evaluate the quality of the prediction
    test_predictions = model.predict(test_dataset)
    quality = sklearn.metrics.mean_absolute_error(test_predictions, test_target)
    print("Errors: %0.2f" % quality)
else:
    print("Fitting model")
    model.fit(dataset, targets)

    print("Predicting ...")
    test_header, test_dataset = load_dataset(file="data/ech_test_clean.csv")

    test_dataset_filtered = [[val for idx, val in enumerate(row) if idx not in non_features] for row in test_dataset]

    with open("result.csv", mode="w") as outfile:
        csv_writer = csv.writer(outfile, delimiter=";")
        result = model.predict(test_dataset_filtered)
        csv_writer.writerow(("id", "prime_tot_ttc"))
        for res, row in zip(result, test_dataset):
            csv_writer.writerow((int(row[0]), res))

print("Done!")