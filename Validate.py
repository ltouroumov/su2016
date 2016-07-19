import csv
import random
from collections import defaultdict
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

def load_dataset(file, columns, value_filter=lambda x: x):
    with open(file, "r") as fd:
        csv_reader = csv.DictReader(fd, fieldnames=columns, delimiter=";")
        _header = next(csv_reader)
        print("Loading dataset ...")
        return list(csv_reader)

def scan_dataset(dataset, features):
    for row in dataset:
        for feature in features:
            feature.collect(row[feature.name], row)

def process_dataset(dataset, features):
    return [[feature.transform(row[feature.name]) for feature in features] for row in dataset]

class PassCollector(object):
    def collect(self, *args):
        pass
    
    def transform(self, value):
        return value    

class NumberCollector(object):
    def __init__(self, ctype=float):
        self.ctype = ctype

    def collect(self, *args):
        pass
    
    def transform(self, value):
        return self.ctype(value) if value not in ["NR", ""] else self.ctype(-1)

class AgeCollector(object):
    def __init__(self, year):
        self.year = year

    def collect(self, *args):
        pass
    
    def transform(self, value):
        return (self.year - int(value)) if value not in ["NR", ""] else -1

class GroupAverage(object):
    class Group(object):
        def __init__(self):
            self.count = 0
            self.sum = 0
        
        def collect(self, value):
            self.count += 1
            self.sum += value
        
        def value(self):
            return self.sum / self.count if self.count > 0 else -1
    
    def __init__(self, vlambda):
        self.vlambda = vlambda
        self.values = defaultdict(GroupAverage.Group)
    
    def collect(self, value, row):
        self.values[value].collect(self.vlambda(row))
    
    def transform(self, value):
        return self.values[value].value()

class GroupCount(object):
    def __init__(self):
        self.values = defaultdict(lambda: 0)
    
    def collect(self, value, row):
        self.values[value] += 1
    
    def transform(self, value):
        return self.values[value]

class OrdinalCollector(object):
    def __init__(self, values, nan):
        self.values = values
        self.nan = nan
    
    def collect(self, *args):
        pass
    
    def transform(self, value):
        if value == self.nan:
            return -1
        else:
            return self.values.index(value)

class Feature(object):
    def __init__(self, name, collector):
        self.name = name
        self.collector = collector
    
    def __repr__(self):
        return self.name

    def collect(self, value, row):
        self.collector.collect(value, row)
    
    def transform(self, value):
        return self.collector.transform(value)

# Open the file
train_data_file = "./data/ech_apprentissage.csv"
columns = [
    'id',
    'annee_naissance',
    'annee_permis',
    'marque',
    'puis_fiscale',
    'anc_veh',
    'codepostal',
    'energie_veh',
    'kmage_annuel',
    'crm',
    'profession',
    'var1',
    'var2',
    'var3',
    'var4',
    'var5',
    'var6',
    'var7',
    'var8',
    'var9',
    'var10',
    'var11',
    'var12',
    'var13',
    'var14',
    'var15',
    'var16',
    'var17',
    'var18',
    'var19',
    'var20',
    'var21',
    'var22',
    'prime_tot_ttc'
]
dataset = load_dataset(file=train_data_file,
                       columns=columns)


print("Filtering features ...")
features = [
    # 'id',
    Feature('annee_naissance', AgeCollector(2016)),
    Feature('annee_permis', AgeCollector(2016)),
    Feature('marque', GroupAverage(lambda row: float(row['prime_tot_ttc']) / float(row['crm']))),
    Feature('puis_fiscale', NumberCollector()),
    Feature('anc_veh', NumberCollector()),
    # 'codepostal',
    # 'energie_veh',
    Feature('kmage_annuel', NumberCollector()),
    Feature('crm', NumberCollector()),
    # 'profession',
    Feature('var1', NumberCollector()),
    Feature('var2', NumberCollector()),
    Feature('var3', NumberCollector()),
    Feature('var4', NumberCollector()),
    Feature('var5', NumberCollector()),
    Feature('var6', OrdinalCollector(('A', 'B', 'C', 'D'), 'N')),
    Feature('var7', NumberCollector()),
    Feature('var8', GroupCount()),
    Feature('var9', NumberCollector()),
    Feature('var10', NumberCollector()),
    Feature('var11', NumberCollector()),
    Feature('var12', NumberCollector()),
    Feature('var13', NumberCollector()),
    Feature('var14', OrdinalCollector(('G', 'F', 'E', 'D', 'C', 'B', 'A'), 'N')),
    Feature('var15', NumberCollector()),
    Feature('var16', NumberCollector()),
    Feature('var17', NumberCollector()),
    Feature('var18', NumberCollector()),
    Feature('var19', NumberCollector()),
    Feature('var20', NumberCollector()),
    Feature('var21', NumberCollector()),
    Feature('var22', NumberCollector()),
    # 'prime_tot_ttc'
]

print("feature_to_filter = {}".format(features))

scan_dataset(dataset, features)
dataset_filtered = process_dataset(dataset, features)

pprint(dataset_filtered[:2])

targets = [float(row["prime_tot_ttc"]) for row in dataset]
print("Splitting dataset")

dataset_size = len(dataset_filtered)
train_dataset_size = int(dataset_size * 0.7)
test_dataset_size = dataset_size - train_dataset_size

train_dataset = dataset_filtered[:train_dataset_size]
test_dataset = dataset_filtered[train_dataset_size:]

train_target = targets[:train_dataset_size]
test_target = targets[train_dataset_size:]

print("train_dataset_size = {}\ntest_dataset_size = {}".format(len(train_dataset), len(test_dataset)))

model = xgb.XGBRegressor(max_depth=8, n_estimators=400, silent=False)
print("Fitting model")
model.fit(train_dataset, train_target)

print("Validating")
###scoring
scores = cross_validation.cross_val_score(model, train_dataset, train_target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#Evaluate the quality of the prediction
test_predictions = model.predict(test_dataset)
quality = sklearn.metrics.mean_absolute_error(test_predictions, test_target)
print("Errors: %0.2f" % quality)

print("Predicting ...")
test_dataset = load_dataset(file="data/ech_test.csv", columns=columns)

test_dataset_filtered = process_dataset(test_dataset, features)

with open("result.csv", mode="w") as outfile:
    csv_writer = csv.writer(outfile, delimiter=";")
    result = model.predict(test_dataset_filtered)
    csv_writer.writerow(("id", "prime_tot_ttc"))
    for res, row in zip(result, test_dataset):
        csv_writer.writerow((row['id'], res))

print("Done!")
