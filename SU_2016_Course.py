import csv
import random
from sklearn import naive_bayes
from sklearn.ensemble.forest import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.linear_model.ridge import Ridge
import sklearn.metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.regression import KNeighborsRegressor
from sklearn import cross_validation
from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from pprint import pprint

# Set the seed
random.seed(42)

do_validate = True
do_predict = False

class Dataset(object):
    def __init__(self, filename, columns):
        self.filename = filename
        self.columns = columns
        self.data = None
    
    def load(self, collectors):
        with open(self.filename, "r") as fd:
            csv_reader = csv.DictReader(fd, fieldnames=self.columns, delimiter=";")
            _header = next(csv_reader)
            raw_data = []
            for row in csv_reader:
                raw_data.append({key: collectors[key].collect(value) if key in collectors else value for key, value in row.items()})
            
            self.data = []
            for row in raw_data:
                self.data.append({key: collectors[key].transform(value) if key in collectors else value for key, value in row.items()})

            return self.data


def load_dataset(file, columns, value_filter=lambda x: x):
    with open(file, "r") as fd:
        csv_reader = csv.DictReader(fd, fieldnames=columns, delimiter=";")
        _header = next(csv_reader)
        print("Loading dataset ...")
        return [{key: value_filter(value) for key, value in row.items()} for row in csv_reader]


def filter_dataset(dataset, features, transform=lambda x: x):
    return [[transform(row[feature]) for feature in features] for row in dataset]


def cleanup_data(value):
    return value if value not in ["NR", ""] else -1


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
                       columns=columns,
                       value_filter=cleanup_data)

print("Filtering features ...")
# ["crm", "annee_naissance", "kmage_annuel"]
feature_to_filter = [
    # 'id',
    # 'annee_naissance',
    # 'annee_permis',
    # 'marque',
    'puis_fiscale',
    'anc_veh',
    # 'codepostal',
    # 'energie_veh',
    'kmage_annuel',
    'crm',
    # 'profession',
    # 'var1',
    # 'var2',
    'var3',
    'var4',
    'var5',
    # 'var6',
    'var7',
    # 'var8',
    'var9',
    # 'var10',
    'var11',
    'var12',
    'var13',
    # 'var14',
    'var15',
    # 'var16',
    'var17',
    'var18',
    'var19',
    'var20',
    'var21',
    'var22',
    # 'prime_tot_ttc'
]
#feature_to_filter = [
#    feature
#    for feature in columns
#    if feature not in feature_list
#]
print("feature_to_filter = {}".format(feature_to_filter))
dataset_filtered = filter_dataset(dataset=dataset,
                                  features=feature_to_filter,
                                  transform=float)

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

# Build the model
# model = ExtraTreesRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
# model = GaussianNB()
# model = Ridge()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
model_classes = [
    ('ExtraTreesRegressor', lambda: ExtraTreesRegressor(n_estimators=100)),
    ('RandomForestRegressor', lambda: RandomForestRegressor(n_estimators=100)),
    # GradientBoostingRegressor,
    # GaussianNB,
    # Ridge,
    # KNeighborsRegressor,
    ('DecisionTreeRegressor', lambda: DecisionTreeRegressor())
]

for name, model_factory in model_classes:
    print("Testing {}".format(name))

    model = model_factory()
    model.fit(train_dataset, train_target)

    ## Cross Validation ###

    #cv = StratifiedKFold(train_dataset, n_folds=5)

    ###scoring
    scores = cross_validation.cross_val_score(model, train_dataset, train_target, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #Evaluate the quality of the prediction
    test_predictions = model.predict(test_dataset)
    quality = sklearn.metrics.mean_absolute_error(test_predictions, test_target)
    print("Errors: %0.2f" % quality)

if do_predict:
    print("Predicting ...")
    test_dataset = load_dataset(file="data/ech_test.csv",
                                columns=columns,
                                value_filter=cleanup_data)

    test_dataset_filtered = filter_dataset(dataset=test_dataset,
                                           features=feature_to_filter,
                                           transform=float)

    with open("result.csv", mode="w") as outfile:
        csv_writer = csv.writer(outfile, delimiter=";")
        result = model.predict(test_dataset_filtered)
        csv_writer.writerow(("id", "prime_tot_ttc"))
        for res, row in zip(result, test_dataset):
            csv_writer.writerow((row['id'], res))

print("Done!")
