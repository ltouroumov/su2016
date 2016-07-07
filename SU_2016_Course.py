# coding: utf-8

# In[47]:

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

# Set the seed
random.seed(42)

# In[28]:

# Open the file
filepath_train = "./data/ech_apprentissage.csv"
fieldnames = "id;annee_naissance;annee_permis;marque;puis_fiscale;anc_veh;codepostal;energie_veh;kmage_annuel;crm;profession;var1;var2;var3;var4;var5;var6;var7;var8;var9;var10;var11;var12;var13;var14;var15;var16;var17;var18;var19;var20;var21;var22;prime_tot_ttc".split(";")
with open(filepath_train, "r") as file_open:
    # Open the csv reader over the file
    csv_reader = csv.reader(file_open, delimiter=";")
    # Read the first line which is the header
    header = next(csv_reader)
    # Load the dataset contained in the file
    # dataset = []
    # for row in csv_reader:
    #     dataset.append(row)
    dataset = list(map(lambda row: [value if value not in ["NR", ""] else -1 for value in row], csv_reader))

# In[38]:

# Replace the missing values
# for index, row in enumerate(dataset):
#     dataset[index] = [value if value not in ["NR", ""] else -1 for value in row]

# In[39]:

# Filter the dataset based on the column name
feature_to_filter = ["crm", "annee_naissance", "kmage_annuel"]
indexes_to_filter = [header.index(feature) for feature in feature_to_filter]
# for feature in feature_to_filter:
#    indexes_to_filter.append(header.index(feature))

dataset_filtered = [[float(row[index]) for index in indexes_to_filter] for row in dataset]
# for row in dataset:
#     dataset_filtered.append([float(row[index]) for index in indexes_to_filter])

# Build the structure containing the target
targets = [float(row[header.index("prime_tot_ttc")]) for row in dataset]
# for row in dataset:
#    targets.append(float(row[header.index("prime_tot_ttc")]))

# In[40]:

# Split the datasets to have one for learning and the other for the test
train_dataset = []
test_dataset = []
train_target = []
test_target = []

for row, target in zip(dataset_filtered, targets):
    if random.random() < 0.70:
        train_dataset.append(row)
        train_target.append(target)
    else:
        test_dataset.append(row)
        test_target.append(target)

# In[41]:

# Build the model
# model=ExtraTreesRegressor()
# model=RandomForestRegressor()
# model=GradientBoostingRegressor()
# model=GaussianNB()
model = Ridge()
# model=KNeighborsRegressor()
# model=DecisionTreeRegressor()
model.fit(train_dataset, train_target)

# Predict with the model
predictions = model.predict(test_dataset)

# In[51]:

### Cross Validation ###

# cv = StratifiedKFold(train_dataset, n_folds=5)

###scoring
scores = cross_validation.cross_val_score(model, train_dataset, train_target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

### getting the predictions ###
# predicted = cross_validation.cross_val_predict(clf, train_dataset, train_target, cv=10)
# print metrics.accuracy_score(train_target, predicted)
model.fit(train_dataset, train_target)
predictions = model.predict(test_dataset)

# In[50]:

# Evaluate the quality of the prediction
print(sklearn.metrics.mean_absolute_error(predictions, test_target))

# Alternative -- Compute the mean absolute percentage error


# In[ ]:

# Now load the test file and use the model built to score the dataset and create the submission file.
