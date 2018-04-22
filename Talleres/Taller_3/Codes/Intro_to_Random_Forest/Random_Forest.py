# Import the Pandas library
import pandas as pd
# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"

test = pd.read_csv(test_url)

##Desicion Tree##

# Import the Numpy library
import numpy as np
# Import 'tree' from scikit-learn library
from sklearn import tree

# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

#Fill the missing data from train set
train.Fare[152] = train["Fare"].median()

#Print the Sex and Embarked columns
#print(train["Embarked"])
#print(train["Sex"])

# Print the train data to see the available features
#print(train)

#Fill the missing data of the train set
train["Age"] = train["Age"].fillna(train["Age"].median())

# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))

# Impute the missing value with the median
test.Fare[152] = test["Fare"].median()

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

# Impute the Embarked variable
test["Embarked"] = test["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

test["Age"] = test["Age"].fillna(test["Age"].median())

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

#print(test_features)

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)
print(my_prediction)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])

# Create a new array with the added features: features_two
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)

#Print the score of the new decison tree
print(my_tree_two.score(features_two,target))

# Extract the features from the test set: "Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"
test_features_two = test[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

# Make your prediction using the test set
my_prediction_two = my_tree_two.predict(test_features_two)
print(my_prediction_two)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution_two = pd.DataFrame(my_prediction_two, PassengerId, columns = ["Survived"])
print(my_solution_two)

# Check that your data frame has 418 entries
print(my_solution_two.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution_two.to_csv("my_solution_two.csv", index_label = ["PassengerId"])
# Create train_two with the newly defined feature

train_two = train.copy()
train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] + 1  

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values

# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three,target)

# Print the score of this decision tree
print(my_tree_three.score(features_three, target))

#Create test_two with the newly defined feature
test_two = test.copy()
test_two["family_size"] = test_two["SibSp"] + test_two["Parch"] + 1  

# Extract the features from the test set: "Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"
test_features_three = test_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values

# Make your prediction using the test set
my_prediction_three = my_tree_three.predict(test_features_three)
print(my_prediction_three)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution_three = pd.DataFrame(my_prediction_three, PassengerId, columns = ["Survived"])
print(my_solution_three)

# Check that your data frame has 418 entries
print(my_solution_three.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution_three.to_csv("my_solution_three.csv", index_label = ["PassengerId"])


# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=5, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest,target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))

my_solution_forest = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])
print(my_solution_forest)

# Check that your data frame has 418 entries
print(my_solution_forest.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution_forest.to_csv("my_solution_forest_1.csv", index_label = ["PassengerId"])