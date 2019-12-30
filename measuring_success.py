import pandas as pd
from sklearn.model_selection import train_test_split

#import data set
titanic = pd.read_csv('Exercise Files/titanic.csv')
titanic.head()

#divide data set by features and labels
features = titanic.drop('Survived', axis= 1)
labels = titanic['Survived']

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size= .4, random_state= 42)
X_test, X_val, Y_test, Y_val = train_test_split(x_test, y_test, test_size= .5, random_state= 42)

print(len(x_train)/len(titanic), len(y_train)/len(titanic), len(X_val)/len(titanic), len(Y_val)/len(titanic))

