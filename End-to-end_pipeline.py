import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Bing in data set
titanic = pd.read_csv('Exercise Files/titanic.csv')

#look as nulls
titanic.isnull().sum()

# Fill NA's for Age
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)

# Combine Sibsp and parch
for i, col in enumerate(['SibSp', 'Parch']):
    plt.figure(i)
    sns.catplot(x=col, y= 'Survived', data=titanic, kind= 'point', aspect=2)

titanic['family_cnt'] = titanic['SibSp']+ titanic['Parch']


#drop unnessiisary variables
titanic.drop(['SibSp', 'Parch', 'PassengerId'], axis =1, inplace=True)
titanic.head()

#LOAD cleaned data
titanic.to_csv('Exercise files/titanic_cleaned.csv', index=False)


#EDIT CATIGORICAL DATA

# Cabin indicaator
titanic['Cab_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1)


#Sex indicator
gen_num = {'male': 0, 'female' : 1}
titanic['Sex'] = titanic['Sex'].map(gen_num)
titanic['Sex'].head()


# Drop unneed cat variables.
titanic.drop(['Cabin', 'Embarked', 'Name', 'Ticket'], axis= 1, inplace=True)
titanic.head()

titanic.to_csv('Exercise files/titanic_cleaned.csv', index=False)

# split data
from sklearn.model_selection import train_test_split

features = titanic.drop('Survived', axis=1)
labels = titanic['Survived']

x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state= 42, test_size= .4 )

X_test, X_val, Y_test, Y_val = train_test_split(x_test, y_test, random_state= 42, test_size= .5)


for dataset in [Y_test, Y_val, x_train]:
    print(round(len(dataset) / len(labels), 2))


# Save data sets.
x_train.to_csv('Exercise Files/x_train.csv', index= False)
X_val.to_csv('Exercise Files/x_val.csv', index= False)
X_test.to_csv('Exercise Files/x_test.csv', index= False)

y_train.to_csv('Exercise Files/y_train.csv', index= False)
Y_test.to_csv('Exercise Files/y_test.csv', index= False)
Y_val.to_csv('Exercise Files/Y_val.csv', index= False)


# Begin Random forest.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


rf = RandomForestClassifier()

scores = cross_val_score(rf, x_train, y_train.values.ravel(), cv=5)
scores


# using grid search.
from sklearn.model_selection import GridSearchCV

# print results function
 def print_results(results):
        print('BEST PARAMS: {}\n'.format(results.best_params_))

        means = results.cv_results_['mean_test_score']
        stds = results.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, results.cv_results_['params']):
            print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

rf = RandomForestClassifier()

paramerters = {
    'n_estimators': [5, 50, 100],
    'max_depth': [2, 10, 20, None]
}

cv = GridSearchCV(rf, paramerters, cv= 5)
cv.fit(x_train, y_train.values.ravel())

#print results
print_results(cv)


## Test on the validation set.
from sklearn.metrics import accuracy_score, precision_score, recall_score


#pick the best predictors.
rf1 = RandomForestClassifier(n_estimators= 5, max_depth= 10)
rf1.fit(x_train, y_train.values.ravel())

rf2 = RandomForestClassifier(n_estimators= 100, max_depth= 100)
rf2.fit(x_train, y_train.values.ravel())

rf3 = RandomForestClassifier(n_estimators=100, max_depth=None)
rf3.fit(x_train, y_train.values.ravel())


for mdl in [rf1, rf2, rf3]:
    Y_pred = mdl.predict(X_val)
    accuracy = round(accuracy_score(Y_val, Y_pred), 3)
    precision = round(precision_score(Y_val, Y_pred), 3)
    recall = round(recall_score(Y_val, Y_pred), 3)
    print('max Depth{} / Number of est {} / -- A: {} / P: {} / R: {}'.format(mdl.max_depth, mdl.n_estimators, accuracy, precision, recall))


#test model on test set.
y2_pred = rf3.predict(X_test)

accuracy2 = round(accuracy_score(Y_test, y2_pred), 3)
precision2 = round(precision_score(Y_test, y2_pred), 3)
recall2 = round(recall_score(Y_test, y2_pred), 3)

print('P: {} / A: {} / Recall: {}'.format(precision2, accuracy2, recall2))









