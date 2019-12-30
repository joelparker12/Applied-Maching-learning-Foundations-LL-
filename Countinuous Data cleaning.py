import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#bring in data
titanic = pd.read_csv('Exercise Files/titanic.csv')

# Drop unneeded varibales
titanic.drop("PassengerId", axis=1, inplace = True)

#fill in missing for age
titanic.groupby(titanic['Age'].isnull()).mean()
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)
titanic.isnull().sum()

titanic['Age'].head(10)

# make family count
titanic['family_cnt'] = titanic["SibSp"] + titanic['Parch']
titanic.drop(['SibSp', 'Parch'], axis=1, inplace=True)

titanic.head(5)



