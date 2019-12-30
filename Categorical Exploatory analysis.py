import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

titanic = pd.read_csv('Exercise Files/titanic.csv')
cont_var = ["PassengerId", "Age", "Parch", "SibSp", "Pclass", "Name", "Fare"]
titanic.drop(cont_var, axis=1, inplace=True)
titanic.head()

# explor high level
titanic.info(0)

#Missing values for cabin
titanic.groupby(titanic['Cabin'].isnull()).mean()

# insert indicator
titanic['Cab_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1)
titanic.head(10)


# Veiw plots for catigorical variables
for i, col in enumerate(['Embarked', 'Sex', 'Cab_ind']):
    plt.figure(i)
    sns.catplot(x=col, y="Survived", data=titanic, kind='point', aspect=2)
    plt.title('Probability of survival for {}'.format(col))

# Look at pivot table for sex compared to Embarked
titanic.pivot_table('Survived', index="Sex", columns= 'Embarked', aggfunc='count')

# look at pivot table for Cabin comared to Emabarked
titanic.pivot_table('Survived', index="Cab_ind", columns= 'Embarked', aggfunc='count')

#bring the titanic data set back in.

titanic = pd.read_csv('Exercise Files/titanic.csv')

#Drop unneeded variables.
titanic.drop(["Ticket", "Name"], axis = 1 , inplace = True)

#Add cabin ind variable.
titanic['Cab_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1)
titanic.head()


# Creat indicator for gender.
gen_num = {"male": 0 , "female": 1}

titanic["Sex"] = titanic['Sex'].map(gen_num)
titanic.head()

# drop unneeded columns.
titanic.drop(["Cabin", "Embarked"], axis= 1, inplace= True)
