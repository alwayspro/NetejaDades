import numpy as np
import pandas as pd

from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white") #fons millor blanc
sns.set(style="whitegrid", color_codes=True)


# data emmagatzemada al DataFrame

titanic_df = pd.read_csv("train.csv")
test_df    = pd.read_csv("test.csv")

# previsualització de les dades que tenim al dataset proporcionat
print("previsualització de les dades carregades:")
print(titanic_df.head(5))

# busquem valors nulls
titanic_df.isnull().sum()

print("numero de valors nulls al dataset:")
print(sum(pd.isnull(titanic_df['Age'])))


ax = titanic_df["Age"].hist(bins=15, alpha=0.8)
ax.set(xlabel='Edat', ylabel='Número')
plt.show()

print("mitja d'edat dels passatgers:")
print(titanic_df["Age"].median(skipna=True))


#ja que nomes tenim 2 passetgers perduts que han embarcat veiem que la majoria han embarcat:
sns.countplot(x='Embarked',data=titanic_df)
print("grafic que ens mostre la proporcio de passatgers embarcats distribuits per la seva zona d'embarc")
plt.show()

