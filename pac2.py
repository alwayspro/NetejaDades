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

# proporcio de "edat" que s'ha perdut
print("proporcio d'Edat respecte al total de passatges perduts:")
print(round(177/(len(titanic_df["PassengerId"])),4))

ax = titanic_df["Age"].hist(bins=15, alpha=0.8)
ax.set(xlabel='Edat', ylabel='Número')
plt.show()

print("mitja d'edat dels passatgers:")
print(titanic_df["Age"].median(skipna=True))

# ratio/ proporció de passetgers perduts respecte total
round(687/len(titanic_df["PassengerId"]),4)

# ratio anterior pero tinguent en compte els que han embarcat
round(2/len(titanic_df["PassengerId"]),4)

#ja que nomes tenim 2 passetgers perduts que han embarcat veiem que la majoria han embarcat:
sns.countplot(x='Embarked',data=titanic_df)
print("grafic que ens mostre la proporcio de passatgers embarcats distribuits per la seva zona d'embarc")
plt.show()


#ara anem a fer ajustos a les dades, tal i com hem vist a teoria intentant imputar valors a on no n'hi hagin
train_data = titanic_df
#si l'edat es null, hi posarem 28 que es la mitja
train_data["Age"].fillna(28, inplace=True)
#si no tenim Embarked, hi posarem S que es la mes comuna
train_data["Embarked"].fillna("S", inplace=True)
#ignorarem la variable 'cabin' ja que tenim masses valors null per poder ho millorar amb metodes de dataQuality
train_data.drop('Cabin', axis=1, inplace=True)


#tot seguit, crearem el que hem vist a teoria que s'anomenen Flags o variables catagoriques, per tant de tenir amb un 1 o un 0 si pertanyen en aquell subconjunt:

train_data['TravelBuds']=train_data["SibSp"]+train_data["Parch"]
train_data['TravelAlone']=np.where(train_data['TravelBuds']>0, 0, 1)
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)
train_data.drop('TravelBuds', axis=1, inplace=True)
#Pclass serà la variable categorica de classe
train2 = pd.get_dummies(train_data, columns=["Pclass"])

train3 = pd.get_dummies(train2, columns=["Embarked"])

train4=pd.get_dummies(train3, columns=["Sex"])
train4.drop('Sex_female', axis=1, inplace=True)

train4.drop('PassengerId', axis=1, inplace=True)
train4.drop('Name', axis=1, inplace=True)
train4.drop('Ticket', axis=1, inplace=True)
train4.head(5)

print("dataset amb les noves variables:")
print (train4.head(5))
df_final = train4



plt.figure(figsize=(15,8))
sns.kdeplot(titanic_df["Age"][df_final.Survived == 1], shade=True)
sns.kdeplot(titanic_df["Age"][df_final.Survived == 0], shade=True)
plt.legend(['Sobreviuen', 'Morts'])
plt.title('Repartiment d edat segons han sobreviscut o no')
plt.show()

#distribució  de sobrevivents respecte l'edad:
plt.figure(figsize=(20,8))
avg_survival_byage = df_final[["Age", "Survived"]].groupby(['Age'], as_index=False).mean()
g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage)


#distribució  de sobrevivents respecte la tarifa del seu tiquet:
plt.figure(figsize=(15,8))
sns.kdeplot(df_final["Fare"][titanic_df.Survived == 1], shade=True)
sns.kdeplot(df_final["Fare"][titanic_df.Survived == 0], shade=True)
plt.legend(['Sobreviuen', 'Morts'])
plt.title('Repartiment dels supervivents segons la tarifa')
# limitem el eix per tal de centrar-lo.
plt.xlim(-20,200)
plt.show()

#distribució segons el port on han embarcat
sns.barplot('Embarked', 'Survived', data=titanic_df, color="teal")
plt.show()

#distribució segons el port on han embarcat
sns.barplot('Pclass', 'Survived', data=titanic_df, color="darkturquoise")
plt.show()