import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

st.title("visualisation et prédiction des types de reponse")
left_column,middle,right_column = st.columns(3)
left_column.button('Visualisation')
middle.write('Dataset mathE')

df=pd.read_csv("/kaggle/input/mathe-data-assesment/MathE.csv",sep=";", encoding='cp1252')
#suppression de la colonne Keywords
df = df.drop('Keywords', axis=1)
df = df.drop('Student ID', axis=1)
df = df.drop('Question ID', axis=1)
#prétraitement des données
#ignorer les notifiation d'erreurs
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#transformation en tableau de valeurs
data=df.values
#frequency encoding sur l'attribut student country
#prétraitement des données
#application du frequency encoding of student country

freq_country=df['Student Country'].value_counts()
df['Student Country']=df['Student Country'].map(freq_country/df.shape[0])
#label encoding de l'attribut Question Level
for level in range(len(df['Question Level'])):
    match df['Question Level'][level]:
        case "Basic":
            df['Question Level'][level]=0
        case "Advanced":
            df['Question Level'][level]=1

#suppression de la colonne keywords
#df=df.drop("Keywords", axis=1)
#frequency encoding sur l'attribut 
freq_topic=df['Topic'].value_counts()
df['Topic']=df['Topic'].map(freq_topic/df.shape[0])

#frequency encoding de l'attribut Subtopic
freq_subtopic=df['Subtopic'].value_counts()
df['Subtopic']=df['Subtopic'].map(freq_subtopic/df.shape[0])

#séparation des données
X = df.drop('Type of Answer', axis=1)
y = df['Type of Answer']

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Normalisation des caractéristiques
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Création du modèle KNN
knn = KNeighborsClassifier(n_neighbors=6)
# Entraîner le modèle
knn.fit(X_train, y_train)
# Prédire les classes de l'ensemble de test
y_pred = knn.predict(X_test)


