import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

st.title("Visualisation et prédiction des iris")
left_column,middle,right_column = st.columns(3)
left_column.button('Visualisation')
middle.write('Dataset iris')




#Charger le fichier iris.csv dans un DataFrame
df = pd.read_csv('iris.csv', delimiter=";")

# Séparer les caractéristiques et la cible
X = df.drop('Species', axis=1)
y = df['Species']

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
                                            
# Normaliser les caractéristiques
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Créer le modèle KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Entraîner le modèle
knn.fit(X_train, y_train)


# prediction sur les données de test
y_pred = knn.predict(X_test)

#prediction sur un nouvel iris
    
right_column.button('Prediction', type="secondary")
L_S = st.slider("Longueur sépal", -10.000000000000, 10.000000000000)
l_S = st.slider("Largeur sépal", -10.000000000000, 10.000000000000)
L_P = st.slider("Longueur pétal", -10.000000000000, 10.000000000000)
l_P = st.slider("Largeur pétal", -10.000000000000, 10.000000000000)

if st.button('Prédire', type="primary"): 
    pred=knn.predict([[L_S,l_S,L_P,l_P]])
    st.write("Votre IRIS est un", pred[0])
    


  
