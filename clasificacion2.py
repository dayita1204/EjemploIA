import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#Importamos datos en un Dataframe
df_data = pd.read_excel(r'C:\Users\user\Downloads\Ejercicio+Codificación+CLASIFICACION\Ingresos_por_persona.xlsx')
print(df_data)
#df_data = df_data[0:1000] #Selecciona solo 1000 datos de entrada
#df_data.info()


#Convertir Variables categóricas de entrada (X)
df_X = df_data.drop('Ingresos', axis=1) #Eliminar la columna ingresos

df_X = pd.get_dummies(df_X) # Raza y sexo son variables categoricas
# Es necesario codificar las variables categoricas en numericas 

#print(df_X)
#df_X.info()

#Crear un array (dataframe) con las variables de entrada (X) y otro para la variable de salida (y)
X = df_X.values
y = df_data['Ingresos'].values


#Dividir datos en conjunto de "Training" (ej: 80%) y conjunto de "Test" (ej:20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y) #stratify --> datos_etiquetados

# Construir Modelos en base a los diferentes algoritmos
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# Evaluar cada modelo
results = []
names = []
for name, model in models:
 kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=42)
 cv_results = model_selection.cross_val_score(model, X, y, cv=kfold)
 results.append(cv_results)
 names.append(name)
 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 print(msg)

 # Seleccionar mejor modelo tras benchmarking
 svc = SVC()

 # Optimizar y entrenar el modelo
 import numpy as np
from sklearn.model_selection import GridSearchCV

Cs = [0.1, 1] # Menores combinaciones para mayor rapidez [0.001, 0.01, 0.1, 1, 10]
gammas = [0.1, 1] # Menores combinaciones para mayor rapidez [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
svc_cv = GridSearchCV(svc, param_grid, cv=5)
svc_cv.fit(X, y)
svc_cv.best_params_
svc_cv.best_score_

# Predecir Resultados de salida (y_prediction) a partir de nuevos datos de entrada (X_new)
df_new = pd.read_excel(r'C:\Users\user\Downloads\Ejercicio+Codificación+CLASIFICACION\Ingresos_nuevos_datos.xlsx')
df_X_new = pd.get_dummies(df_new)
X_new = df_X_new.values

X_new

y_prediction = svc_cv.predict(X_new)
print("Prediccion: {}".format(y_prediction))

