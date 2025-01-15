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
df_data = df_data[0:1000] #Selecciona solo 1000 datos de entrada


#Convertir Variables categóricas de entrada (X)
df_X = df_data.drop('Ingresos', axis=1) #Eliminar la columna ingresos
print(df_X)

df_X = pd.get_dummies(df_X) # Raza y sexo son variables categoricas
# Es necesario codificar las variables categoricas en numericas 

print(df_X)

#Crear un array (dataframe) con las variables de entrada (X) y otro para la variable de salida (y)
X = df_X.values
y = df_data['Ingresos'].values


#Dividir datos en conjunto de "Training" (ej: 80%) y conjunto de "Test" (ej:20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y) #stratify --> datos_etiquetados


# Construir Modelos en base a los diferentes algoritmos y  Evaluar cada modelo con validacion cruzada
#La validación cruzada implica dividir repetidamente los datos en conjuntos de entrenamiento y prueba 
# para evaluar el rendimiento de un modelo de aprendizaje automático. 
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))



results = []
names = []

#models=
#  name,  model
#[('LR', LogisticRegression()), 
# ('LDA', LinearDiscriminantAnalysis()),
#  ('KNN', KNeighborsClassifier()), 
# ('CART', DecisionTreeClassifier()), 
# ('NB', GaussianNB()), 
# ('SVM', SVC())]

for name, model in models:
 kfold = model_selection.KFold(n_splits=10)
 #supongamos que especificamos el pliegue como 10 (k = 10), entonces la validación cruzada de K-Fold divide los datos de entrada en 10 pliegues, 
 # lo que significa que tenemos 10 conjuntos de datos para entrenar y probar nuestro modelo. Entonces, 
 # para cada iteración, el modelo usa un pliegue como datos de prueba y el resto como datos de entrenamiento 
 # (9 pliegues). Cada vez, elige un pliegue diferente para la evaluación y el resultado es una matriz de puntajes 
 # de evaluación para cada pliegue.
 cv_results = model_selection.cross_val_score(model, X, y, cv=kfold)
#cv_results = model_selection.cross_val_score(LogisticRegression(), X, y, cv=kfold)

 results.append(cv_results)
 names.append(name)
# names.append("LR")
 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 print(msg)



lr = LogisticRegression()
lr.fit(X,y)


 # Seleccionar mejor modelo tras benchmarking
# entrenar el modelo
#lda= LinearDiscriminantAnalysis()
#lda.fit(X, y)

# Predecir Resultados de salida (y_prediction) a partir de nuevos datos de entrada (X_new)
df_new = pd.read_excel(r'C:\Users\user\Downloads\Ejercicio+Codificación+CLASIFICACION\Ingresos_nuevos_datos.xlsx')
df_X_new = pd.get_dummies(df_new)
X_new = df_X_new.values


y_prediction = lr.predict(X_new)
print("Prediccion: {}".format(y_prediction))
