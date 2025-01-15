from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Cargar el dataset iris
iris = load_iris() #Utilizamos el conjunto de datos iris. Este dataset contiene 150 muestras de flores iris, con 4 características (sepal length, sepal width, petal length, petal width) y 3 clases (setosa, versicolor, virginica).
X = iris.data  # Características (features)
y = iris.target  # Etiquetas (labels)

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar las características 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear el clasificador KNN
knn = KNeighborsClassifier(n_neighbors=3) #k = 3  # Número de vecinos más cercanos

# Entrenar el modelo
knn.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = knn.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo KNN: {accuracy * 100:.2f}%")
