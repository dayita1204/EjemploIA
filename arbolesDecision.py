from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar el dataset iris
iris = load_iris()
X = iris.data  # Características (features)
y = iris.target  # Etiquetas (labels)

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar las características 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear el clasificador de Árbol de Decisión
tree_clf = DecisionTreeClassifier(random_state=42) #ontrolar la aleatoriedad en el proceso de entrenamiento y asegurar resultados reproducibles

# Entrenar el modelo
tree_clf.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = tree_clf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo de Árbol de Decisión: {accuracy * 100:.2f}%")
