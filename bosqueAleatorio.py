from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar el dataset iris
iris = load_iris()
X = iris.data  # Características (features)
y = iris.target  # Etiquetas (labels)

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar las características (opcional, pero recomendado)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear el clasificador de Bosque Aleatorio
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42) #utilizando 100 árboles en el bosque

# Entrenar el modelo
rf_clf.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = rf_clf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo de Bosque Aleatorio: {accuracy * 100:.2f}%")
