# Importa las bibliotecas necesarias
import pandas as pd
from pycaret.timeseries import *

# Carga tus datos
data = pd.read_csv('datos.csv')

# Establece el índice de tiempo si no está establecido
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')

# Dividir datos en conjuntos de entrenamiento y prueba
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Inicializa el entorno de PyCaret para análisis de series temporales
exp = setup(data=train, session_id=123, silent=True, handle_unknown_categorical=True, fold_strategy='expanding')

# Entrena modelos de series temporales
best_model = compare_models(fold=3)

# Pronóstico utilizando el mejor modelo
predictions = predict_model(best_model)

# Muestra los resultados del pronóstico
print(predictions)
