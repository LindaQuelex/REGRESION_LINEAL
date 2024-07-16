import pandas as pd
from pycaret.time_series import *

# Crear un ejemplo de dataset mensual para el año 2023
data = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
    'quantity': [150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260]
})
data.set_index('date', inplace=True)

# Configuración del entorno con menor número de pliegues y horizonte de pronóstico
ts = setup(data, fh=3, fold=2, session_id=123)

# Comparación de modelos y selección del mejor
best_model = compare_models()

# Imprimir el mejor modelo
print(best_model)

# Crear y ajustar el mejor modelo
tuned_model = tune_model(best_model)

# Generar pronósticos
predictions = predict_model(tuned_model)

# Visualizar resultados
plot_model(tuned_model, plot='forecast')

# Mostrar predicciones
print(predictions)