import pandas as pd
from pycaret.time_series import *

# Crear un ejemplo de dataset anual
data = pd.DataFrame({
    'date': pd.date_range(start='2015-01-01', periods=8, freq='Y'),
    'quantity': [100, 120, 130, 140, 150, 160, 170, 180]
})
data.set_index('date', inplace=True)

# Configuración del entorno
ts = setup(data, fh=2, fold=2, session_id=123)

# Comparación de modelos
best_model = compare_models()

# Crear y ajustar el modelo
model = create_model('arima')
tuned_model = tune_model(model)

# Generar pronósticos
predictions = predict_model(tuned_model)

# Visualizar resultados
plot_model(tuned_model, plot='forecast')