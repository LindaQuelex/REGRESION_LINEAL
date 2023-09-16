import os
import pandas as pd
from flask import Flask, render_template, request, send_file
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from io import BytesIO

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            
            df = pd.read_excel(filename)
            # Aquí debes agregar el código para generar pronósticos con algoritmos de ML
            

            # Ejemplo: Linear Regression
            X = df.index.values.reshape(-1, 1)
            y = df['cantidad mensual'].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            # Crear un nuevo DataFrame con resultados de pronóstico
            forecast_data = pd.DataFrame({'Fecha': X_test.flatten(), 'Valor Real': y_test, 'Valor Pronosticado': y_pred})
            
            # Guardar el DataFrame en un archivo Excel
            forecast_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'forecast_result.xlsx')
            forecast_data.to_excel(forecast_filename, index=False)
            
            return send_file(forecast_filename, as_attachment=True)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)