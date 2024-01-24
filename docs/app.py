import os
import pandas as pd
from flask import Flask, render_template, request, send_file
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from io import BytesIO
from pycaret.internal.pycaret_experiment import TimeSeriesExperiment

app = Flask(__name__)

UPLOAD_FOLDER = 'docs\\uploads'
ALLOWED_EXTENSIONS = {'xlsx','cvs'}

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
            y = df['CANTIDAD ANUAL'].values
            estudio= df['NOMBRE DEL ESTUDIO'].values
            año= df['AÑO'].values
            
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
          
            
            # Crear un nuevo DataFrame con resultados de pronóstico
            forecast_data = pd.DataFrame({'FLATTEN': X_test.flatten(),  'CANTIDAD ANUAL REAL': y_test, 'CANTIDAD ANUAL PRONOSTICADA': y_pred, 'ERROR MEDIO CUADRÁTICO': mse})
            
            # Guardar el DataFrame en un archivo Excel
            forecast_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'pronósticos.xlsx')
            forecast_data.to_excel(forecast_filename, index=False)
            path_doc_dowload='uploads\\pronósticos.xlsx'
            return send_file(path_doc_dowload, as_attachment=True)
        
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)