import os
import pandas as pd
from flask import Flask, render_template, request, send_file
from io import BytesIO
from pycaret.classification import *
from pycaret.regression import *

app = Flask(__name__)

UPLOAD_FOLDER = 'docs/uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}

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
            
            # Aquí debes agregar el código para generar pronósticos con algoritmos de PyCaret
            
            # Inicializa el entorno PyCaret para regresión
            exp_reg = setup(data=df, target='CANTIDAD ANUAL', fold_strategy='timeseries', session_id=123)
            
            # Comparar varios modelos
            best = compare_models(fold=3, sort='RMSE', n_select=1)
            
            # Crear modelo
            final_model = finalize_model(best)
            
            # Predicciones
            pred_df = predict_model(final_model)
            
            # Guardar el DataFrame en un archivo Excel
            forecast_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'pronósticos.xlsx')
            pred_df.to_excel(forecast_filename, index=False)
            path_doc_dowload = 'uploads/pronósticos.xlsx'
            return send_file(path_doc_dowload, as_attachment=True)
        
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
