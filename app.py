import os
import io
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename

from utils.dataset_analyzer import analyze_dataset
from preprocessing.data_processor import DataProcessor
from ann.model_builder import build_ann_model
from ann.trainer import train_model
from ann.evaluator import evaluate_classification, evaluate_regression

app = Flask(__name__)
app.secret_key = 'super_secret_key_autoann'
import tempfile
temp_dir = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = os.path.join(temp_dir, 'autoann_data')
app.config['MODELS_FOLDER'] = os.path.join(temp_dir, 'autoann_models')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                df = pd.read_csv(filepath)
                if df.empty or df.shape[1] < 2:
                    flash('Dataset must contain at least 2 columns.', 'error')
                    return redirect(request.url)
                session['filepath'] = filepath
                session['filename'] = filename
                return redirect(url_for('dataset_overview'))
            except Exception as e:
                flash(f'Error reading CSV: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Only CSV.', 'error')
            return redirect(request.url)
    return render_template('index.html')

@app.route('/dataset', methods=['GET', 'POST'])
def dataset_overview():
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        flash("Upload a dataset first.", 'error')
        return redirect(url_for('index'))
    try:
        df = pd.read_csv(filepath)
        stats = analyze_dataset(df)
        if request.method == 'POST':
            target_col = request.form.get('target_column')
            if target_col and target_col in df.columns:
                session['target_column'] = target_col
                return redirect(url_for('preprocessing_setup'))
            else:
                flash("Invalid target column.", 'error')
        return render_template('dataset.html', stats=stats, filename=session.get('filename'))
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/preprocessing')
def preprocessing_setup():
    target_column = session.get('target_column')
    filepath = session.get('filepath')
    if not target_column or not filepath:
        return redirect(url_for('index'))
    
    df = pd.read_csv(filepath)
    unique_vals = df[target_column].nunique()
    is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
    
    if is_numeric and unique_vals > 10:
        problem_type = "Regression"
    else:
        problem_type = "Classification"
    return render_template('preprocessing.html', target_column=target_column, problem_type=problem_type)

@app.route('/build_model', methods=['POST'])
def build_model():
    target_column = session.get('target_column')
    filepath = session.get('filepath')
    problem_type = request.form.get('problem_type')
    test_size = float(request.form.get('test_size', 0.2))
    
    session['problem_type'] = problem_type
    session['test_size'] = test_size
    
    try:
        df = pd.read_csv(filepath)
        df = df.dropna(subset=[target_column])
        if len(df) < 10:
            flash("Dataset too small after dropping missing target values.", "error")
            return redirect(url_for('preprocessing_setup'))
            
        processor = DataProcessor(target_column, problem_type)
        X_train, X_test, y_train, y_test, input_dim, model_type, num_classes = processor.process(df, test_size=test_size)
        
        processor_path = os.path.join(app.config['MODELS_FOLDER'], 'processor.pkl')
        processor.save(processor_path)
        
        model, arch_info = build_ann_model(input_dim, model_type, num_classes)
        
        history = train_model(model, X_train, y_train, X_test, y_test, epochs=20)
        
        model_path = os.path.join(app.config['MODELS_FOLDER'], 'ann_model.keras')
        model.save(model_path)
        
        if problem_type == 'Classification':
            metrics = evaluate_classification(model, X_test, y_test, num_classes==2)
        else:
            metrics = evaluate_regression(model, X_test, y_test)
            
        session['metrics'] = metrics
        session['architecture'] = arch_info
        
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        flash(f"Error during training: {str(e)}", "error")
        return redirect(url_for('preprocessing_setup'))

@app.route('/dashboard')
def dashboard():
    metrics = session.get('metrics')
    arch = session.get('architecture')
    problem_type = session.get('problem_type')
    if not metrics:
        return redirect(url_for('index'))
    return render_template('dashboard.html', metrics=metrics, arch=arch, problem_type=problem_type)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
