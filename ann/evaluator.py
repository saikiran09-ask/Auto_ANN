import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

def evaluate_classification(model, X_test, y_test, is_binary):
    y_pred_prob = model.predict(X_test)
    if is_binary:
        y_pred = (y_pred_prob > 0.5).astype(int)
    else:
        y_pred = np.argmax(y_pred_prob, axis=1)
        
    acc = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    recall = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'confusion_matrix': cm.tolist()
    }

def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_test = np.array(y_test)
    y_pred = y_pred.flatten()
    
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    
    return {
        'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2
    }
