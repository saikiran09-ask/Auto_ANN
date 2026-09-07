import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

class DataProcessor:
    def __init__(self, target_column, problem_type):
        self.target_column = target_column
        self.problem_type = problem_type
        self.preprocessor = None
        self.label_encoder = None
        
    def process(self, df, test_size=0.2, random_state=42):
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Handle target column
        if self.problem_type == 'Classification':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            num_classes = len(self.label_encoder.classes_)
            if num_classes == 2:
                model_type = 'binary_classification'
            else:
                model_type = 'multiclass_classification'
        else:
            y = y.values
            model_type = 'regression'
            num_classes = 1
            
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if self.problem_type == 'Classification' else None)
        
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        input_dim = X_train_processed.shape[1]
        
        return X_train_processed, X_test_processed, y_train, y_test, input_dim, model_type, num_classes
        
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'preprocessor': self.preprocessor,
                'label_encoder': self.label_encoder,
                'problem_type': self.problem_type,
                'target_column': self.target_column
            }, f)
            
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        obj = cls(data['target_column'], data['problem_type'])
        obj.preprocessor = data['preprocessor']
        obj.label_encoder = data['label_encoder']
        return obj
