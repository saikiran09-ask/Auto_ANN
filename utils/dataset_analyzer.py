import pandas as pd
import numpy as np

def analyze_dataset(df):
    stats = {}
    stats['num_rows'] = int(df.shape[0])
    stats['num_cols'] = int(df.shape[1])
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    stats['num_numerical'] = len(numerical_cols)
    stats['num_categorical'] = len(categorical_cols)
    
    missing = df.isnull().sum().sum()
    stats['missing_values'] = int(missing)
    stats['duplicate_rows'] = int(df.duplicated().sum())
    
    columns_info = []
    for col in df.columns:
        col_info = {
            'name': col,
            'type': str(df[col].dtype),
            'missing': int(df[col].isnull().sum()),
            'unique': int(df[col].nunique())
        }
        columns_info.append(col_info)
        
    stats['columns_info'] = columns_info
    
    preview_df = df.head(15).replace({np.nan: "NaN"})
    stats['preview'] = preview_df.to_dict(orient='records')
    
    return stats
