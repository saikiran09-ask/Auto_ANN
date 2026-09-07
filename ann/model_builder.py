import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_ann_model(input_dim, model_type, num_classes=1):
    model = Sequential()
    
    # Simple heuristic-based architecture
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.2))
    
    model.add(Dense(32, activation='relu'))
    
    if 'classification' in model_type:
        if num_classes == 2 or model_type == 'binary_classification':
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            output_activation = 'sigmoid'
        else:
            model.add(Dense(num_classes, activation='softmax'))
            # since labels are encoded as integers
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            output_activation = 'softmax'
    else:
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        output_activation = 'linear'
        
    architecture_info = {
        'input_dim': input_dim,
        'layers': [
            {'type': 'Dense', 'neurons': 64, 'activation': 'relu'},
            {'type': 'Dropout', 'rate': 0.2},
            {'type': 'Dense', 'neurons': 32, 'activation': 'relu'},
            {'type': 'Dense', 'neurons': 1 if (num_classes==2 or 'regression' in model_type) else num_classes, 'activation': output_activation}
        ],
        'optimizer': 'adam',
        'loss': model.loss
    }
    
    return model, architecture_info
