import tensorflow as tf

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history.history
