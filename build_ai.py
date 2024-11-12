from keras.api.models import Sequential
from keras.api.layers import Dense, BatchNormalization, Dropout
from keras.api.optimizers import Adam


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

from data_processing import load_dataset

def build_model(X_train, func='relu', eta=0.001, dropout_rate=0.3):
    model = Sequential()

    # Input layer and first hidden layer
    model.add(Dense(units=128, activation=func, input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # More hidden layers
    model.add(Dense(units=64, activation=func))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(units=32, activation=func))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(units=16, activation=func))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(units=1, activation='sigmoid'))
    
    optimizer = Adam(eta)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model



def build_ai():
    X_train, X_test, y_train, y_test = load_dataset()


    model = build_model(X_train=X_train)

    history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))


    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Final Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

    # Making predictions
    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int)  # Convert probabilities to 0 or 1

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    # Compare predictions to actual labels
    print("Predictions:", predictions.flatten())
    
    model.save("loan_approval_model.h5")
    

build_ai()