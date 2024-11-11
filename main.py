from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from data_processing import load_dataset

def build_model(X_train, func='relu', eta=0.001):
    
    model = Sequential()
    
    model.add(Dense(units=64, activation=func, input_dim=X_train.shape[1]))
    model.add(Dense(units=32, activation=func))
    model.add(Dense(units=1, activation='sigmoid'))
    
    optimizer = Adam(eta)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model



X_train, X_test, y_train, y_test = load_dataset()


model = build_model(X_train=X_train)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


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

# Compare predictions to actual labels
print("Predictions:", predictions.flatten())