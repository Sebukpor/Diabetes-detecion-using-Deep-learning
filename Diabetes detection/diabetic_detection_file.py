# -*- coding: utf-8 -*-
"""Diabetic_detection

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mkDn_CFlTgHOTXcOAiGDFItNzbZsBSuF
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("/content/diabetes_prediction_dataset.csv")

# Check for and handle missing values in labels
data.dropna(subset=['diabetes'], inplace=True)

data.head()

data.tail()

# Separate features and labels
X = data.drop('diabetes', axis=1)
Y = data['diabetes']

print(X)
print(Y)

# Ensure labels are integers
Y = Y.astype(int)

# Encode categorical features
le = LabelEncoder()
X['gender'] = le.fit_transform(X['gender'])
X['smoking_history'] = le.fit_transform(X['smoking_history'])

# Standardize the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Check the shapes of X_scaled and Y to ensure they match
print(f"X_scaled shape: {X_scaled.shape}")
print(f"Y shape: {Y.shape}")

# Ensure data cardinality matches
assert X_scaled.shape[0] == Y.shape[0], "Mismatch in number of samples between X and Y"

mean_values = scaler.mean_
scale_values = scaler.scale_

print("Mean Values:", mean_values)
print("Scale Values:", scale_values)

"""# 0---> Non-diabetic

# 1---> diabetic

"""

# Define the neural network model
model = tf.keras.Sequential(name="my_neural_network_model")
model.add(tf.keras.layers.Dense(units=16, input_shape=(X_scaled.shape[1],), activation='relu'))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model to the training data
model.fit(X_scaled, Y, epochs=10, batch_size=10, validation_split=0.2)

# Save the model to a file
model.save('my_neural_network_model.h5')

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_scaled, Y, verbose=2)

# Print the evaluation metrics
print("Test accuracy:", accuracy)
print("Test loss:", loss)

# Fit the model to the training data
history = model.fit(X_scaled, Y, epochs=10, batch_size=15, validation_split=0.2)

# Plot the training and validation accuracy and loss at each epoch
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

from sklearn.metrics import f1_score, confusion_matrix

# Make predictions on the test data
y_pred = model.predict(X_scaled)

from sklearn.metrics import f1_score, confusion_matrix

# Calculate F1 score
f1 = f1_score(Y, Y, average='binary')
print("F1 score:", f1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(Y, Y)
print("Confusion matrix:\n", conf_matrix)

"""## Preprocess user inputs"""

# Load the trained model
model = tf.keras.models.load_model('my_neural_network_model.h5')

# Load the original dataset for preprocessing purposes
original_data = pd.read_csv("/content/diabetes_prediction_dataset.csv")

# Drop rows with missing target values
original_data.dropna(subset=['diabetes'], inplace=True)

# Separate features and labels
X = original_data.drop('diabetes', axis=1)

# Encode categorical features
le_gender = LabelEncoder().fit(X['gender'])
le_smoking_history = LabelEncoder().fit(X['smoking_history'])
X['gender'] = le_gender.transform(X['gender'])
X['smoking_history'] = le_smoking_history.transform(X['smoking_history'])

# Standardize the feature values
scaler = StandardScaler().fit(X)

# Function to preprocess new input data
def preprocess_input(data, scaler, le_gender, le_smoking_history):
    # Convert to DataFrame if necessary
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    elif isinstance(data, list):
        data = pd.DataFrame(data)

    # Encode categorical features
    data['gender'] = le_gender.transform(data['gender'])
    data['smoking_history'] = le_smoking_history.transform(data['smoking_history'])

    # Standardize numerical features
    data_scaled = scaler.transform(data)

    return data_scaled

"""# Testing with user data

"""

# Sample new data for prediction
new_data = {
    'gender': 'Male',
    'age': 50,
    'hypertension': 1,
    'heart_disease': 0,
    'smoking_history': 'current',
    'bmi': 78.5,
    'HbA1c_level': 5.9,
    'blood_glucose_level': 200
}

# Assuming 'scaler', 'le_gender', and 'le_smoking_history' are pre-defined
new_data_scaled = preprocess_input(new_data, scaler, le_gender, le_smoking_history)

# Make predictions
predictions = model.predict(new_data_scaled)

# Decode the predicted class
predicted_class = np.argmax(predictions, axis=1)

# Define labels for interpretation
class_labels = {0: "non-diabetic", 1: "diabetic"}

# Output the predictions with labels
predicted_label = class_labels[predicted_class[0]]
print(f"Predicted class: {predicted_class[0]} ({predicted_label})")
print(f"Prediction confidence: {predictions[0]}")  # Probabilities for each class