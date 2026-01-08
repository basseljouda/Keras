import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense 
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
from tensorflow.keras.layers import Dropout, Input, Dense, BatchNormalization
from tensorflow.keras.models import Model

# Define the input layer
input_layer = Input(shape=(20,))

# Add hidden layers with dropout
"""""
hidden_layer1 = Dense(64, activation='relu')(input_layer)
dropout1 = Dropout(0.5)(hidden_layer1)
hidden_layer2 = Dense(64, activation='relu')(dropout1)
dropout2 = Dropout(0.5)(hidden_layer2)
# Define the output layer
output_layer = Dense(1, activation='sigmoid')(dropout2)
"""
hidden_layer1 = Dense(64, activation='relu')(input_layer)
batch_norm1 = BatchNormalization()(hidden_layer1)
hidden_layer2 = Dense(64, activation='relu')(batch_norm1)
batch_norm2 = BatchNormalization()(hidden_layer2)

# Define the output layer
output_layer = Dense(1, activation='sigmoid')(batch_norm2)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

import numpy as np
X_train = np.random.rand(1000, 20)
y_train = np.random.randint(2, size=(1000, 1))
model.fit(X_train, y_train, epochs=10, batch_size=32)
X_test = np.random.rand(200, 20) 
y_test = np.random.randint(2, size=(200, 1)) 
loss, accuracy = model.evaluate(X_test, y_test) 
print(f'Test loss: {loss}') 
print(f'Test accuracy: {accuracy}') 