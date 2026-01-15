import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
# 1. Define the Input Shape
# X_train. shape[1] is the number of time steps (10)
# X_train.shape[2] is the number of features (approx 70-80 depending on the CSV)
input_shape = (X_train.shape[1], X_train.shape[2])

# 2. Build the Sequential Model
model = Sequential([
    # --- CNN BLOCK: Spatial Feature Extraction ---
    # Extracts local patterns within the flow sequences
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
    MaxPooling1D(pool_size=2),

    # --- LSTM BLOCK: Temporal Learning ---
    # Learns how traffic patterns evolve over the 10-step window
    LSTM(64, return_sequences=False),

    # --- DENSE BLOCK: Decision Making ---
    Dense(64, activation='relu'),
    Dropout(0.5),  # Prevents overfitting by randomly 'turning off' neurons

    # --- OUTPUT LAYER: Binary Prediction ---
    # Sigmoid outputs a value between 0 and 1 (Probability of Attack)
    Dense(1, activation='sigmoid')
])

# 3. Compile the Model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# 4. Display the Architecture
model.summary()