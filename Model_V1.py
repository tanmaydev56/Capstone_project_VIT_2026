import pandas as pd
import numpy as np
import glob
from pygments.lexers.textfmts import TodotxtLexer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# here we are selecting all the files in cicids folder by the use of glob
csv_files = glob.glob('Dataset CICIDS/*.csv')
# Columns to drop (Non-negotiable)
col_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']

data_frame = []
for file in csv_files:
    df = pd.read_csv(file,skipinitialspace=True)
    df.drop(columns=col_drop,inplace=True,errors="ignore")
    data_frame.append(df)

# merging all files into one large datset
full_df = pd.concat(data_frame,ignore_index=True)

# 3.3 Label Conversion: Binary Classification
# 0 for BENIGN, 1 for everything else (attacks)

full_df['Label'] = full_df['Label'].apply(lambda x:0 if x == 'BENIGN' else 1)

# 3.4 Handling Bad Values (Infinity and NaN)
# Replace inf and -inf with NaN
full_df.replace([np.inf,-np.inf], np.nan, inplace=True)

# droping all rows with Nan values
full_df.dropna(inplace=True)

print(f"Dataset cleaned. Remaining rows: {len(full_df)}")


# FEATURE SCALING

# Separate features (X) and labels (y)
X = full_df.drop('Label', axis=1)
y = full_df['Label'].values

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype('float32')
y = y.astype('int8') # Labels only need 1 byte, not 8!


 # TODO future improvement
 # To avoid data leakage (very important academically).
# scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

print("Features scaled using StandardScaler.")

# why not min max scalar


# 3.6: Sliding Window Creation (The "Spatio-Temporal" Logic)
# This is where we transform the flat data into a 3D Tensor
# $(Samples, Time\_Steps, Features)$ that the CNN-LSTM needs.

# Use a Pre-allocated NumPy Array (Efficiency)
# Your current create_windows function uses a Python List (X_windows.append).
# Appending to a list 2.8 million times is very slow and memory-intensive.
# It is better to "pre-allocate" a NumPy array of zeros and fill it.

def create_windows(features, labels, window_size=10):
    # Pre-allocate memory instead of using .append()
    num_samples = len(features) - window_size
    num_features = features.shape[1]

    # Initialize zero arrays with the correct types
    X_out = np.zeros((num_samples, window_size, num_features), dtype='float32')
    y_out = np.zeros((num_samples,), dtype='int8')

    for i in range(num_samples):
        X_out[i] = features[i: i + window_size]
        y_out[i] = labels[i + window_size - 1]

    return X_out, y_out
# def create_windows(features, labels, window_size=10, stride=1):
#     X_windows = []
#     y_windows = []
#
#     for i in range(0, len(features) - window_size, stride):
#         # Extract window of size 'window_size'
#         window = features[i: i + window_size]
#         # The label for this window is the label of the LAST flow in the window
#         label = labels[i + window_size - 1]
#
#         X_windows.append(window)
#         y_windows.append(label)
#
#     return np.array(X_windows), np.array(y_windows)



# Apply windowing
# Note: For massive datasets, window_size=10 is standard for IDS


X_tensor, y_tensor = create_windows(X_scaled, y)
print(f"Final Input Shape: {X_tensor.shape}")  # Should be (N, 10, Features)


# 70% Training, 30% Testing
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.3, random_state=42, stratify=y_tensor
)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")


# model architecture


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# 1. Define the Input Shape
# X_train.shape[1] is the number of time steps (10)
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



# TRAINING OF THE MODEL

from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# --- 5.1: Handle Class Imbalance ---
# Since Benign traffic is much more than Attack traffic, we give
# 'Attacks' a higher weight so the model pays more attention to them.
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"Class Weights (0: Benign, 1: Attack): {class_weight_dict}")

# --- 5.2: Training the Model ---
print("\n--- Starting Training (Baseline) ---")
# Using 5 epochs because the dataset is massive (nearly 2 million rows)
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=256,         # Balanced size for speed and stability
    validation_split=0.1,   # Use 10% of training data to check for overfitting
    class_weight=class_weight_dict,
    verbose=1
)

# --- 5.3: Evaluation on Test Data ---
print("\n--- Final Evaluation on Test Set ---")
results = model.evaluate(X_test, y_test, verbose=1)

# Print results clearly
metrics_names = model.metrics_names
for name, value in zip(metrics_names, results):
    print(f"Test {name.capitalize()}: {value:.4f}")

# --- 5.4: Visualizing Training Progress (Optional but recommended) ---
plt.figure(figsize=(12, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()

plt.show()