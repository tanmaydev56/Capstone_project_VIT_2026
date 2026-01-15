import pandas as pd
import numpy as np
import glob
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# --- 1. KEY IMPORTS ---
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef, balanced_accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 2. DATA LOADING & CLEANING WITH PROPER SPLITTING ---
print("--- Loading Dataset ---")

csv_files = sorted(glob.glob('Dataset CICIDS/*.csv'))
print(f"Found {len(csv_files)} CSV files")

col_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Src IP', 'Dst IP', 'Flow ID']

# Load all data first
all_dfs = []
for idx, file in enumerate(csv_files):
    print(f"Reading {os.path.basename(file)}...")
    df = pd.read_csv(file, skipinitialspace=True, encoding='utf-8')
    df.drop(columns=[c for c in col_drop if c in df.columns], inplace=True, errors="ignore")
    
    # Convert labels: binary classification
    df['Label'] = df['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
    
    # Handle bad values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    all_dfs.append(df)
    print(f"  File {idx+1}: {len(df)} rows, {df['Label'].mean()*100:.1f}% attacks")

# Combine all data
full_df = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal dataset: {len(full_df)} rows")

# --- 🚨 FIX: BALANCED STRATIFIED SAMPLING 🚨 ---
# Instead of time-based split, use stratified sampling but ensure both classes
# We'll take a smaller sample but maintain class distribution

print("\n[STRATIFIED SAMPLING] Ensuring both classes in all splits...")

# First check class distribution
class_counts = full_df['Label'].value_counts()
print(f"Class distribution: Benign={class_counts.get(0, 0)}, Attack={class_counts.get(1, 0)}")

# Take a balanced sample (adjust sample_size based on your RAM)
sample_size = min(500000, len(full_df))  # Max 500k samples or all if less
if class_counts.get(0, 0) > 0 and class_counts.get(1, 0) > 0:
    # Sample equally from both classes to ensure balance
    benign_samples = full_df[full_df['Label'] == 0].sample(
        n=min(sample_size // 2, class_counts.get(0, 0)), 
        random_state=42
    )
    attack_samples = full_df[full_df['Label'] == 1].sample(
        n=min(sample_size // 2, class_counts.get(1, 0)), 
        random_state=42
    )
    full_df = pd.concat([benign_samples, attack_samples], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
else:
    # If only one class exists, take a regular sample
    full_df = full_df.sample(n=min(sample_size, len(full_df)), random_state=42)

print(f"After balanced sampling: {len(full_df)} rows")
print(f"New distribution: {full_df['Label'].value_counts().to_dict()}")

# --- 3. PREPARE DATA WITH STRATIFIED SPLIT ---
X = full_df.drop('Label', axis=1)
y = full_df['Label'].values

print(f"\nFeatures: {X.shape[1]}, Samples: {X.shape[0]}")

# --- CRITICAL FIX: SCALING AFTER SPLITTING ---
# First split, then scale to avoid data leakage
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train_raw)} samples")
print(f"Test set: {len(X_test_raw)} samples")
print(f"Train class distribution: {np.unique(y_train_raw, return_counts=True)}")
print(f"Test class distribution: {np.unique(y_test_raw, return_counts=True)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw).astype('float32')
X_test_scaled = scaler.transform(X_test_raw).astype('float32')

y_train = y_train_raw.astype('int8')
y_test = y_test_raw.astype('int8')

# --- 4. SLIDING WINDOW CREATION ---
def create_windows(features, labels, window_size=10, step_size=1):
    """
    Create sliding windows with overlap.
    step_size controls overlap (1 = maximum overlap, window_size = no overlap)
    """
    num_samples = (len(features) - window_size) // step_size + 1
    num_features = features.shape[1]
    
    X_out = np.zeros((num_samples, window_size, num_features), dtype='float32')
    y_out = np.zeros((num_samples,), dtype='int8')
    
    for i in range(num_samples):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        X_out[i] = features[start_idx:end_idx]
        
        # Use majority vote for labeling
        window_labels = labels[start_idx:end_idx]
        unique, counts = np.unique(window_labels, return_counts=True)
        y_out[i] = unique[np.argmax(counts)]
    
    return X_out, y_out

print("\n--- Creating Sliding Windows ---")
window_size = 10
step_size = 5  # 50% overlap

X_train_windows, y_train_windows = create_windows(
    X_train_scaled, y_train, window_size=window_size, step_size=step_size
)
X_test_windows, y_test_windows = create_windows(
    X_test_scaled, y_test, window_size=window_size, step_size=step_size
)

print(f"Train windows: {X_train_windows.shape}")
print(f"Test windows: {X_test_windows.shape}")
print(f"Window class distribution - Train: {np.unique(y_train_windows, return_counts=True)}")
print(f"Window class distribution - Test: {np.unique(y_test_windows, return_counts=True)}")

# --- 5. CLASS WEIGHTS FOR IMBALANCED DATA ---
print("\n--- Computing Class Weights ---")
try:
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train_windows), 
        y=y_train_windows
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"Class weights: {class_weight_dict}")
except ValueError as e:
    print(f"Warning: {e}")
    print("Using equal class weights")
    class_weight_dict = {0: 1.0, 1: 1.0}

# --- 6. MODEL ARCHITECTURE ---
HYPERPARAMS = {
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs": 20,       
    "l2_reg": 0.0001,
    "dropout_rate": 0.4,
    "patience": 5
}

def build_enhanced_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        
        Conv1D(filters=64, kernel_size=3, activation='relu', 
               kernel_regularizer=l2(HYPERPARAMS["l2_reg"])),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.0),
        
        Dense(64, activation='relu', kernel_regularizer=l2(HYPERPARAMS["l2_reg"])),
        Dropout(HYPERPARAMS["dropout_rate"]),
        
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=HYPERPARAMS["learning_rate"])
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy', 
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

input_shape = (X_train_windows.shape[1], X_train_windows.shape[2])
print(f"\nInput shape: {input_shape}")

# --- 7. STRATIFIED K-FOLD CROSS-VALIDATION (FIXED) ---
print(f"\n--- Starting {5}-Fold Stratified Cross-Validation ---")

# Use StratifiedKFold to ensure both classes in each fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = {
    'accuracy': [], 'precision': [], 'recall': [], 'auc': [],
    'f1': [], 'mcc': [], 'balanced_accuracy': [], 'fpr': []
}

for fold_idx, (train_index, val_index) in enumerate(skf.split(X_train_windows, y_train_windows), 1):
    print(f"\n[Fold {fold_idx}/5]")
    
    X_train_fold, X_val_fold = X_train_windows[train_index], X_train_windows[val_index]
    y_train_fold, y_val_fold = y_train_windows[train_index], y_train_windows[val_index]
    
    # Check if both classes are present
    unique_classes = np.unique(y_train_fold)
    if len(unique_classes) < 2:
        print(f"  Warning: Only {len(unique_classes)} class(es) in training fold. Skipping...")
        continue
    
    unique_val_classes = np.unique(y_val_fold)
    if len(unique_val_classes) < 2:
        print(f"  Warning: Only {len(unique_val_classes)} class(es) in validation fold. Skipping...")
        continue
    
    # Build fresh model
    model = build_enhanced_model(input_shape)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=HYPERPARAMS["patience"], 
                     restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 
                         min_lr=1e-6, verbose=0)
    ]
    
    # Train with class weights
    history = model.fit(
        X_train_fold, y_train_fold,
        epochs=HYPERPARAMS["epochs"], 
        batch_size=HYPERPARAMS["batch_size"],
        verbose=0,
        validation_data=(X_val_fold, y_val_fold),
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    
    # Predict
    y_pred_prob = model.predict(X_val_fold, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics with safety checks
    try:
        # Basic metrics
        acc = np.mean(y_pred == y_val_fold)
        precision = tf.keras.metrics.Precision()(y_val_fold, y_pred).numpy()
        recall = tf.keras.metrics.Recall()(y_val_fold, y_pred).numpy()
        
        # AUC with safety check
        if len(np.unique(y_val_fold)) > 1:
            auc = roc_auc_score(y_val_fold, y_pred_prob)
        else:
            auc = 0.5  # Neutral value for single class
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # MCC
        mcc = matthews_corrcoef(y_val_fold, y_pred)
        
        # Balanced accuracy
        bal_acc = balanced_accuracy_score(y_val_fold, y_pred)
        
        # FPR
        cm = confusion_matrix(y_val_fold, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn + 1e-8)
        else:
            fpr = 0.0
        
        # Store metrics
        fold_metrics['accuracy'].append(acc)
        fold_metrics['precision'].append(precision)
        fold_metrics['recall'].append(recall)
        fold_metrics['auc'].append(auc)
        fold_metrics['f1'].append(f1)
        fold_metrics['mcc'].append(mcc)
        fold_metrics['balanced_accuracy'].append(bal_acc)
        fold_metrics['fpr'].append(fpr)
        
        print(f"   > Accuracy:  {acc*100:.2f}%")
        print(f"   > Precision: {precision*100:.2f}%")
        print(f"   > Recall:    {recall*100:.2f}%")
        print(f"   > F1-Score:  {f1*100:.2f}%")
        print(f"   > AUC:       {auc:.4f}")
        print(f"   > MCC:       {mcc:.4f}")
        print(f"   > FPR:       {fpr*100:.2f}%")
        
    except Exception as e:
        print(f"  Error calculating metrics: {e}")
        continue

# Check if we have valid folds
if len(fold_metrics['accuracy']) == 0:
    print("\n[WARNING] No valid folds completed. Check your data distribution.")
    print("Proceeding with full training...")
    valid_folds = False
else:
    valid_folds = True

# --- REPORT CROSS-VALIDATION RESULTS ---
if valid_folds:
    print("\n" + "="*60)
    print(f"FINAL CROSS-VALIDATION RESULTS ({len(fold_metrics['accuracy'])} valid folds)")
    print("="*60)
    
    for metric_name, values in fold_metrics.items():
        if values:  # Check if list is not empty
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if metric_name in ['mcc', 'auc']:
                print(f"{metric_name:20s}: {mean_val:.4f} (+/- {std_val:.4f})")
            else:
                print(f"{metric_name:20s}: {mean_val*100:.2f}% (+/- {std_val*100:.2f}%)")
    
    print("="*60)

# --- 8. FINAL MODEL TRAINING ---
print("\n--- Training Final Model on Full Training Set ---")
final_model = build_enhanced_model(input_shape)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=HYPERPARAMS["patience"], 
                 restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 
                     min_lr=1e-6, verbose=1)
]

# Create validation split
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_train_windows, y_train_windows, test_size=0.2, random_state=42, stratify=y_train_windows
)

history = final_model.fit(
    X_train_final, y_train_final,
    epochs=HYPERPARAMS["epochs"],
    batch_size=HYPERPARAMS["batch_size"],
    verbose=1,
    validation_data=(X_val_final, y_val_final),
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# --- 9. FINAL TEST SET EVALUATION ---
print("\n" + "="*60)
print("FINAL TEST SET EVALUATION")
print("="*60)

# Evaluate on test set
test_results = final_model.evaluate(X_test_windows, y_test_windows, verbose=0)
print(f"Test Loss:       {test_results[0]:.4f}")
print(f"Test Accuracy:   {test_results[1]*100:.2f}%")
print(f"Test Precision:  {test_results[2]*100:.2f}%")
print(f"Test Recall:     {test_results[3]*100:.2f}%")
print(f"Test AUC:        {test_results[4]:.4f}")

# Additional metrics
y_test_pred_prob = final_model.predict(X_test_windows, verbose=0).flatten()
y_test_pred = (y_test_pred_prob > 0.5).astype(int)

test_f1 = 2 * (test_results[2] * test_results[3]) / (test_results[2] + test_results[3] + 1e-8)

try:
    test_mcc = matthews_corrcoef(y_test_windows, y_test_pred)
except:
    test_mcc = 0.0

try:
    test_bal_acc = balanced_accuracy_score(y_test_windows, y_test_pred)
except:
    test_bal_acc = test_results[1]

# Confusion matrix
cm = confusion_matrix(y_test_windows, y_test_pred)
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    test_fpr = fp / (fp + tn + 1e-8)
    test_tpr = tp / (tp + fn + 1e-8)
    print(f"Test FPR:        {test_fpr*100:.2f}%")
    print(f"Test TPR:        {test_tpr*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
else:
    print(f"\nConfusion Matrix (simplified):")
    print(cm)

print(f"\nTest F1-Score:   {test_f1*100:.2f}%")
print(f"Test MCC:        {test_mcc:.4f}")
print(f"Test Bal. Acc:   {test_bal_acc*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test_windows, y_test_pred, 
                           target_names=['Benign', 'Attack']))

print("="*60)

# --- 10. SHAP ANALYSIS (OPTIONAL - Can be skipped if slow) ---
run_shap = False  # Set to True if you want SHAP analysis
if run_shap:
    print("\n" + "="*60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("="*60)
    
    # Select background from BENIGN samples only
    benign_indices = np.where(y_train_windows == 0)[0]
    if len(benign_indices) > 50:
        background_indices = np.random.choice(benign_indices, 50, replace=False)
    else:
        background_indices = benign_indices[:min(10, len(benign_indices))]
    
    if len(background_indices) > 0:
        background_data = X_train_windows[background_indices]
        
        # Select a few test samples
        test_indices = np.random.choice(len(X_test_windows), 5, replace=False)
        test_samples = X_test_windows[test_indices]
        
        print(f"Using {len(background_data)} benign samples as background")
        print(f"Explaining {len(test_samples)} test samples")
        
        # Define wrapper function
        def model_predict_wrapper(data_flat):
            data_3d = data_flat.reshape(-1, X_train_windows.shape[1], X_train_windows.shape[2])
            return final_model.predict(data_3d, verbose=0).flatten()
        
        # Prepare flattened data
        background_flat = background_data.reshape(len(background_data), -1)
        test_samples_flat = test_samples.reshape(len(test_samples), -1)
        
        # Initialize KernelExplainer
        print("Initializing KernelExplainer...")
        explainer = shap.KernelExplainer(model_predict_wrapper, background_flat)
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(test_samples_flat, nsamples=50)
        
        # Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, test_samples_flat, show=False)
        plt.title("SHAP Feature Importance", fontsize=14)
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] SHAP plot saved as 'shap_summary.png'")
        plt.show()
    else:
        print("Not enough benign samples for SHAP analysis")

# --- 11. SAVE RESULTS ---
print("\n--- Saving Results ---")
final_model.save('ids_cnn_lstm_final_model.h5')
print("Model saved as 'ids_cnn_lstm_final_model.h5'")

# Save metrics
if valid_folds:
    cv_results = pd.DataFrame({
        'fold': range(1, len(fold_metrics['accuracy']) + 1),
        'accuracy': fold_metrics['accuracy'],
        'precision': fold_metrics['precision'],
        'recall': fold_metrics['recall'],
        'auc': fold_metrics['auc'],
        'f1': fold_metrics['f1'],
        'mcc': fold_metrics['mcc'],
        'balanced_accuracy': fold_metrics['balanced_accuracy'],
        'fpr': fold_metrics['fpr']
    })
    cv_results.to_csv('cross_validation_results.csv', index=False)
    print("Cross-validation results saved as 'cross_validation_results.csv'")

test_results_dict = {
    'accuracy': test_results[1],
    'precision': test_results[2],
    'recall': test_results[3],
    'auc': test_results[4],
    'f1_score': test_f1,
    'mcc': test_mcc,
    'balanced_accuracy': test_bal_acc,
    'fpr': test_fpr if 'test_fpr' in locals() else 0.0,
    'tpr': test_tpr if 'test_tpr' in locals() else 0.0
}

test_results_df = pd.DataFrame([test_results_dict])
test_results_df.to_csv('test_set_results.csv', index=False)
print("Test set results saved as 'test_set_results.csv'")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)