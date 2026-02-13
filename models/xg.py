import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN
import xgboost as xgb
import joblib


# =========================
# CONFIG - TUNABLE PARAMETERS
# =========================

DATASET = 2

if DATASET == 0:
    DATA_DIR = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\clean"
    OUTPUT_DIR = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\results\xgboost"
    
if DATASET == 1:
    DATA_DIR = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\clean2"
    OUTPUT_DIR = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\results\xgboost(2)"
    
if DATASET == 2:
    DATA_DIR = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\clean3"
    OUTPUT_DIR = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\results\xgboost(3)"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "filled": "filled.csv",
    "wma": "wma.csv",
    "ultrasonic_raw": "ultrasonic_raw.csv",
    "raw": "raw.csv",
    "pir": "pir.csv"
}

FEATURE_SETS = {
    "full": ["pir_left", "pir_right", "us_left", "us_mid", "us_right"],
    "left": ["pir_left", "us_left", "us_mid"],
    "right": ["pir_right", "us_mid", "us_right"],
    "ultrasonic_only": ["us_left", "us_mid", "us_right"], 
    "pir_only": ["pir_left", "pir_right"]
}

LABEL_COL = "yolo_total"

# ============== XGBOOST TUNING PARAMETERS ==============
TEST_SIZE = 0.3
RANDOM_STATE = 42

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'max_depth': 6,              # Maximum tree depth
    'learning_rate': 0.25,        # Step size shrinkage (eta)
    'n_estimators': 100,         # Number of boosting rounds
    'min_child_weight': 1,       # Minimum sum of instance weight needed in a child
    'gamma': 0.1,                # Minimum loss reduction required to make a split
    'subsample': 0.8,            # Subsample ratio of the training instances
    'colsample_bytree': 0.8,     # Subsample ratio of columns when constructing each tree
    'reg_alpha': 0.1,            # L1 regularization term
    'reg_lambda': 1.0,           # L2 regularization term
    'scale_pos_weight': 1.0,     # Balance of positive and negative weights (will be calculated)
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc', 'error'],
    'use_label_encoder': False,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,                # Use all CPU cores
    'tree_method': 'hist',       # Faster histogram-based algorithm
}

DECISION_THRESHOLD = 0.5         # Decision threshold for classification
USE_SMOTE = False                 # Whether to use SMOTE/ADASYN
SMOTE_STRATEGY = 1            # Partial balance (50% minority vs majority)
USE_ADASYN = False                # Use ADASYN instead of SMOTE if True
OPTIMIZE_METRIC = "recall"           # "f1", "recall", or "precision"
EARLY_STOPPING_ROUNDS = 20       # Stop if no improvement for N rounds
# ========================================================


def find_best_threshold(y_probs, y_true):
    """Find optimal threshold by maximizing F1 score"""
    best_f1, best_t = 0, 0.5
    for t in np.linspace(0.1, 0.9, 81):
        preds = (y_probs > t).astype(int)
        report = classification_report(y_true, preds, output_dict=True, zero_division=0)
        f1 = report.get("1", {}).get("f1-score", 0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def calculate_metrics(y_true, y_pred):
    """Calculate precision, recall, and F1 score"""
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


# =========================
# PIPELINE
# =========================
results = []
# Store models, scalers, and features for best model saving
model_artifacts = []

for name, file in FILES.items():
    path = os.path.join(DATA_DIR, file)

    if not os.path.exists(path):
        print(f"Skipping missing file: {file}")
        continue

    df = pd.read_csv(path)

    if LABEL_COL not in df.columns:
        print(f"Skipping {file} (no label column)")
        continue

    y = (df[LABEL_COL] > 0).astype(int).values

    for feat_name, features in FEATURE_SETS.items():
        if not all(col in df.columns for col in features):
            continue

        print(f"\n{'='*70}")
        print(f"Dataset: {name} | Features: {feat_name}")
        print(f"{'='*70}")

        X = df[features].values

        # Scale (XGBoost doesn't strictly need scaling, but it can help with convergence)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train/test split
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Further split train into train/validation for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE
        )

        # Apply SMOTE or ADASYN if enabled
        if USE_SMOTE:
            print(f"Before resampling - Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
            
            if USE_ADASYN:
                resampler = ADASYN(
                    sampling_strategy=SMOTE_STRATEGY,
                    random_state=RANDOM_STATE,
                    n_neighbors=5
                )
                resample_name = "ADASYN"
            else:
                resampler = SMOTE(
                    sampling_strategy=SMOTE_STRATEGY,
                    random_state=RANDOM_STATE
                )
                resample_name = "SMOTE"
            
            X_train, y_train = resampler.fit_resample(X_train, y_train)
            print(f"After {resample_name}  - Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")

        # Calculate scale_pos_weight (ratio of negative to positive samples)
        class_counts = np.bincount(y_train)
        scale_pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0
        print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

        # Update XGBoost params with calculated scale_pos_weight
        params = XGBOOST_PARAMS.copy()
        params['scale_pos_weight'] = scale_pos_weight

        # Create DMatrix objects for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Training with early stopping
        print(f"\nTraining XGBoost (optimizing {OPTIMIZE_METRIC})...")
        
        evals = [(dtrain, 'train'), (dval, 'validation')]
        evals_result = {}
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=evals,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            evals_result=evals_result,
            verbose_eval=10  # Print every 10 rounds
        )
        
        print(f"\nBest iteration: {model.best_iteration}")
        print(f"Best score: {model.best_score:.4f}")

        # Find optimal threshold on validation set
        val_probs = model.predict(dval)
        best_threshold, best_val_f1 = find_best_threshold(val_probs, y_val)
        print(f"\nOptimal threshold on validation set: {best_threshold:.3f} | Val F1: {best_val_f1:.4f}")

        # Test with optimized threshold
        y_probs = model.predict(dtest)
        y_pred = (y_probs > best_threshold).astype(int)
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n{'='*70}")
        print(f"TEST RESULTS (threshold={best_threshold:.3f})")
        print(f"{'='*70}")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test Recall (Class 1): {report.get('1', {}).get('recall', 0):.4f}")
        print(f"Test Precision (Class 1): {report.get('1', {}).get('precision', 0):.4f}")
        print(f"Test F1 (Class 1): {report.get('1', {}).get('f1-score', 0):.4f}")
        print("\nFull Classification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=["No Person", "Person Present"], 
            zero_division=0
        ))

        # Feature importance
        importance = model.get_score(importance_type='gain')
        print("\nFeature Importance (gain):")
        for feat_idx, feat in enumerate(features):
            feat_key = f"f{feat_idx}"
            imp_value = importance.get(feat_key, 0)
            print(f"  {feat}: {imp_value:.2f}")

        # Store model artifacts for potential saving later
        model_artifacts.append({
            'model': model,
            'scaler': scaler,
            'features': features,
            'dataset': name,
            'feature_set': feat_name,
            'f1_score': report.get('1', {}).get('f1-score', 0),
            'threshold': best_threshold
        })

        results.append({
            "dataset": name,
            "features": feat_name,
            "accuracy": acc,
            "threshold": best_threshold,
            "best_iteration": model.best_iteration,
            
            # Class 0 (no person)
            "precision_0": report.get("0", {}).get("precision", 0),
            "recall_0": report.get("0", {}).get("recall", 0),
            "f1_0": report.get("0", {}).get("f1-score", 0),
            "support_0": report.get("0", {}).get("support", 0),

            # Class 1 (person present)
            "precision_1": report.get("1", {}).get("precision", 0),
            "recall_1": report.get("1", {}).get("recall", 0),
            "f1_1": report.get("1", {}).get("f1-score", 0),
            "support_1": report.get("1", {}).get("support", 0),
        })

# =========================
# SAVE BEST MODEL ONLY
# =========================
if model_artifacts:
    # Find the best model based on F1 score
    best_artifact = max(model_artifacts, key=lambda x: x['f1_score'])
    
    print("\n" + "="*70)
    print("SAVING BEST MODEL")
    print("="*70)
    print(f"Best configuration:")
    print(f"  Dataset: {best_artifact['dataset']}")
    print(f"  Features: {best_artifact['feature_set']}")
    print(f"  F1 Score: {best_artifact['f1_score']:.4f}")
    print(f"  Threshold: {best_artifact['threshold']:.3f}")
    
    # Save best model
    model_path = os.path.join(
        OUTPUT_DIR,
        f"xgboost_best_model.json"
    )
    best_artifact['model'].save_model(model_path)
    print(f"\nSaved best model to: {model_path}")
    
    # Save scaler for best model
    scaler_path = os.path.join(
        OUTPUT_DIR,
        f"scaler_best_model.pkl"
    )
    joblib.dump(best_artifact['scaler'], scaler_path)
    print(f"Saved scaler to: {scaler_path}")
    
    # Save feature list and metadata
    metadata = {
        'dataset': best_artifact['dataset'],
        'feature_set': best_artifact['feature_set'],
        'features': best_artifact['features'],
        'f1_score': best_artifact['f1_score'],
        'threshold': best_artifact['threshold']
    }
    metadata_path = os.path.join(OUTPUT_DIR, "best_model_metadata.pkl")
    joblib.dump(metadata, metadata_path)
    print(f"Saved metadata to: {metadata_path}")

# =========================
# SAVE RESULTS
# =========================
results_df = pd.DataFrame(results)
resampler_name = "adasyn" if USE_ADASYN else "smote" if USE_SMOTE else "no_resample"
results_csv = os.path.join(OUTPUT_DIR, f"xgboost_{resampler_name}_{OPTIMIZE_METRIC}_results.csv")
results_df.to_csv(results_csv, index=False)

print("\n" + "="*70)
print(f"FINAL RESULTS - SORTED BY F1 SCORE")
print("="*70)
print("\nSaved results to:", results_csv)
print(results_df.sort_values("f1_1", ascending=False))

# =========================
# PLOT METRICS COMPARISON
# =========================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
labels = results_df["dataset"] + " | " + results_df["features"]
x_pos = np.arange(len(labels))

# Plot 1: Recall
axes[0, 0].bar(x_pos, results_df["recall_1"], color='green', alpha=0.7)
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(labels, rotation=45, ha="right")
axes[0, 0].set_ylabel("Recall (Class 1)")
axes[0, 0].set_title("Recall for Person Detection")
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Precision
axes[0, 1].bar(x_pos, results_df["precision_1"], color='orange', alpha=0.7)
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(labels, rotation=45, ha="right")
axes[0, 1].set_ylabel("Precision (Class 1)")
axes[0, 1].set_title("Precision for Person Detection")
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: F1 Score
axes[1, 0].bar(x_pos, results_df["f1_1"], color='purple', alpha=0.7)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(labels, rotation=45, ha="right")
axes[1, 0].set_ylabel("F1 Score (Class 1)")
axes[1, 0].set_title("F1 Score for Person Detection")
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Accuracy
axes[1, 1].bar(x_pos, results_df["accuracy"], color='blue', alpha=0.7)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(labels, rotation=45, ha="right")
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].set_title("Overall Accuracy")
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, f"xgboost_{resampler_name}_{OPTIMIZE_METRIC}_metrics.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nSaved metrics plot to: {plot_path}")

# =========================
# PLOT TRAINING HISTORY (FIRST MODEL)
# =========================
if evals_result:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    train_loss = evals_result['train']['logloss']
    val_loss = evals_result['validation']['logloss']
    epochs = range(len(train_loss))
    
    axes[0].plot(epochs, train_loss, label='Training Loss', color='blue', alpha=0.7)
    axes[0].plot(epochs, val_loss, label='Validation Loss', color='orange', alpha=0.7)
    axes[0].set_xlabel('Boosting Round')
    axes[0].set_ylabel('Log Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot AUC
    train_auc = evals_result['train']['auc']
    val_auc = evals_result['validation']['auc']
    
    axes[1].plot(epochs, train_auc, label='Training AUC', color='blue', alpha=0.7)
    axes[1].plot(epochs, val_auc, label='Validation AUC', color='orange', alpha=0.7)
    axes[1].set_xlabel('Boosting Round')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Training and Validation AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    history_path = os.path.join(OUTPUT_DIR, f"xgboost_{resampler_name}_training_history.png")
    plt.savefig(history_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training history plot to: {history_path}")

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"Optimization metric: {OPTIMIZE_METRIC}")
print(f"Resampling method: {resampler_name}")
if USE_SMOTE:
    print(f"SMOTE/ADASYN strategy: {SMOTE_STRATEGY}")
print(f"\nXGBoost Parameters:")
print(f"  Max depth: {XGBOOST_PARAMS['max_depth']}")
print(f"  Learning rate: {XGBOOST_PARAMS['learning_rate']}")
print(f"  N estimators: {XGBOOST_PARAMS['n_estimators']}")
print(f"  L1 regularization: {XGBOOST_PARAMS['reg_alpha']}")
print(f"  L2 regularization: {XGBOOST_PARAMS['reg_lambda']}")
print(f"\nAverage Metrics:")
print(f"  Recall:    {results_df['recall_1'].mean():.4f} ± {results_df['recall_1'].std():.4f}")
print(f"  Precision: {results_df['precision_1'].mean():.4f} ± {results_df['precision_1'].std():.4f}")
print(f"  F1 Score:  {results_df['f1_1'].mean():.4f} ± {results_df['f1_1'].std():.4f}")
print(f"  Accuracy:  {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
print(f"\nBest performing configuration:")
best_idx = results_df['f1_1'].idxmax()
best_config = results_df.iloc[best_idx]
print(f"  Dataset: {best_config['dataset']}")
print(f"  Features: {best_config['features']}")
print(f"  F1 Score: {best_config['f1_1']:.4f}")
print(f"  Recall: {best_config['recall_1']:.4f}")
print(f"  Precision: {best_config['precision_1']:.4f}")
print(f"  Accuracy: {best_config['accuracy']:.4f}")