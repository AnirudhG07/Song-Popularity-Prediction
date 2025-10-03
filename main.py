import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import optuna
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load train and test data
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)

# Convert to binary classification
y_train_binary = (train_df['song_popularity'] > 0.37).astype(int)
print(f"Target distribution: {np.bincount(y_train_binary)}")

def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal threshold using Youden's J statistic"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def advanced_feature_engineering(X):
    """Advanced feature engineering with domain knowledge"""
    X_featured = X.copy()
    
    # Basic audio features
    numerical_features = ['song_duration_ms', 'acousticness', 'danceability', 'energy', 
                         'instrumentalness', 'liveness', 'loudness', 'speechiness', 
                         'tempo', 'audio_valence']
    
    # 1. Interaction features (domain knowledge)
    X_featured['energy_danceability'] = X_featured['energy'] * X_featured['danceability']
    X_featured['valence_energy'] = X_featured['audio_valence'] * X_featured['energy']
    X_featured['acoustic_energy'] = X_featured['acousticness'] * X_featured['energy']
    X_featured['dance_valence'] = X_featured['danceability'] * X_featured['audio_valence']
    X_featured['loudness_energy'] = X_featured['loudness'] * X_featured['energy']
    X_featured['speechiness_energy'] = X_featured['speechiness'] * X_featured['energy']
    
    # 2. Ratio features
    X_featured['energy_loudness_ratio'] = X_featured['energy'] / (abs(X_featured['loudness']) + 1e-5)
    X_featured['danceability_energy_ratio'] = X_featured['danceability'] / (X_featured['energy'] + 1e-5)
    X_featured['acousticness_energy_ratio'] = X_featured['acousticness'] / (X_featured['energy'] + 1e-5)
    X_featured['valence_energy_ratio'] = X_featured['audio_valence'] / (X_featured['energy'] + 1e-5)
    
    # 3. Polynomial features
    X_featured['energy_squared'] = X_featured['energy'] ** 2
    X_featured['danceability_squared'] = X_featured['danceability'] ** 2
    X_featured['loudness_squared'] = X_featured['loudness'] ** 2
    X_featured['tempo_squared'] = X_featured['tempo'] ** 2
    
    # 4. Duration features
    X_featured['duration_minutes'] = X_featured['song_duration_ms'] / 60000
    X_featured['duration_seconds'] = X_featured['song_duration_ms'] / 1000
    X_featured['log_duration'] = np.log1p(X_featured['song_duration_ms'])
    
    # 5. Loudness features
    X_featured['loudness_abs'] = abs(X_featured['loudness'])
    X_featured['loudness_normalized'] = (X_featured['loudness'] - X_featured['loudness'].min()) / (X_featured['loudness'].max() - X_featured['loudness'].min() + 1e-5)
    
    # 6. Tempo features (keep as numerical)
    X_featured['tempo_zscore'] = (X_featured['tempo'] - X_featured['tempo'].mean()) / (X_featured['tempo'].std() + 1e-5)
    
    # 7. Audio feature combinations
    X_featured['audio_intensity'] = X_featured['energy'] + X_featured['loudness'] / 10
    X_featured['melodic_complexity'] = X_featured['instrumentalness'] + X_featured['speechiness']
    X_featured['emotional_tone'] = X_featured['audio_valence'] * (1 - X_featured['acousticness'])
    
    # 8. Statistical features
    X_featured['feature_sum'] = X_featured[['energy', 'danceability', 'loudness']].sum(axis=1)
    X_featured['feature_mean'] = X_featured[['energy', 'danceability', 'loudness', 'audio_valence']].mean(axis=1)
    X_featured['feature_std'] = X_featured[['energy', 'danceability', 'loudness']].std(axis=1)
    
    # 9. Binning features (convert to numerical for XGBoost compatibility)
    X_featured['energy_bin'] = pd.qcut(X_featured['energy'], q=5, labels=False, duplicates='drop').astype(int)
    X_featured['danceability_bin'] = pd.qcut(X_featured['danceability'], q=5, labels=False, duplicates='drop').astype(int)
    X_featured['valence_bin'] = pd.qcut(X_featured['audio_valence'], q=5, labels=False, duplicates='drop').astype(int)
    
    # 10. Interaction with categorical features (convert to numerical)
    X_featured['energy_key'] = X_featured['energy'] * X_featured['key']
    X_featured['danceability_mode'] = X_featured['danceability'] * X_featured['audio_mode']
    
    # Convert original categorical features to numerical
    X_featured['key'] = X_featured['key'].astype(int)
    X_featured['audio_mode'] = X_featured['audio_mode'].astype(int)
    X_featured['time_signature'] = X_featured['time_signature'].astype(int)
    
    return X_featured

def preprocess_data(df, is_train=True, imputer_dict=None, scaler_dict=None, feature_selector=None):
    """
    Preprocess the data with proper handling for train/test sets
    """
    df_processed = df.copy()
    
    # Separate features and target if it's training data
    if is_train:
        X = df_processed.drop('song_popularity', axis=1)
    else:
        X = df_processed
    
    # Handle missing values
    numerical_features = ['song_duration_ms', 'acousticness', 'danceability', 'energy', 
                         'instrumentalness', 'liveness', 'loudness', 'speechiness', 
                         'tempo', 'audio_valence']
    
    categorical_features = ['key', 'time_signature', 'audio_mode']
    
    # Initialize imputers if not provided
    if imputer_dict is None:
        imputer_dict = {
            'num_imputer': KNNImputer(n_neighbors=7),
            'cat_imputer': SimpleImputer(strategy='most_frequent')
        }
    
    # Apply imputation
    X[numerical_features] = imputer_dict['num_imputer'].fit_transform(X[numerical_features]) if is_train else imputer_dict['num_imputer'].transform(X[numerical_features])
    X[categorical_features] = imputer_dict['cat_imputer'].fit_transform(X[categorical_features]) if is_train else imputer_dict['cat_imputer'].transform(X[categorical_features])
    
    # Advanced feature engineering
    X = advanced_feature_engineering(X)
    
    # Identify numerical features after engineering
    all_numerical = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID from numerical features
    if 'id' in all_numerical:
        all_numerical.remove('id')
    
    # Scale numerical features with RobustScaler
    if scaler_dict is None:
        scaler_dict = {
            'scaler': RobustScaler()
        }
    
    X_scaled = scaler_dict['scaler'].fit_transform(X[all_numerical]) if is_train else scaler_dict['scaler'].transform(X[all_numerical])
    
    # Create final DataFrame with ALL features first
    X_final = pd.DataFrame(X_scaled, columns=all_numerical, index=X.index)
    
    # Store all feature names before selection for reference
    all_feature_names = X_final.columns.tolist()
    
    # Feature selection (only on training)
    if is_train and feature_selector is None:
        selector = SelectFromModel(
            XGBClassifier(random_state=42, n_estimators=100),
            threshold='median'
        )
        selector.fit(X_final, y_train_binary)
        feature_selector = selector
        # Store the selected feature names
        selected_features = X_final.columns[selector.get_support()].tolist()
        feature_selector.selected_features_ = selected_features
        print(f"Selected {len(selected_features)} features out of {len(all_feature_names)}")
    
    # Apply feature selection to both train and test
    if feature_selector is not None:
        if hasattr(feature_selector, 'selected_features_'):
            # Use the stored feature names
            X_final = X_final[feature_selector.selected_features_]
        else:
            # Fallback to selector method
            X_final = X_final.loc[:, feature_selector.get_support()]
    
    return X_final, imputer_dict, scaler_dict, feature_selector

# Preprocess training data
X_train, imputer_dict, scaler_dict, feature_selector = preprocess_data(train_df, is_train=True)

# Preprocess test data - make sure to use the same feature_selector
X_test, _, _, _ = preprocess_data(test_df, is_train=False, 
                                 imputer_dict=imputer_dict, 
                                 scaler_dict=scaler_dict,
                                 feature_selector=feature_selector)

print("Processed train features shape:", X_train.shape)
print("Processed test features shape:", X_test.shape)
print("Train features:", X_train.columns.tolist())
print("Test features:", X_test.columns.tolist())

# Verify feature alignment
if not X_train.columns.equals(X_test.columns):
    print("WARNING: Feature mismatch detected!")
    print("Missing in test:", set(X_train.columns) - set(X_test.columns))
    print("Extra in test:", set(X_test.columns) - set(X_train.columns))
    
    # Align features manually
    common_features = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    print(f"Using {len(common_features)} common features")

# Optuna optimization functions
def objective_xgb(trial):
    """Objective function for XGBoost hyperparameter optimization"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 3.0),
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
    }
    
    model = XGBClassifier(
        **params,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss',
        tree_method='hist'
    )
    
    score = cross_val_score(model, X_train, y_train_binary, cv=3, scoring='roc_auc', n_jobs=-1).mean()
    return score

def objective_catboost(trial):
    """Objective function for CatBoost hyperparameter optimization"""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20),
    }
    
    model = CatBoostClassifier(
        **params,
        random_state=42,
        verbose=0,
        auto_class_weights='Balanced',
        thread_count=-1
    )
    
    score = cross_val_score(model, X_train, y_train_binary, cv=3, scoring='roc_auc', n_jobs=-1).mean()
    return score

# Run Optuna optimizations
print("Optimizing XGBoost hyperparameters with Optuna...")
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=True)

print("Optimizing CatBoost hyperparameters with Optuna...")
study_catboost = optuna.create_study(direction='maximize')
study_catboost.optimize(objective_catboost, n_trials=30, show_progress_bar=True)

print("\nBest XGBoost parameters:")
print(study_xgb.best_params)
print(f"Best XGBoost AUC: {study_xgb.best_value:.4f}")

print("\nBest CatBoost parameters:")
print(study_catboost.best_params)
print(f"Best CatBoost AUC: {study_catboost.best_value:.4f}")

# Define a much more efficient ensemble class
class FastWeightedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, models=None, weights=None):
        self.models = models if models is not None else []
        self.weights = weights if weights is not None else []
        
    def get_params(self, deep=True):
        return {
            'models': self.models,
            'weights': self.weights
        }
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
    
    def _calculate_fast_weights(self, X, y):
        """Fast weight calculation using quick CV"""
        scores = []
        for name, model in self.models:
            # Quick 2-fold CV for speed
            try:
                cv_score = cross_val_score(model, X, y, cv=2, scoring='roc_auc', n_jobs=-1).mean()
            except:
                # If CV fails, use simple score
                model.fit(X, y)
                y_pred = model.predict_proba(X)[:, 1]
                cv_score = roc_auc_score(y, y_pred)
            scores.append(cv_score)
        
        scores = np.array(scores)
        # Add small epsilon to avoid zero weights
        weights = (scores + 0.01) / (scores.sum() + 0.01 * len(scores))
        return weights.tolist()
    
    def fit(self, X, y):
        if not self.weights:
            self.weights = self._calculate_fast_weights(X, y)
        
        # Fit models sequentially (more reliable than parallel for this case)
        fitted_models = []
        for name, model in self.models:
            print(f"Fitting {name}...")
            fitted_model = clone(model)
            fitted_model.fit(X, y)
            fitted_models.append((name, fitted_model))
        
        self.models = fitted_models
        return self
    
    def predict_proba(self, X):
        # Predict sequentially
        probas = []
        for name, model in self.models:
            proba = model.predict_proba(X)[:, 1]
            probas.append(proba)
        
        # Weighted average
        weighted_sum = np.zeros_like(probas[0])
        for proba, weight in zip(probas, self.weights):
            weighted_sum += proba * weight
        
        return np.column_stack([1 - weighted_sum, weighted_sum])
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

# Train individual models with optimized hyperparameters
print("Training individual models with optimized hyperparameters...")

# XGBoost with optimized parameters
xgb_model_1 = XGBClassifier(
    **study_xgb.best_params,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='logloss',
    tree_method='hist'
)

# Create a slightly different XGBoost with variations for diversity
xgb_params_2 = study_xgb.best_params.copy()
xgb_params_2['learning_rate'] = xgb_params_2['learning_rate'] * 0.8  # Slightly lower learning rate
xgb_params_2['subsample'] = max(0.7, xgb_params_2['subsample'] * 0.9)  # Slightly different subsample

xgb_model_2 = XGBClassifier(
    **xgb_params_2,
    random_state=43,  # Different seed for diversity
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='logloss',
    tree_method='hist'
)

# CatBoost with optimized parameters
catboost_model_1 = CatBoostClassifier(
    **study_catboost.best_params,
    random_state=42,
    verbose=0,
    auto_class_weights='Balanced',
    thread_count=-1
)

# Train models
individual_models = {
    'XGBoost_1': xgb_model_1,
    'XGBoost_2': xgb_model_2,
    'CatBoost_1': catboost_model_1
}

for name, model in individual_models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train_binary)
    # Quick score calculation
    y_pred = model.predict_proba(X_train)[:, 1]
    train_score = roc_auc_score(y_train_binary, y_pred)
    print(f"{name}: Train AUC = {train_score:.4f}")

# Create ensemble with pre-trained models
base_models = [
    ('xgb1', xgb_model_1),
    ('xgb2', xgb_model_2),
    ('cat1', catboost_model_1)
]

# Create and train ensemble quickly
print("\nCreating fast ensemble...")
ensemble = FastWeightedEnsemble(models=base_models)
ensemble.fit(X_train, y_train_binary)

# Quick evaluation
train_pred_proba = ensemble.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train_binary, train_pred_proba)
optimal_threshold = find_optimal_threshold(y_train_binary, train_pred_proba)

print(f"Ensemble Train AUC: {train_auc:.4f}")
print(f"Optimal threshold: {optimal_threshold:.6f}")

# Use a simple holdout validation for quick ensemble evaluation
X_train_fast, X_val_fast, y_train_fast, y_val_fast = train_test_split(
    X_train, y_train_binary, test_size=0.2, random_state=42, stratify=y_train_binary
)

# Quick ensemble validation
val_ensemble = FastWeightedEnsemble(models=base_models)
val_ensemble.fit(X_train_fast, y_train_fast)
val_pred_proba = val_ensemble.predict_proba(X_val_fast)[:, 1]
val_auc = roc_auc_score(y_val_fast, val_pred_proba)

print(f"Validation AUC: {val_auc:.4f}")

# Final ensemble - use the one trained on full data
final_ensemble = ensemble
final_pred_proba = final_ensemble.predict_proba(X_train)[:, 1]
final_auc = roc_auc_score(y_train_binary, final_pred_proba)
final_threshold = find_optimal_threshold(y_train_binary, final_pred_proba)

print(f"\nFinal Ensemble Performance:")
print(f"Train AUC: {final_auc:.4f}")
print(f"Optimal threshold: {final_threshold:.6f}")

# Save the final ensemble
joblib.dump(final_ensemble, 'fast_ensemble_model.pkl')
joblib.dump(imputer_dict, 'imputer_dict.pkl')
joblib.dump(scaler_dict, 'scaler_dict.pkl')
joblib.dump(feature_selector, 'feature_selector.pkl')

# Make predictions on test set
print("Making predictions on test set...")
test_pred_proba = final_ensemble.predict_proba(X_test)[:, 1]
test_predictions = (test_pred_proba >= final_threshold).astype(int)
# Create submission file
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'song_popularity': test_predictions
})

# Save Probabilities as well
submission_df_proba = pd.DataFrame({
    'id': test_df['id'],
    'song_popularity': test_pred_proba
})
submission_df_proba.to_csv("submission_10_probabilities.csv", index=False)

# Save predictions
file_output_name = "submission_10.csv"
submission_df.to_csv(file_output_name, index=False)
print(f"Submission file '{file_output_name}' created!")