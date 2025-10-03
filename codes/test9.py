import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

# Create binary target
y_train_binary = (train_df['song_popularity'] >= 0.5).astype(int)

def advanced_feature_engineering(df):
    """Enhanced feature engineering with more creative features"""
    df_eng = df.copy()
    
    # Basic features
    df_eng['tempo_to_duration'] = df_eng['tempo'] / (df_eng['song_duration_ms'] + 1)
    df_eng['energy_loudness_ratio'] = df_eng['energy'] / (df_eng['loudness'].abs() + 1)
    df_eng['dance_valence_interaction'] = df_eng['danceability'] * df_eng['audio_valence']
    
    # Advanced interaction features
    df_eng['acoustic_energy_balance'] = (df_eng['acousticness'] - df_eng['energy']).abs()
    df_eng['speech_instrument_ratio'] = df_eng['speechiness'] / (df_eng['instrumentalness'] + 0.001)
    df_eng['liveness_energy_product'] = df_eng['liveness'] * df_eng['energy']
    
    # Complex combinations
    df_eng['music_complexity'] = (df_eng['speechiness'] + df_eng['instrumentalness']) * df_eng['tempo']
    df_eng['emotional_intensity'] = (df_eng['audio_valence'] + df_eng['energy'] + df_eng['danceability']) / 3
    df_eng['acoustic_emotional_balance'] = df_eng['acousticness'] * df_eng['emotional_intensity']
    
    # Statistical features
    audio_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                     'liveness', 'loudness', 'speechiness', 'audio_valence']
    
    df_eng['audio_mean'] = df_eng[audio_features].mean(axis=1)
    df_eng['audio_std'] = df_eng[audio_features].std(axis=1)
    df_eng['audio_range'] = df_eng[audio_features].max(axis=1) - df_eng[audio_features].min(axis=1)
    
    # Polynomial features for key interactions
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(df_eng[['energy', 'danceability', 'audio_valence']])
    poly_cols = ['energy_danceability', 'energy_valence', 'danceability_valence', 
                'energy_danceability_valence']
    
    for i, col in enumerate(poly_cols):
        df_eng[col] = poly_features[:, i]
    
    # Normalize some features to avoid extreme values
    df_eng['norm_tempo'] = (df_eng['tempo'] - df_eng['tempo'].mean()) / df_eng['tempo'].std()
    df_eng['norm_duration'] = (df_eng['song_duration_ms'] - df_eng['song_duration_ms'].mean()) / df_eng['song_duration_ms'].std()
    
    return df_eng

def preprocess_data(df, is_train=True, imputer_dict=None, scaler_dict=None, feature_selector=None):
    """Enhanced preprocessing with better feature handling"""
    df_processed = df.copy()
    
    if is_train:
        X = df_processed.drop('song_popularity', axis=1)
    else:
        X = df_processed
    
    # Handle missing values
    numerical_features = ['song_duration_ms', 'acousticness', 'danceability', 'energy', 
                         'instrumentalness', 'liveness', 'loudness', 'speechiness', 
                         'tempo', 'audio_valence']
    
    categorical_features = ['key', 'time_signature', 'audio_mode']
    
    if imputer_dict is None:
        imputer_dict = {
            'num_imputer': KNNImputer(n_neighbors=5),
            'cat_imputer': SimpleImputer(strategy='most_frequent')
        }
    
    # Apply imputation
    if is_train:
        X[numerical_features] = imputer_dict['num_imputer'].fit_transform(X[numerical_features])
        X[categorical_features] = imputer_dict['cat_imputer'].fit_transform(X[categorical_features])
    else:
        X[numerical_features] = imputer_dict['num_imputer'].transform(X[numerical_features])
        X[categorical_features] = imputer_dict['cat_imputer'].transform(X[categorical_features])
    
    # Advanced feature engineering
    X = advanced_feature_engineering(X)
    
    # Identify numerical features
    all_numerical = X.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in all_numerical:
        all_numerical.remove('id')
    
    # Scale numerical features
    if scaler_dict is None:
        scaler_dict = {'scaler': RobustScaler()}
    
    X_scaled = scaler_dict['scaler'].fit_transform(X[all_numerical]) if is_train else scaler_dict['scaler'].transform(X[all_numerical])
    X_final = pd.DataFrame(X_scaled, columns=all_numerical, index=X.index)
    
    # Feature selection
    if is_train and feature_selector is None:
        selector = SelectFromModel(
            XGBClassifier(random_state=42, n_estimators=100, use_label_encoder=False, eval_metric='logloss'),
            threshold='median'
        )
        selector.fit(X_final, y_train_binary)
        feature_selector = selector
        selected_features = X_final.columns[selector.get_support()].tolist()
        feature_selector.selected_features_ = selected_features
        print(f"Selected {len(selected_features)} features out of {len(all_numerical)}")
    
    if feature_selector is not None:
        if hasattr(feature_selector, 'selected_features_'):
            X_final = X_final[feature_selector.selected_features_]
        else:
            X_final = X_final.loc[:, feature_selector.get_support()]
    
    return X_final, imputer_dict, scaler_dict, feature_selector

# Preprocess data
X_train, imputer_dict, scaler_dict, feature_selector = preprocess_data(train_df, is_train=True)
X_test, _, _, _ = preprocess_data(test_df, is_train=False, 
                                 imputer_dict=imputer_dict, 
                                 scaler_dict=scaler_dict,
                                 feature_selector=feature_selector)

# Align features
common_features = list(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_features]
X_test = X_test[common_features]

print(f"Final feature count: {len(common_features)}")

class AdvancedWeightedEnsemble(BaseEstimator, ClassifierMixin):
    """Advanced ensemble with stacking-like capabilities and dynamic weighting"""
    
    def __init__(self, models=None, use_meta_learning=True):
        self.models = models if models is not None else []
        self.use_meta_learning = use_meta_learning
        self.weights = []
        self.meta_weights = None
        
    def _calculate_dynamic_weights(self, X, y):
        """Calculate weights based on cross-validation performance"""
        scores = []
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for name, model in self.models:
            try:
                cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
                avg_score = cv_scores.mean()
            except:
                model.fit(X, y)
                y_pred = model.predict_proba(X)[:, 1]
                avg_score = roc_auc_score(y, y_pred)
            
            scores.append(avg_score)
        
        # Softmax transformation for weights
        scores = np.array(scores)
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        weights = exp_scores / exp_scores.sum()
        
        return weights.tolist()
    
    def _train_meta_learner(self, X, y):
        """Train a meta-learner on base model predictions"""
        from sklearn.linear_model import LogisticRegression
        
        # Get base model predictions
        base_predictions = []
        for name, model in self.models:
            model.fit(X, y)
            pred_proba = model.predict_proba(X)[:, 1]
            base_predictions.append(pred_proba)
        
        # Stack predictions
        X_meta = np.column_stack(base_predictions)
        
        # Train meta-learner
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        meta_learner.fit(X_meta, y)
        
        return meta_learner, base_predictions
    
    def fit(self, X, y):
        # Calculate dynamic weights
        self.weights = self._calculate_dynamic_weights(X, y)
        print(f"Model weights: {self.weights}")
        
        # Train meta-learner if enabled
        if self.use_meta_learning:
            self.meta_learner, _ = self._train_meta_learner(X, y)
        
        # Fit all base models
        fitted_models = []
        for (name, model), weight in zip(self.models, self.weights):
            print(f"Fitting {name} (weight: {weight:.3f})...")
            fitted_model = clone(model)
            fitted_model.fit(X, y)
            fitted_models.append((name, fitted_model))
        
        self.models = fitted_models
        return self
    
    def predict_proba(self, X):
        # Get predictions from all models
        base_predictions = []
        for name, model in self.models:
            pred_proba = model.predict_proba(X)[:, 1]
            base_predictions.append(pred_proba)
        
        if self.use_meta_learning and self.meta_learner is not None:
            # Use meta-learner for final prediction
            X_meta = np.column_stack(base_predictions)
            final_proba = self.meta_learner.predict_proba(X_meta)[:, 1]
        else:
            # Weighted average
            final_proba = np.zeros_like(base_predictions[0])
            for pred, weight in zip(base_predictions, self.weights):
                final_proba += pred * weight
        
        return np.column_stack([1 - final_proba, final_proba])
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

def create_boosted_ensemble(X, y, n_iterations=3):
    """Create an iteratively boosted ensemble"""
    all_models = []
    
    for iteration in range(n_iterations):
        print(f"\n=== Boosting Iteration {iteration + 1} ===")
        
        # Create diverse models for this iteration
        models = [
            ('XGBoost_1', XGBClassifier(
                n_estimators=200 + iteration * 50,
                max_depth=6 - iteration,
                learning_rate=0.05 + iteration * 0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + iteration,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=2.0 + iteration * 0.5,
                tree_method='hist'
            )),
            ('XGBoost_2', XGBClassifier(
                n_estimators=180 + iteration * 40,
                max_depth=5,
                learning_rate=0.06,
                subsample=0.9 - iteration * 0.1,
                colsample_bytree=0.7,
                random_state=43 + iteration,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=2.5,
                tree_method='hist'
            )),
            ('CatBoost_1', CatBoostClassifier(
                iterations=200 + iteration * 30,
                depth=6,
                learning_rate=0.05,
                random_state=44 + iteration,
                verbose=0,
                auto_class_weights='Balanced',
                thread_count=-1
            )),
            ('CatBoost_2', CatBoostClassifier(
                iterations=180,
                depth=5 + iteration,
                learning_rate=0.06 - iteration * 0.01,
                random_state=45 + iteration,
                verbose=0,
                thread_count=-1
            ))
        ]
        
        # Train ensemble for this iteration
        ensemble = AdvancedWeightedEnsemble(models=models, use_meta_learning=True)
        ensemble.fit(X, y)
        
        all_models.extend(ensemble.models)
        
        # Evaluate
        y_pred_proba = ensemble.predict_proba(X)[:, 1]
        auc_score = roc_auc_score(y, y_pred_proba)
        print(f"Iteration {iteration + 1} AUC: {auc_score:.4f}")
    
    # Create final mega-ensemble from all iterations
    final_ensemble = AdvancedWeightedEnsemble(models=all_models, use_meta_learning=True)
    final_ensemble.fit(X, y)
    
    return final_ensemble

def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal threshold for classification"""
    thresholds = np.linspace(0.3, 0.7, 50)
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        score = roc_auc_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold

# Train the boosted ensemble
print("Training boosted ensemble...")
final_ensemble = create_boosted_ensemble(X_train, y_train_binary, n_iterations=2)

# Final evaluation
train_pred_proba = final_ensemble.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train_binary, train_pred_proba)
optimal_threshold = find_optimal_threshold(y_train_binary, train_pred_proba)

print(f"\nFinal Ensemble Performance:")
print(f"Train AUC: {train_auc:.4f}")
print(f"Optimal threshold: {optimal_threshold:.6f}")

# Cross-validation for better estimate
cv_scores = cross_val_score(final_ensemble, X_train, y_train_binary, 
                           cv=3, scoring='roc_auc', n_jobs=-1)
print(f"CV AUC Scores: {cv_scores}")
print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Save models
joblib.dump(final_ensemble, 'advanced_ensemble_model.pkl')
joblib.dump(imputer_dict, 'imputer_dict.pkl')
joblib.dump(scaler_dict, 'scaler_dict.pkl')
joblib.dump(feature_selector, 'feature_selector.pkl')

# Make predictions
print("Making predictions on test set...")
test_pred_proba = final_ensemble.predict_proba(X_test)[:, 1]
# Save probabiliies
submission_proba_df = pd.DataFrame({
    'id': test_df['id'],
    'song_popularity': test_pred_proba
})

submission_proba_df.to_csv("submission_9_probabilities.csv", index=False)

test_predictions = (test_pred_proba >= optimal_threshold).astype(int)

# Create submission
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'song_popularity': test_predictions
})

file_output_name = "submission_9.csv"
submission_df.to_csv(file_output_name, index=False)
print(f"Submission file '{file_output_name}' created!")