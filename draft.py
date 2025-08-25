import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import accuracy_score
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import uniform, randint
from pathlib import Path
from datetime import datetime
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit # <-- Added import
import sklearn
from sklearn.base import clone # <-- Added import

N_ITER = 20

LOCAL = True

try:
    # local_data_path = Path("c:\\users\\cauchepy\\Datasets\\Datatables\\kaggle_classificationbanks5e8\\")
    local_data_path = Path("/home/yanncauchepin/Datasets/Datatables/kaggle_classificationbanks5e8/")
    train_df = pd.read_csv(Path(local_data_path, "train.csv"))
    test_df = pd.read_csv(Path(local_data_path, "test.csv"))
except FileNotFoundError:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
    LOCAL = False

# This is not needed for this script, but it's good practice to have.
sklearn.set_config(enable_metadata_routing=True)
    
target_col = "y"

X = train_df.drop(columns=[target_col, 'id'])
y = train_df[target_col].values.astype(np.int64)
X_test = test_df.drop(columns=['id'])

numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

for col in numerical_features:
    X[col] = X[col].astype(np.float32)
    X_test[col] = X_test[col].astype(np.float32)

for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])
    
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
    ],
    remainder='passthrough'
)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(MLP, self).__init__()
        layers = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_size
        layers.append(nn.Linear(current_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.float() 
        return self.layers(x)

# --- FIX 1: Correctly configure skorch model for early stopping ---
mlp_model = NeuralNetClassifier(
    MLP,
    module__input_size=len(numerical_features) + len(categorical_features),
    module__output_size=2,
    iterator_train__shuffle=True,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    # Use skorch's internal validation split for early stopping
    train_split=ValidSplit(cv=0.15, stratified=True, random_state=42),
    verbose=0,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    # Monitor validation loss, which is more stable than accuracy
    callbacks=[EarlyStopping(monitor='valid_loss', patience=10, lower_is_better=True)]
)

# --- FIX 2: Simplify model definitions ---
# The `early_stopping_rounds` parameter is sufficient. The model will automatically use
# the `eval_set` provided during the .fit() call.
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, early_stopping_rounds=10)
catboost_model = CatBoostClassifier(verbose=0, random_state=42, early_stopping_rounds=10)

voting_clf = VotingClassifier(
    estimators=[
        ('mlp', mlp_model),
        ('xgb', xgb_model),
        ('catboost', catboost_model)
    ],
    voting='soft'
)

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', voting_clf)
])

# --- FIX 3: Use the entire dataset (X, y) for the pipeline ---
# The pipeline will handle preprocessing.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

param_distributions = {
    # We add prefixes to easily assign params to the right model
    'xgb__n_estimators': randint(500, 1500),
    'xgb__learning_rate': uniform(0.01, 0.2),
    'xgb__max_depth': randint(3, 8),
    'catboost__iterations': randint(500, 1500),
    'catboost__learning_rate': uniform(0.01, 0.2),
    'catboost__depth': randint(3, 8),
    'mlp__max_epochs': randint(20, 100),
    'mlp__optimizer__lr': uniform(0.0005, 0.005),
    'mlp__module__hidden_sizes': [[64, 32], [128, 64, 32], [256, 128]],
    'mlp__module__dropout_rate': uniform(0.2, 0.4)
}

# --- DEFINITIVE FIX: Manual Ensemble Loop to Bypass Routing Issues ---

print(f"Starting manual hyperparameter search with {N_ITER} iteration(s)...")

param_sampler = ParameterSampler(
    param_distributions,
    n_iter=N_ITER,
    random_state=int(datetime.now().strftime("%M%S"))
)

best_model = None
best_score = -1
best_params = {}

# Preprocess the data ONCE before the loop
preprocessor.fit(X_train, y_train)
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)

for i, params in enumerate(param_sampler):
    print(f"\n--- Iteration {i + 1}/{N_ITER} ---")
    
    # 1. Separate params for each model
    mlp_params = {k.replace('mlp__', ''): v for k, v in params.items() if k.startswith('mlp__')}
    xgb_params = {k.replace('xgb__', ''): v for k, v in params.items() if k.startswith('xgb__')}
    cat_params = {k.replace('catboost__', ''): v for k, v in params.items() if k.startswith('catboost__')}
    
    print("Parameters:", params)

    # 2. Instantiate and fit each model individually
    print("\nTraining models...")
    try:
        # --- MLP ---
        mlp = clone(mlp_model).set_params(**mlp_params)
        mlp.fit(X_train_processed, y_train)
        
        # --- XGBoost ---
        xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, early_stopping_rounds=10)
        xgb_clf.set_params(**xgb_params)
        xgb_clf.fit(X_train_processed, y_train, eval_set=[(X_val_processed, y_val)], verbose=False)

        # --- CatBoost ---
        cat_clf = CatBoostClassifier(random_state=42, early_stopping_rounds=10)
        cat_clf.set_params(**cat_params)
        cat_clf.fit(X_train_processed, y_train, eval_set=[(X_val_processed, y_val)], verbose=False)

        # 3. Get predictions and manually average them (soft voting)
        print("Evaluating ensemble...")
        pred_mlp = mlp.predict_proba(X_val_processed)
        pred_xgb = xgb_clf.predict_proba(X_val_processed)
        pred_cat = cat_clf.predict_proba(X_val_processed)

        # Average the probabilities for the positive class
        avg_preds_proba = (pred_mlp + pred_xgb + pred_cat) / 3
        # Convert averaged probabilities to class labels
        y_pred_val = np.argmax(avg_preds_proba, axis=1)

        accuracy = accuracy_score(y_val, y_pred_val)
        print(f"Validation Accuracy for this iteration: {accuracy:.4f}")

        if accuracy > best_score:
            best_score = accuracy
            best_params = params
            print(f"** New best score found: {best_score:.4f} **")

    except Exception as e:
        print(f"An error occurred during iteration {i+1}: {e}")
        continue


print("\n--- Search Complete ---")
print(f"Best validation accuracy: {best_score:.4f}")
print("Best parameters found:", best_params)

# 4. Manually train the final models using the best parameters found
print("\nTraining final model with best parameters...")

# First, set the best parameters on the pipeline's classifier
final_params = {f'classifier__{k}': v for k, v in best_params.items()}
final_pipeline.set_params(**final_params)

# Extract the final configured classifier
final_classifier = final_pipeline.named_steps['classifier']

# Extract each individual model from the final classifier
mlp_final = final_classifier.named_estimators['mlp']
xgb_final = final_classifier.named_estimators['xgb']
cat_final = final_classifier.named_estimators['catboost']

# Manually fit each model on the processed training data, using the
# processed validation data for early stopping where needed.
print("Fitting final MLP...")
mlp_final.fit(X_train_processed, y_train)

print("Fitting final XGBoost model...")
xgb_final.fit(X_train_processed, y_train, eval_set=[(X_val_processed, y_val)], verbose=False)

print("Fitting final CatBoost model...")
cat_final.fit(X_train_processed, y_train, eval_set=[(X_val_processed, y_val)], verbose=False)

print("Final model training complete.")

val_score = best_score 

# 5. Make final predictions
print("\nMaking final predictions on the test data...")

# Preprocess the test data using the preprocessor fitted on the training data
X_test_processed = preprocessor.transform(X_test)

# Use the fully-trained classifier to predict
pred_mlp = mlp_final.predict_proba(X_test_processed)
pred_xgb = xgb_final.predict_proba(X_test_processed)
pred_cat = cat_final.predict_proba(X_test_processed)

# Average the probabilities for the positive class
avg_preds_proba = (pred_mlp + pred_xgb + pred_cat) / 3


submission_df = pd.DataFrame({
    'id': test_df['id'],
    'y': avg_preds_proba[:,1]
})

submission_filename = f'submission_{val_score:.4f}.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"\nSubmission file '{submission_filename}' created successfully!")
print(submission_df.head())
