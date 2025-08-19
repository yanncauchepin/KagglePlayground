import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, ParameterSampler
from sklearn.metrics import accuracy_score
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import uniform, randint
from pathlib import Path
from datetime import datetime
from skorch.callbacks import EarlyStopping

LOCAL = True

try:
    local_data_path = Path("c:\\users\\cauchepy\\Datasets\\Datatables\\kaggle_classificationbanks5e8\\")
    # local_data_path = Path("/home/yanncauchepin/Datasets/Datatables/kaggle_classificationbanks5e8/")
    train_df = pd.read_csv(Path(local_data_path, "train.csv"))
    test_df = pd.read_csv(Path(local_data_path, "test.csv"))
except FileNotFoundError:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
    LOCAL = False
    
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
    X[col] = X[col].astype('category').cat.codes
    X_test[col] = X_test[col].astype('category').cat.codes
    
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
    ],
    remainder='passthrough'
)

# --- Start of new code ---
N_ITER = 1
# --- End of new code ---

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

mlp_model = NeuralNetClassifier(
    MLP,
    module__input_size=len(numerical_features) + len(categorical_features),
    module__output_size=2,
    module__hidden_sizes=[128, 64, 32], 
    module__dropout_rate=0.3, 
    iterator_train__drop_last=True,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    train_split=None,
    verbose=0,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    callbacks=[EarlyStopping(monitor='valid_acc', patience=10, lower_is_better=False)]
)

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
    # ('preprocessor', preprocessor),
    ('classifier', voting_clf)
])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# Preprocess data before splitting
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)

# Split processed data
X_train_processed, X_val_processed, y_train, y_val = train_test_split(X_processed, y, test_size=0.15, random_state=42)


param_distributions = {
    'classifier__xgb__n_estimators': randint(500, 1500),
    'classifier__xgb__learning_rate': uniform(0.01, 0.2),
    'classifier__xgb__max_depth': randint(3, 8),
    'classifier__catboost__iterations': randint(500, 1500),
    'classifier__catboost__learning_rate': uniform(0.01, 0.2),
    'classifier__catboost__depth': randint(3, 8),
    'classifier__mlp__max_epochs': randint(20, 100),
    'classifier__mlp__optimizer__lr': uniform(0.0005, 0.005),
    'classifier__mlp__module__hidden_sizes': [[64, 32], [128, 64, 32], [256, 128]],
    'classifier__mlp__module__dropout_rate': uniform(0.2, 0.4)
}


# --- Start of new code ---
if N_ITER == 1:
    print("Running a single attempt without cross-validation...")

    # Sample one set of parameters
    param_sampler = ParameterSampler(param_distributions, n_iter=1, random_state=int(datetime.now().strftime("%M%S")))
    params = list(param_sampler)[0]
    
    print("\nSelected parameters for this run:")
    print(params)

    # Set parameters to the pipeline
    final_pipeline.set_params(**params)

    # Fit the model
    print("\nTraining the model...")
    fit_params = {
        'classifier__xgb__eval_set': [(X_val_processed, y_val)],
        'classifier__catboost__eval_set': [(X_val_processed, y_val)],
        'classifier__mlp__X': X_train_processed,
        'classifier__mlp__y': y_train,
        'classifier__mlp__validation_data': (X_val_processed, y_val)
    }
    final_pipeline.fit(X_train_processed, y_train, **fit_params)

    # Evaluate on validation set
    print("\nEvaluating the model on the validation set...")
    y_pred_val = final_pipeline.predict(X_val_processed)
    accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Validation Accuracy: {accuracy:.4f}")

    best_model = final_pipeline
    val_score = accuracy

else:
    print(f"Starting Randomized Search for hyperparameter tuning with n_iter={N_ITER}...")
    random_search = RandomizedSearchCV(
        final_pipeline,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        cv=3,      
        scoring='accuracy',
        verbose=2,
        random_state=int(datetime.now().strftime("%M%S")),
        n_jobs=-1
    )

    fit_params = {
        'classifier__xgb__eval_set': [(X_val_processed, y_val)],
        'classifier__catboost__eval_set': [(X_val_processed, y_val)],
        'classifier__mlp__X': X_train_processed,
        'classifier__mlp__y': y_train,
        'classifier__mlp__validation_data': (X_val_processed, y_val)
    }
    random_search.fit(X_train_processed, y_train, **fit_params)

    print("\nTuning complete.")
    print("Best parameters found: ", random_search.best_params_)
    print("Best cross-validation accuracy: {:.4f}".format(random_search.best_score_))

    best_model = random_search.best_estimator_
    val_score = random_search.best_score_

    # Re-evaluate on the validation set to get a final validation score
    print("\nEvaluating the best model on the validation set...")
    y_pred_val = best_model.predict(X_val_processed)
    accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Validation Accuracy: {accuracy:.4f}")
    val_score = accuracy # Use the direct validation score for the filename

print("\nMaking final predictions on the test data...")
test_predictions_proba = best_model.predict_proba(X_test_processed)

submission_df = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_predictions_proba[:,1]
})

submission_filename = f'submission_{val_score:.4f}.csv'
if LOCAL:
    ROOT_DIR = Path(__file__).parent
    submission_df.to_csv(Path(ROOT_DIR, submission_filename), index=False)
else:
    submission_df.to_csv(submission_filename, index=False)

print(f"\nSubmission file '{submission_filename}' created successfully!")
print(submission_df.head())
# --- End of new code ---