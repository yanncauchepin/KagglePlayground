import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.metrics import accuracy_score
import xgboost as xgb
from catboost import CatBoostClassifier
from skorch import NeuralNetClassifier
from scipy.stats import uniform, randint
from datetime import datetime
from pathlib import Path

N_ITER = 1

try:
    local_data_path = Path("C:\\Users\\yanncauchepin\\Datasets\\Datatables\\kaggle_introvertsextrovertss5e7")
    train_df = pd.read_csv(Path(local_data_path, "train.csv"))
    test_df = pd.read_csv(Path(local_data_path, "test.csv"))
except FileNotFoundError:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

target_col = 'Personality'
le = LabelEncoder()
train_df[target_col] = le.fit_transform(train_df[target_col])

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
    ('imputer', IterativeImputer(max_iter=10, random_state=0)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
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
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
catboost_model = CatBoostClassifier(verbose=0, random_state=42)

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


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

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

print("Starting Randomized Search for hyperparameter tuning...")
random_search.fit(X_train, y_train)

print("\nTuning complete.")
print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation accuracy: {:.4f}".format(random_search.best_score_))

best_model = random_search.best_estimator_

print("\nEvaluating the best model on the validation set...")
y_pred_val = best_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy: {accuracy:.4f}")


print("\nMaking final predictions on the test data...")
test_predictions_encoded = best_model.predict(X_test)
test_predictions = le.inverse_transform(test_predictions_encoded)

submission_df = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_predictions
})

submission_save_path = Path(Path(__file__).parent, f"submission_{accuracy:.4f}.csv")

submission_df.to_csv(submission_save_path, index=False)

print(f"\nSubmission file '{submission_save_path.name}' created successfully!")
print(submission_df.head())