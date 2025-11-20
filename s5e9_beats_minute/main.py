import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from functools import reduce

ROOT_PATH = Path(__file__).parent

try:
    local_data_path = Path("c:\\users\\yanncauchepin\\Datasets\\Datatables\\kaggle_beatsminutes5e9\\")
    train_df = pd.read_csv(Path(local_data_path, "train.csv"))
    test_df = pd.read_csv(Path(local_data_path, "test.csv"))
except FileNotFoundError:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

now = datetime.now().strftime("%Y%m%d_%H%M%S")

print(train_df.shape)
for col in train_df.columns:
    print(col, train_df[col].dtype, np.round(len(train_df[col].value_counts())/len(train_df[col])*100, 2), train_df[col].isna().sum())
    
target = "BeatsPerMinute"

features = [col for col in train_df.columns if col not in ["id", "BeatsPerMinute"]]

X = train_df[features].values
y = train_df[target].values
X_test = test_df[features].values

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1))
X_test = scaler_X.transform(X_test)

def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

sequence_length = 30
X_train, y_train = create_sequences(X, y, sequence_length)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

def build_transformer_regressor(input_shape, head_size, num_heads, ff_dim, num_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    x = GlobalAveragePooling1D(data_format="channels_last")(x)
    
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
        
    outputs = Dense(1, name='output')(x)
    
    return Model(inputs, outputs)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    ff_x = Dense(ff_dim, activation="relu")(x)
    ff_x = Dense(inputs.shape[-1])(ff_x)
    ff_x = Dropout(dropout)(ff_x)
    return LayerNormalization(epsilon=1e-6)(x + ff_x)

input_shape = (X_train.shape[1], X_train.shape[2])
model = build_transformer_regressor(
    input_shape, head_size=256, num_heads=4, ff_dim=4,
    num_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25,
)


model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='mean_squared_error',
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

if len(X_train) > 0 and len(y_train) > 0:
    model.fit(
        X_train, y_train, epochs=2, batch_size=32, 
        validation_data=(X_val, y_val), verbose=1, callbacks=[early_stopping]
    )

def create_test_sequences(X, seq_length):
    Xs = []
    for i in range(len(X) - seq_length + 1):
        Xs.append(X[i:(i + seq_length)])
    return np.array(Xs)

X_test_seq = create_test_sequences(X_test, sequence_length)

if len(X_test_seq) > 0:
    y_pred_scaled = model.predict(X_test_seq)
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    print("Predictions generated.")
    print(y_pred[:5]) 

    num_predictions = len(y_pred)
    num_test_samples = len(test_df)
    

    full_y_pred = np.full((num_test_samples, 1), np.nan)
    

    full_y_pred[:sequence_length - 1] = y_pred[0] 
    full_y_pred[sequence_length - 1:] = y_pred


    submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': full_y_pred.flatten()})
    
    submission_path = ROOT_PATH / f"submission_{now}.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")
    print(submission_df.head())

else:
    print("Test data is not long enough to create sequences for prediction.")

