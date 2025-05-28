import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from collections import Counter
import os
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.callbacks import Callback

class LoggingCallback(Callback):
    def __init__(self, log_callback, total_epochs):
        super().__init__()
        self.log_callback = log_callback
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        if self.log_callback:
            msg = f"Epoch [{epoch + 1}/{self.total_epochs}]  " \
                  f"Train Loss: {logs.get('loss', 0):.6f}  " \
                  f"Val Loss: {logs.get('val_loss', 0):.6f}"
            self.log_callback(msg)

class StopTrainingCallback(Callback):
    def __init__(self, stop_flag_getter):
        super().__init__()
        self.stop_flag_getter = stop_flag_getter

    def on_epoch_end(self, epoch, logs=None):
        if self.stop_flag_getter and self.stop_flag_getter():
            print("🛑 Training stopped by user.")
            self.model.stop_training = True


# === MODELS ===

class ImprovedLSTMModel:
    def __init__(self, input_shape, num_classes=2, dropout=0.3, num_layers=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_layers = num_layers

    def build(self):
        model = keras.Sequential()
        model.add(layers.Input(shape=self.input_shape))
        model.add(layers.LSTM(64, return_sequences=True, dropout=self.dropout if self.num_layers > 1 else 0))
        model.add(layers.LSTM(32, return_sequences=False))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(32, activation='relu'))
        if self.num_classes == 1:
            model.add(layers.Dense(1, activation='sigmoid'))
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
        return model

class RNNModel:
    def __init__(self, input_shape, num_classes=2, dropout=0.3, num_layers=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_layers = num_layers

    def build(self):
        model = keras.Sequential()
        model.add(layers.Input(shape=self.input_shape))
        model.add(layers.SimpleRNN(64, return_sequences=True, dropout=self.dropout if self.num_layers > 1 else 0))
        model.add(layers.SimpleRNN(32, return_sequences=False))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(32, activation='relu'))
        if self.num_classes == 1:
            model.add(layers.Dense(1, activation='sigmoid'))
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
        return model

class GRUModel:
    def __init__(self, input_shape, num_classes=2, dropout=0.3, num_layers=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_layers = num_layers

    def build(self):
        model = keras.Sequential()
        model.add(layers.Input(shape=self.input_shape))
        model.add(layers.GRU(64, return_sequences=False, dropout=self.dropout))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(32, activation='relu'))
        if self.num_classes == 1:
            model.add(layers.Dense(1, activation='sigmoid'))
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
        return model

class TransformerModel:
    def __init__(self, input_shape, d_model=64, num_heads=8, num_layers=2, num_classes=2, dropout=0.1):
        self.input_shape = input_shape
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

    def build(self):
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Dense(self.d_model)(inputs)
        sequence_length = self.input_shape[0]
        positions = tf.range(start=0, limit=sequence_length, delta=1)
        positions = tf.cast(positions, tf.float32)
        position_embedding = layers.Embedding(input_dim=sequence_length, output_dim=self.d_model)(positions)
        x = x + position_embedding
        for _ in range(self.num_layers):
            attn_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.d_model
            )(x, x)
            attn_output = layers.Dropout(self.dropout)(attn_output)
            x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
            ffn = keras.Sequential([
                layers.Dense(self.d_model * 4, activation="relu"),
                layers.Dense(self.d_model),
            ])
            ffn_output = ffn(x)
            ffn_output = layers.Dropout(self.dropout)(ffn_output)
            x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        if self.num_classes == 1:
            outputs = layers.Dense(1, activation="sigmoid")(x)
        else:
            outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        return model

class TinyTransformerModel:
    def __init__(self, input_shape, d_model=32, num_heads=4, num_layers=1, num_classes=2, dropout=0.1):
        self.input_shape = input_shape
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

    def build(self):
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Dense(self.d_model)(inputs)
        sequence_length = self.input_shape[0]
        positions = tf.range(start=0, limit=sequence_length, delta=1)
        positions = tf.cast(positions, tf.float32)
        position_embedding = layers.Embedding(input_dim=sequence_length, output_dim=self.d_model)(positions)
        x = x + position_embedding
        attn_output = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model
        )(x, x)
        attn_output = layers.Dropout(self.dropout)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        ffn_output = layers.Dense(64, activation="relu")(x)
        ffn_output = layers.Dense(self.d_model)(ffn_output)
        ffn_output = layers.Dropout(self.dropout)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(16, activation="relu")(x)
        if self.num_classes == 1:
            outputs = layers.Dense(1, activation="sigmoid")(x)
        else:
            outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        return model

class HybridNNModel:
    def __init__(self, input_shape, layer_types, num_classes=2, dropout=0.3):
        self.input_shape = input_shape
        self.layer_types = layer_types
        self.num_classes = num_classes
        self.dropout = dropout

    def build(self):
        inputs = layers.Input(shape=self.input_shape)
        x = inputs
        for i, layer_type in enumerate(self.layer_types):
            return_sequences = (i < len(self.layer_types) - 1)
            if layer_type == "RNN":
                x = layers.SimpleRNN(64, return_sequences=return_sequences)(x)
            elif layer_type == "GRU":
                x = layers.GRU(64, return_sequences=return_sequences)(x)
            elif layer_type == "LSTM":
                x = layers.LSTM(64, return_sequences=return_sequences)(x)
            elif layer_type == "Transformer":
                attn = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
                x = layers.LayerNormalization()(x + attn)
                if not return_sequences:
                    x = layers.GlobalAveragePooling1D()(x)
            elif layer_type == "TinyTransformer":
                attn = layers.MultiHeadAttention(num_heads=1, key_dim=32)(x, x)
                x = layers.LayerNormalization()(x + attn)
                if not return_sequences:
                    x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation="relu")(x)
        if self.num_classes == 1:
            outputs = layers.Dense(1, activation="sigmoid")(x)
        else:
            outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        return model

def get_model(model_type, input_shape, num_classes=2, num_layers=2):
    if isinstance(model_type, list):  # Hybrid model
        return HybridNNModel(input_shape, model_type, num_classes).build()
    elif model_type == "LSTM":
        return ImprovedLSTMModel(input_shape, num_classes, num_layers=num_layers).build()
    elif model_type == "RNN":
        return RNNModel(input_shape, num_classes, num_layers=num_layers).build()
    elif model_type == "GRU":
        return GRUModel(input_shape, num_classes, num_layers=num_layers).build()
    elif model_type == "Transformer":
        return TransformerModel(input_shape, num_layers=num_layers, num_classes=num_classes).build()
    elif model_type == "TinyTransformer":
        return TinyTransformerModel(input_shape, num_layers=num_layers, num_classes=num_classes).build()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_optimizer(name, lr=0.001):
    if name == "Adam":
        return optimizers.Adam(learning_rate=lr)
    elif name == "SGD":
        return optimizers.SGD(learning_rate=lr)
    elif name == "AdamW":
        return optimizers.AdamW(learning_rate=lr)
    else:
        raise ValueError("Optimizer not supported.")

def get_loss_function(loss_name, class_weights=None):
    if loss_name == "MSELoss":
        return losses.MeanSquaredError()
    elif loss_name == "SmoothL1Loss":
        return losses.Huber(delta=1.0)
    elif loss_name == "HuberLoss":
        return losses.Huber()
    elif loss_name == "CrossEntropyLoss":
        return losses.SparseCategoricalCrossentropy()
    elif loss_name == "BCEWithLogitsLoss":
        return losses.BinaryCrossentropy()
    else:
        raise ValueError("Loss not supported.")

# === DATA LOADING & PREPROCESSING ===

def load_and_preprocess_data(path, window_size=20, stride=5, selected_keys=None):
    X_all, y_all = [], []
    with h5py.File(path, 'r') as h5file:
        keys = list(h5file.keys())
        if any(k.startswith("trial_") for k in keys):
            for trial_name in keys:
                if not trial_name.startswith("trial_"):
                    continue
                trial = h5file[trial_name]
                available_keys = list(trial.keys())
                emg_keys = [k for k in available_keys if k.startswith("emg")]
                imu_keys = [k for k in available_keys if k.startswith("imu")]
                if selected_keys is not None:
                    emg_keys = [k for k in emg_keys if k in selected_keys]
                    imu_keys = [k for k in imu_keys if k in selected_keys]
                X = []
                for key in emg_keys:
                    signal = np.array(trial[key])[0]
                    std = signal.std()
                    if std < 1e-6:
                        continue
                    signal = (signal - signal.mean()) / (std + 1e-8)
                    X.append(signal)
                for key in imu_keys:
                    imu_data = np.array(trial[key])
                    for i in range(imu_data.shape[0]):
                        signal = imu_data[i]
                        std = signal.std()
                        if std < 1e-6:
                            continue
                        signal = (signal - signal.mean()) / (std + 1e-8)
                        X.append(signal)
                if len(X) == 0 or "button_ok" not in trial:
                    continue
                X = np.stack(X, axis=-1)
                y = np.array(trial["button_ok"])[0]
                X_windows, y_windows = create_sliding_windows(X, y, window_size, stride)
                X_all.append(X_windows)
                y_all.append(y_windows)
        elif "Sensor" in keys and "Controller" in keys:
            sensor = h5file["Sensor"]
            ctrl = h5file["Controller"]
            available_keys = list(sensor.keys())
            emg_keys = [k for k in available_keys if k.startswith("emg")]
            imu_keys = [k for k in available_keys if k.startswith("imu")]
            if selected_keys is not None:
                emg_keys = [k for k in emg_keys if k in selected_keys]
                imu_keys = [k for k in imu_keys if k in selected_keys]
            if not emg_keys and not imu_keys:
                raise ValueError("Aucun signal EMG ou IMU trouvé dans Sensor.")
            if "button_ok" not in ctrl:
                raise ValueError("Clé 'button_ok' manquante dans Controller.")
            X = []
            for key in emg_keys:
                signal = np.array(sensor[key])
                if signal.ndim > 1:
                    for i in range(signal.shape[0]):
                        ch = signal[i]
                        std = ch.std()
                        if std < 1e-6:
                            continue
                        ch = (ch - ch.mean()) / (std + 1e-8)
                        X.append(ch)
                else:
                    std = signal.std()
                    if std < 1e-6:
                        continue
                    signal = (signal - signal.mean()) / (std + 1e-8)
                    X.append(signal)
            for key in imu_keys:
                imu_data = np.array(sensor[key])
                if imu_data.ndim == 2:
                    for i in range(imu_data.shape[1]):
                        signal = imu_data[:, i]
                        std = signal.std()
                        if std < 1e-6:
                            continue
                        signal = (signal - signal.mean()) / (std + 1e-8)
                        X.append(signal)
            if len(X) == 0:
                raise ValueError("Aucun signal exploitable dans Sensor.")
            X = np.stack(X, axis=-1)
            y = np.array(ctrl["button_ok"])
            X_windows, y_windows = create_sliding_windows(X, y, window_size, stride)
            X_all.append(X_windows)
            y_all.append(y_windows)
        else:
            raise ValueError("Structure de fichier non reconnue (ni trial_*, ni Sensor/Controller).")
    if not X_all:
        raise ValueError("Aucune donnée exploitable trouvée dans le fichier.")
    X_final = np.concatenate(X_all)
    y_final = np.concatenate(y_all)
    return X_final, y_final

def create_sliding_windows(X, y, window_size, stride):
    X_windows = []
    y_windows = []
    for i in range(0, len(X) - window_size + 1, stride):
        window_x = X[i:i + window_size]
        window_y = y[i:i + window_size]
        label = 1 if np.any(window_y > 0.5) else 0
        X_windows.append(window_x)
        y_windows.append(label)
    return np.array(X_windows), np.array(y_windows)

def rebalance_classes(X, y):
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    if len(class_0) > len(class_1):
        class_1 = resample(class_1, replace=True, n_samples=len(class_0), random_state=42)
    else:
        class_0 = resample(class_0, replace=True, n_samples=len(class_1), random_state=42)
    X_bal = np.concatenate([class_0, class_1])
    y_bal = np.array([0] * len(class_0) + [1] * len(class_1))
    X_bal, y_bal = shuffle(X_bal, y_bal, random_state=42)
    return X_bal, y_bal

# === TRAINING FUNCTION ===

def train_model(X_train, X_val, X_test, y_train, y_val, y_test,
               model_type, optimizer_name, loss_name, epochs, learning_rate,
               batch_size=64, num_layers=2, sequence_length=None, verbose=1,
               stop_flag_getter=None,log_callback=None):
    from sklearn.metrics import accuracy_score, classification_report
    tf.random.set_seed(42)
    np.random.seed(42)
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))
    if loss_name in ["BCEWithLogitsLoss", "MSELoss", "SmoothL1Loss", "HuberLoss"]:
        num_classes = 1
    model = get_model(model_type, input_shape, num_classes, num_layers=num_layers)
    optimizer = get_optimizer(optimizer_name, learning_rate)
    class_counts = Counter(y_train)
    total = sum(class_counts.values())
    class_weights = {i: total / class_counts[i] for i in class_counts}
    loss_fn = get_loss_function(loss_name, class_weights)
    metrics = ['accuracy'] if num_classes == 1 else ['sparse_categorical_accuracy']
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    y_train_ = y_train.astype(np.float32) if num_classes == 1 else y_train
    y_val_ = y_val.astype(np.float32) if num_classes == 1 else y_val
    y_test_ = y_test.astype(np.float32) if num_classes == 1 else y_test
    # 🛑 Check for stop signal before launching training
    if stop_flag_getter is not None and stop_flag_getter():
        print("🛑 Training stopped by user.")
        return model, {}, {
            'accuracy': 0.0,
            'report': 'Training was stopped.',
            'predictions': [],
            'true_labels': []
        }, None
    # Ajoute juste avant le fit :
    logging_cb = LoggingCallback(log_callback, epochs)
    stop_cb = StopTrainingCallback(stop_flag_getter)

    # Entraînement du modèle
    history = model.fit(
        X_train, y_train_,
        validation_data=(X_val, y_val_),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[reduce_lr, early_stop, logging_cb, stop_cb],
        verbose=verbose,
        class_weight=class_weights if num_classes > 1 else None
    )                         
    test_predictions = model.predict(X_test)
    if num_classes == 1:
        test_preds = (test_predictions > 0.5).astype(int).flatten()
    else:
        test_preds = np.argmax(test_predictions, axis=1)
    test_acc = accuracy_score(y_test, test_preds)
    test_report = classification_report(y_test, test_preds, zero_division=0)
    test_results = {
        'accuracy': test_acc,
        'predictions': test_preds,
        'true_labels': y_test,
        'report': test_report
    }
    history_dict = {
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'train_acc': history.history.get('accuracy', history.history.get('sparse_categorical_accuracy', [])),
        'val_acc': history.history.get('val_accuracy', history.history.get('val_sparse_categorical_accuracy', [])),
        'lr': [reduce_lr.get_lr() if hasattr(reduce_lr, 'get_lr') else learning_rate] * len(history.history['loss'])
    }
    return model, history_dict, test_results, optimizer

# === UTILS FOR SAVING/LOADING MODELS ===

def save_model_and_results(model, history, test_results, file_path):
    import json
    import numpy as np
    if not file_path.endswith(".h5"):
        file_path += ".h5"
    model.save(file_path)
    base = file_path[:-3]
    def make_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj
    with open(base + "_history.json", "w") as f:
        json.dump(history, f, default=make_json_serializable, indent=2)
    with open(base + "_results.json", "w") as f:
        json.dump(test_results, f, default=make_json_serializable, indent=2)

def load_model_and_results(file_path):
    import json
    import numpy as np
    model = keras.models.load_model(file_path)
    base = file_path[:-3]
    hist_path = base + "_history.json"
    res_path = base + "_results.json"
    history = None
    test_results = None
    if os.path.exists(hist_path) and os.path.exists(res_path):
        with open(hist_path, "r") as f:
            history = json.load(f)
        with open(res_path, "r") as f:
            test_results = json.load(f)
        for key in ['true_labels', 'predictions']:
            if key in test_results:
                raw = test_results[key]
                if isinstance(raw, str):
                    import ast
                    raw = ast.literal_eval(raw)
                arr = np.asarray(raw)
                if arr.ndim == 1:
                    test_results[key] = arr
                elif arr.ndim == 2 and arr.shape[1] == 1:
                    test_results[key] = arr.flatten()
                else:
                    test_results[key] = arr
    return model, history, test_results

# === END OF MODULE ===

class StopTrainingCallback(Callback):
    def __init__(self, stop_flag_getter):
        super().__init__()
        self.stop_flag_getter = stop_flag_getter

    def on_epoch_end(self, epoch, logs=None):
        if self.stop_flag_getter and self.stop_flag_getter():
            print("🛑 Stopping training early via StopTrainingCallback.")
            self.model.stop_training = True

    