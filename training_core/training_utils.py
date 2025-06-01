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
from PyQt5.QtCore import QThread, pyqtSignal

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

def build_flexible_model(layer_configs, input_shape, num_classes=2):
    from tensorflow.keras import layers, Model

    inputs = layers.Input(shape=input_shape)
    x = inputs

    for i, cfg in enumerate(layer_configs):
        ltype = cfg.get("type", "LSTM")
        return_seq = i < len(layer_configs) - 1
        dropout = float(cfg.get("dropout", 0.0))
        activation = cfg.get("activation", "tanh")

        if ltype in ["LSTM", "GRU", "RNN"]:
            units = int(cfg.get("units", 64))
            bidirectional = str(cfg.get("bidirectional", "False")).lower() == "true"

            rnn_layer = None
            if ltype == "LSTM":
                rnn_layer = layers.LSTM(units, return_sequences=return_seq, dropout=dropout, activation=activation)
            elif ltype == "GRU":
                rnn_layer = layers.GRU(units, return_sequences=return_seq, dropout=dropout, activation=activation)
            elif ltype == "RNN":
                rnn_layer = layers.SimpleRNN(units, return_sequences=return_seq, dropout=dropout, activation=activation)

            if bidirectional:
                x = layers.Bidirectional(rnn_layer)(x)
            else:
                x = rnn_layer(x)

        elif ltype in ["Transformer", "TinyTransformer"]:
            d_model = int(cfg.get("d_model", 64))
            num_heads = int(cfg.get("num_heads", 2))
            attention_dropout = float(cfg.get("attention_dropout", 0.1))

            # Project input to d_model
            x = layers.Dense(d_model)(x)

            # Multi-head attention block
            attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads,
                                             dropout=attention_dropout)(x, x)
            x = layers.LayerNormalization()(x + attn)

            # Feed-forward block
            ff = layers.Dense(d_model * 4, activation=activation)(x)
            ff = layers.Dense(d_model)(ff)
            x = layers.LayerNormalization()(x + ff)

            if not return_seq:
                x = layers.GlobalAveragePooling1D()(x)

    # Final pooling if necessary
    if len(x.shape) == 3:  # Replace x.shape.rank with len(x.shape)
        x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    if num_classes == 1:
        output = layers.Dense(1, activation="sigmoid")(x)
    else:
        output = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, output)


def get_model(model_type, input_shape, num_classes=2, num_layers=2):
    # Nouveau système dynamique
    if isinstance(model_type, list) and isinstance(model_type[0], dict):
        return build_flexible_model(model_type, input_shape, num_classes)
    
    # Système simple mono-type (optionnel, legacy)
    elif model_type == "LSTM":
        return build_flexible_model(
            [{"type": "LSTM", "units": 64, "dropout": 0.3} for _ in range(num_layers)],
            input_shape, num_classes
        )
    elif model_type == "GRU":
        return build_flexible_model(
            [{"type": "GRU", "units": 64, "dropout": 0.3} for _ in range(num_layers)],
            input_shape, num_classes
        )
    elif model_type == "RNN":
        return build_flexible_model(
            [{"type": "RNN", "units": 64, "dropout": 0.3} for _ in range(num_layers)],
            input_shape, num_classes
        )
    elif model_type == "Transformer":
        return build_flexible_model(
            [{"type": "Transformer", "d_model": 64, "num_heads": 8, "dropout": 0.1} for _ in range(num_layers)],
            input_shape, num_classes
        )
    elif model_type == "TinyTransformer":
        return build_flexible_model(
            [{"type": "TinyTransformer", "d_model": 32, "num_heads": 2, "dropout": 0.1} for _ in range(num_layers)],
            input_shape, num_classes
        )
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

def load_and_preprocess_data(path, window_size=10, stride=3, selected_keys=None):
    X_all, y_all, all_time= [], [], []
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
                y = np.array(trial["label_int"])
                
                X_windows, y_windows = create_sliding_windows(X, y, window_size, stride)
                # Filtrer et équilibrer les classes
                y_unique, y_counts = np.unique(y_windows, return_counts=True)
                min_count = min(y_counts)

                # Sélection aléatoire pour équilibrer les classes
                balanced_indices = []
                for label in y_unique:
                    indices = np.where(y_windows == label)[0]
                    selected_indices = np.random.choice(indices, min_count, replace=False)
                    balanced_indices.extend(selected_indices)

                balanced_indices = np.array(balanced_indices)

                # Mettre à jour X_windows et y_windows
                X_windows = X_windows[balanced_indices]
                y_windows = y_windows[balanced_indices]

                # Vérifier l'équilibre des classes
                print(f"Classes après équilibrage : {dict(zip(*np.unique(y_windows, return_counts=True)))}")

                print("Nombre de fenêtres générées :", len(X_windows))
                X_all.append(X_windows)
                y_all.append(y_windows)
                if "time" in trial:
                    try:
                        time_data = np.array(trial["time"])
                        print(f"🧪 time_data pour {trial_name} : {time_data.shape} → valeurs : {time_data[0, :10] if time_data.ndim == 2 else time_data[:10]}")
                        print(f"📁 {trial_name} → time range: [{time_data.min()} - {time_data.max()}]")
                        if time_data.ndim == 2 and time_data.shape[0] == 1:
                            time_data = time_data[0]  # (1, N) → (N,)
                        elif time_data.ndim == 1:
                            pass
                        else:
                            raise ValueError("Format non supporté pour time")

                        if time_data.shape[0] != y.shape[0]:
                            raise ValueError(f"Time vector length mismatch with y: {time_data.shape[0]} vs {y.shape[0]}")

                        if time_data.shape[0] != y.shape[0]:
                            print(f"⚠️ Incohérence entre time_data ({time_data.shape[0]}) et y ({y.shape[0]}) dans {trial_name}")
                            min_len = min(time_data.shape[0], y.shape[0])
                            time_data = time_data[:min_len]
                            y = y[:min_len]
                            
                        _, time_windows = create_sliding_windows(time_data, y, window_size, stride, raw_labels=True)
                        for i, tw in enumerate(time_windows[:5]):
                            print(f"🪟 Fenêtre {i} → tw = {tw[-5:]} → dernier = {tw[-1]}")

                        time_vector = np.array([tw[0] for tw in time_windows])

                        if len(time_vector) != len(y_windows):
                            print(f"⚠️ {trial_name} : désalignement entre time ({len(time_vector)}) et y ({len(y_windows)}) → time tronqué")
                            min_len = min(len(time_vector), len(y_windows))
                            all_time.append(time_vector[:min_len])
                            X_all[-1] = X_all[-1][:min_len]
                            y_all[-1] = y_all[-1][:min_len]
                        else:
                            all_time.append(time_vector)

                    except Exception as e:
                        print(f"⚠️ Erreur lors du traitement des timestamps dans {trial_name} → vecteur temps ignoré : {e}")
                else:
                    print(f"ℹ️ Aucun champ 'time' dans {trial_name} → ignoré")
        # === DÉBUT DU BLOC SENSOR/CONTROLLER À COMMENTER ===
        # elif "Sensor" in keys and "Controller" in keys:
        #     sensor = h5file["Sensor"]
        #     ctrl = h5file["Controller"]
        #     button_ok = np.array(ctrl["button_ok"])
        #     time_data = np.array(sensor["time"])
        #     press_indices = np.where(np.diff(button_ok.astype(int)) == 1)[0] + 1
        #
        #     num_trials = len(press_indices) // 4
        #     for trial_idx in range(num_trials):
        #         trial_press = press_indices[trial_idx*4:(trial_idx+1)*4]
        #         if len(trial_press) < 4:
        #             continue  # trial incomplet
        #
        #         # Définir les bornes du trial (par exemple, de la première à la dernière pression + un peu avant/après)
        #         start = trial_press[0] - 500 if trial_press[0] - 500 > 0 else 0
        #         end = trial_press[3] + 1000 if trial_press[3] + 1000 < len(button_ok) else len(button_ok)
        #
        #         # Extraire les signaux pour ce trial
        #         trial_sensor = {k: np.array(sensor[k])[start:end] for k in emg_keys + imu_keys + ["time"]}
        #         trial_button_ok = button_ok[start:end]
        #
        #         # Générer les labels comme pour cropped
        #         temp = {
        #             "button_ok": trial_button_ok,
        #             "time": trial_sensor["time"]
        #         }
        #         try:
        #             from functions import generate_fixed_sequence_labels, convert_labels_to_int, phase_to_int
        #             temp["label"] = generate_fixed_sequence_labels(temp, fs=100)
        #             temp = convert_labels_to_int(temp, phase_to_int)
        #             y = np.array(temp["label_int"])
        #         except Exception as e:
        #             print(f"⚠️ Erreur génération labels trial {trial_idx} : {e}")
        #             continue
        #
        #         # Stack les signaux comme d'habitude
        #         X = []
        #         for key in emg_keys:
        #             signal = trial_sensor[key]
        #             std = signal.std()
        #             if std < 1e-6:
        #                 continue
        #             signal = (signal - signal.mean()) / (std + 1e-8)
        #             X.append(signal)
        #         for key in imu_keys:
        #             imu_data = trial_sensor[key]
        #             if imu_data.ndim == 2:
        #                 for i in range(imu_data.shape[1]):
        #                     signal = imu_data[:, i]
        #                     std = signal.std()
        #                     if std < 1e-6:
        #                         continue
        #                     signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        #                     X.append(signal)
        #             else:
        #                 std = imu_data.std()
        #                 if std < 1e-6:
        #                     continue
        #                 imu_data = (imu_data - imu_data.mean()) / (imu_data.std() + 1e-8)
        #                 X.append(imu_data)
        #         if len(X) == 0:
        #             continue
        #         X = np.stack(X, axis=-1)
        #
        #         # Sliding windows etc. comme pour cropped
        #         X_windows, y_windows = create_sliding_windows(X, y, window_size, stride)
        #         # Filtrer et équilibrer les classes
        #         y_unique, y_counts = np.unique(y_windows, return_counts=True)
        #         min_count = min(y_counts)
        #
        #         # Sélection aléatoire pour équilibrer les classes
        #         balanced_indices = []
        #         for label in y_unique:
        #             indices = np.where(y_windows == label)[0]
        #             selected_indices = np.random.choice(indices, min_count, replace=False)
        #             balanced_indices.extend(selected_indices)
        #
        #         balanced_indices = np.array(balanced_indices)
        #
        #         # Mettre à jour X_windows et y_windows
        #         X_windows = X_windows[balanced_indices]
        #         y_windows = y_windows[balanced_indices]
        #
        #         # Vérifier l'équilibre des classes
        #         print(f"Classes après équilibrage : {dict(zip(*np.unique(y_windows, return_counts=True)))}")
        #
        #         X_all.append(X_windows)
        #         y_all.append(y_windows)
        #
        #         if "time" in sensor:
        #             time_data = np.array(sensor["time"])
        #             print(f"🧪 time_data pour {os.path.basename(path)} : {time_data.shape} → valeurs : {time_data[:10]}")
        #             print(f"📁 {os.path.basename(path)} → time range: [{time_data.min()} - {time_data.max()}]")
        #
        #             if time_data.ndim == 2 and time_data.shape[0] == 1:
        #                 time_data = time_data[0]  # (1, N) → (N,)
        #             elif time_data.ndim != 1:
        #                 raise ValueError("Format non supporté pour time")
        #
        #             if time_data.shape[0] != y.shape[0]:
        #                 print(f"⚠️ Incohérence entre time_data ({time_data.shape[0]}) et y ({y.shape[0]}) dans {os.path.basename(path)}")
        #                 min_len = min(time_data.shape[0], y.shape[0])
        #                 time_data = time_data[:min_len]
        #                 y = y[:min_len]
        #
        #             _, time_windows = create_sliding_windows(time_data, y, window_size, stride, raw_labels=True)
        #             for i, tw in enumerate(time_windows[:5]):
        #                 print(f"🪟 Fenêtre {i} → tw = {tw[-5:]} → dernier = {tw[-1]}")
        #
        #             time_vector = np.array([tw[0] for tw in time_windows])
        #
        #             # 🔍 Ajout à time_origin_map (important)
        #             if not hasattr(load_and_preprocess_data, "time_origin_map"):
        #                 load_and_preprocess_data.time_origin_map = {}
        #             origin_map = load_and_preprocess_data.time_origin_map
        #             for t in time_vector:
        #                 origin_map[int(t)] = os.path.basename(path)
        #
        #             if len(time_vector) != len(y_windows):
        #                 print(f"⚠️ {os.path.basename(path)} : désalignement entre time ({len(time_vector)}) et y ({len(y_windows)}) → time tronqué")
        #                 min_len = min(len(time_vector), len(y_windows))
        #                 all_time.append(time_vector[:min_len])
        #                 X_all[-1] = X_all[-1][:min_len]
        #                 y_all[-1] = y_all[-1][:min_len]
        #             else:
        #                 all_time.append(time_vector)
        #         else:
        #             print(f"ℹ️ Aucun champ 'time' dans Sensor → génération synthétique")
        # === FIN DU BLOC SENSOR/CONTROLLER À COMMENTER ===
        else:
           raise ValueError("Structure de fichier non reconnue (ni trial_*, ni Sensor/Controller).")
    
    # Modified time handling section
    X_all = [x for x in X_all if x.ndim >= 2 and x.shape[0] > 0]
    y_all = [y for y in y_all if y.ndim >= 1 and y.shape[0] > 0]
    all_time = [t for t in all_time if np.ndim(t) > 0]

    if not X_all:
        raise ValueError("Aucune donnée exploitable trouvée dans le fichier.")

    X_final = np.concatenate(X_all)
    y_final = np.concatenate(y_all)
    
    # Enhanced time handling with better feedback
    # Cas 1 : all_time est une liste de scalaires
    if all_time and np.ndim(all_time[0]) == 0:
        time_final = np.array(all_time).squeeze()
        print(f"✅ Timestamps collectés (scalaires) : {len(time_final)}")
    # Cas 2 : all_time est une liste de vecteurs
    elif all_time and np.ndim(all_time[0]) >= 1:
        try:
            time_final = np.concatenate(all_time).squeeze()
            print(f"✅ Timestamps collectés (concaténés) : {len(time_final)}")
        except Exception as e:
            print(f"⚠️ Erreur concaténation : {e}")
            time_final = np.arange(X_final.shape[0]) * 0.01
    # Cas vide
    else:
        print(f"ℹ️ Aucun timestamp collecté depuis {path} → génération synthétique")
        time_final = np.arange(X_final.shape[0]) * 0.01

    # Vérification finale de l'alignement
    assert time_final.size == X_final.shape[0], f"Désalignement : {time_final.size} timestamps vs {X_final.shape[0]} échantillons"  

    min_len = min(len(X_final), len(y_final), len(time_final))
    X_final = X_final[:min_len]
    y_final = y_final[:min_len]
    time_final = time_final[:min_len]

    print(f"✅ Chargé : {path} → X={X_final.shape}, y={y_final.shape}, time={time_final.shape}")
    print(f"🕒 Extrait time (10 premiers) : {time_final[:10]}")
    # 🔍 Construction du dictionnaire d'origine des timestamps
    if not hasattr(load_and_preprocess_data, "time_origin_map"):
        load_and_preprocess_data.time_origin_map = {}

    origin_map = load_and_preprocess_data.time_origin_map

    with h5py.File(path, 'r') as h5file:
        for trial_name in h5file.keys():
            if not trial_name.startswith("trial_"):
                continue
            trial = h5file[trial_name]
            if "time" not in trial:
                continue
            time_data = trial["time"][:][0] if trial["time"].ndim == 2 else trial["time"][:]
            for t in time_data:
                origin_map[int(t)] = trial_name

    # Réassignation globale finale (fusion complète)
    load_and_preprocess_data.time_origin_map = origin_map
    return X_final, y_final, time_final

def create_sliding_windows(X, y, window_size, stride, raw_labels=False):
    """Create sliding windows from the input data.
    
    Args:
        X: Input features array
        y: Labels array 
        window_size: Size of each window
        stride: Step size between windows
        raw_labels: If True, use raw labels instead of processing them
        
    Returns:
        X_windows: Array of windowed features
        y_windows: Array of labels for each window
    """
    X_windows = []
    y_windows = []

    if raw_labels and y.ndim > 1:
        y = y.squeeze()

    for i in range(0, len(X) - window_size + 1, stride):
        window_x = X[i:i + window_size]
        window_y = y[i:i + window_size]

        if raw_labels:
            label = window_x  # Pour les timestamps
        else:
            # Prendre l'étiquette du milieu de la fenêtre comme label
            label = window_y[window_size // 2]

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
               stop_flag_getter=None, log_callback=None, test_time=None):
    from sklearn.metrics import accuracy_score, classification_report
    tf.random.set_seed(42)
    np.random.seed(42)
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))
    if loss_name in ["BCEWithLogitsLoss", "MSELoss", "SmoothL1Loss", "HuberLoss"]:
        num_classes = 1

    # Force multi-class classification if we detect multiple classes
    num_unique_classes = len(np.unique(y_train))
    if num_unique_classes > 2:
        print(f"✅ Detected {num_unique_classes} classes - forcing multi-class mode")
        num_classes = num_unique_classes
        metrics = ['sparse_categorical_accuracy']
        loss_fn = losses.SparseCategoricalCrossentropy()
        
        # Calculate class weights properly
        class_counts = Counter(y_train)
        total_samples = sum(class_counts.values())
        class_weights = {i: total_samples / (len(class_counts) * count) for i, count in class_counts.items()}
        
        print("📊 Class distribution:", dict(class_counts))
        print("⚖️ Class weights:", class_weights)
    else:
        # Binary classification case
        num_classes = 1 if loss_name in ["BCEWithLogitsLoss"] else 2
        metrics = ['accuracy']
        loss_fn = get_loss_function(loss_name)
        class_weights = None

    model = get_model(model_type, input_shape, num_classes, num_layers=num_layers)
    optimizer = get_optimizer(optimizer_name, learning_rate)
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
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

    print("✅ Entraînement sur le modèle suivant :")
    model.summary()

    print("✅ Couches utilisées :")
    for idx, layer in enumerate(model.layers):
        config = getattr(layer, 'get_config', lambda: {})()
        print(f"Layer {idx + 1}: {layer.__class__.__name__} - Config: {config}")

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
        # For multi-class, always use argmax
        test_preds = np.argmax(test_predictions, axis=1)

    test_acc = accuracy_score(y_test, test_preds)
    test_report = classification_report(y_test, test_preds, zero_division=0)
    
    print("🔍 Classes uniques dans y_test:", np.unique(y_test))
    print("🔍 Classes uniques dans predictions:", np.unique(test_preds))

    test_results = {
        'accuracy': test_acc,
        'predictions': test_preds.tolist(),
        'true_labels': y_test.tolist(),
        'report': test_report
    }

    # Add time data if provided
    if test_time is not None:
        test_results['time'] = test_time.tolist() if hasattr(test_time, 'tolist') else test_time
        print("✅ Vecteur temps injecté:", test_results['time'])
    else:
        print("ℹ️ Aucun temps trouvé dans X_test.")

    print("🧪 X_test shape:", X_test.shape)
    print("🧪 test_time shape:", test_time.shape)

    # 🔎 Vérifie l'origine des timestamps de test
    origin_map = getattr(load_and_preprocess_data, "time_origin_map", {})
    for t in test_time[:20]:  # ou plus si tu veux
        print(f"🔎 {t} vient de {origin_map.get(int(t), '❓ inconnu')}")

    return model, history, test_results, optimizer


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

class TrainingThread(QThread):
    training_finished = pyqtSignal(object, object, object)
    training_log = pyqtSignal(str)

    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, params, test_time=None, parent=None):
        super().__init__(parent)
        # ...existing code...
        self._stop_flag = False

    def run(self):
        try:
            # HARRY: Check stop flag before starting
            if self._stop_flag:
                return
            
            # HARRY: Periodically check stop flag during training
            if self._stop_flag:
                self.training_log.emit("Training stopped by user")
                return
                
        except Exception as e:
            self.training_log.emit(f"Error during training: {str(e)}")
        finally:
            # HARRY: Ensure cleanup happens even if training fails
            if self._stop_flag:
                self.training_log.emit("Training cleanup completed")

    def stop(self):
        """HARRY: Enhanced stop mechanism"""
        self._stop_flag = True
        self.wait(1000)  # Wait up to 1 second for natural completion

