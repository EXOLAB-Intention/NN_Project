# requirements: pyqt5, tensorflow, numpy, h5py, matplotlib, scikit-learn
import sys
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, callbacks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample, shuffle
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, 
    QWidget, QLabel, QComboBox, QSpinBox, QHBoxLayout, QTextEdit, QProgressBar,
    QGroupBox, QGridLayout, QScrollArea, QMessageBox, QListWidget, QLineEdit, QListWidgetItem
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import math
import os
import random
from collections import Counter

# Configuration GPU pour TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class ImprovedLSTMModel:
    def __init__(self, input_shape, num_classes=2, dropout=0.3, num_layers=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_layers = num_layers
        
    def build(self):
        model = keras.Sequential()
        model.add(layers.Input(shape=self.input_shape))
        
        # Premier LSTM
        model.add(layers.LSTM(64, return_sequences=True, dropout=self.dropout if self.num_layers > 1 else 0))
        
        # Deuxième LSTM
        model.add(layers.LSTM(32, return_sequences=False))
        model.add(layers.Dropout(self.dropout))
        
        # Couches denses
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
        
        # Premier RNN
        model.add(layers.SimpleRNN(64, return_sequences=True, dropout=self.dropout if self.num_layers > 1 else 0))
        
        # Deuxième RNN
        model.add(layers.SimpleRNN(32, return_sequences=False))
        model.add(layers.Dropout(self.dropout))
        
        # Couches denses
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
        
        # GRU layers
        model.add(layers.GRU(64, return_sequences=False, dropout=self.dropout))
        model.add(layers.Dropout(self.dropout))
        
        # Couches denses  
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
        
        # Projection d'entrée
        x = layers.Dense(self.d_model)(inputs)
        
        # Encodage positionnel
        sequence_length = self.input_shape[0]
        positions = tf.range(start=0, limit=sequence_length, delta=1)
        positions = tf.cast(positions, tf.float32)
        
        position_embedding = layers.Embedding(input_dim=sequence_length, output_dim=self.d_model)(positions)
        x = x + position_embedding
        
        # Couches Transformer
        for _ in range(self.num_layers):
            # Multi-head attention
            attn_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.d_model
            )(x, x)
            attn_output = layers.Dropout(self.dropout)(attn_output)
            x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # Feed forward
            ffn = keras.Sequential([
                layers.Dense(self.d_model * 4, activation="relu"),
                layers.Dense(self.d_model),
            ])
            ffn_output = ffn(x)
            ffn_output = layers.Dropout(self.dropout)(ffn_output)
            x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification head
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
        
        # Projection d'entrée
        x = layers.Dense(self.d_model)(inputs)
        
        # Encodage positionnel simple
        sequence_length = self.input_shape[0]
        positions = tf.range(start=0, limit=sequence_length, delta=1)
        positions = tf.cast(positions, tf.float32)
        
        position_embedding = layers.Embedding(input_dim=sequence_length, output_dim=self.d_model)(positions)
        x = x + position_embedding
        
        # Couche Transformer
        attn_output = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model
        )(x, x)
        attn_output = layers.Dropout(self.dropout)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed forward simple
        ffn_output = layers.Dense(64, activation="relu")(x)
        ffn_output = layers.Dense(self.d_model)(ffn_output)
        ffn_output = layers.Dropout(self.dropout)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification head simple
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
                # Transformer simple
                attn = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
                x = layers.LayerNormalization()(x + attn)
                if not return_sequences:
                    x = layers.GlobalAveragePooling1D()(x)
            elif layer_type == "TinyTransformer":
                # Tiny Transformer
                attn = layers.MultiHeadAttention(num_heads=1, key_dim=32)(x, x)
                x = layers.LayerNormalization()(x + attn)
                if not return_sequences:
                    x = layers.GlobalAveragePooling1D()(x)
        
        # Classification head
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
    """Factory function to create optimizers"""
    if name == "Adam":
        return optimizers.Adam(learning_rate=lr)
    elif name == "SGD":
        return optimizers.SGD(learning_rate=lr)
    elif name == "AdamW":
        return optimizers.AdamW(learning_rate=lr)
    else:
        raise ValueError("Optimizer not supported.")

def get_loss_function(loss_name, class_weights=None):
    """Factory function to create loss functions"""
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

class FocalLoss(losses.Loss):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def call(self, y_true, y_pred):
        ce_loss = losses.sparse_categorical_crossentropy(y_true, y_pred)
        pt = tf.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return tf.reduce_mean(focal_loss)

class LabelSmoothingLoss(losses.Loss):
    def __init__(self, num_classes=2, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def call(self, y_true, y_pred):
        y_pred = tf.nn.log_softmax(y_pred, axis=-1)
        true_dist = tf.one_hot(y_true, self.num_classes)
        true_dist = true_dist * self.confidence + (1 - true_dist) * self.smoothing / (self.num_classes - 1)
        return tf.reduce_mean(tf.reduce_sum(-true_dist * y_pred, axis=-1))

def load_and_preprocess_data(path, window_size=20, stride=5, selected_keys=None):
    """Charge un fichier HDF5 (format trial_* ou Sensor/Controller) avec affichage des signaux."""
    X_all, y_all = [], []

    print(f"\n📂 Chargement : {os.path.basename(path)}")

    with h5py.File(path, 'r') as h5file:
        keys = list(h5file.keys())

        # === CAS 1 : fichiers avec trial_1, trial_2, ...
        if any(k.startswith("trial_") for k in keys):
            print("📁 Format : trial_*")
            for trial_name in keys:
                if not trial_name.startswith("trial_"):
                    continue

                trial = h5file[trial_name]
                print(f"  🧪 {trial_name} :")
                available_keys = list(trial.keys())
                emg_keys = [k for k in available_keys if k.startswith("emg")]
                imu_keys = [k for k in available_keys if k.startswith("imu")]

                if selected_keys is not None:
                    emg_keys = [k for k in emg_keys if k in selected_keys]
                    imu_keys = [k for k in imu_keys if k in selected_keys]

                X = []

                for key in emg_keys:
                    signal = np.array(trial[key])[0]
                    print(f"    ➤ {key} | shape: {signal.shape}")
                    print(f"      Extrait : {signal[:10]}")
                    std = signal.std()
                    if std < 1e-6:
                        continue
                    signal = (signal - signal.mean()) / (std + 1e-8)
                    X.append(signal)

                for key in imu_keys:
                    imu_data = np.array(trial[key])
                    for i in range(imu_data.shape[0]):
                        signal = imu_data[i]
                        print(f"    ➤ {key}[{i}] | shape: {signal.shape}")
                        print(f"      Extrait : {signal[:10]}")
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

        # === CAS 2 : Sensor/Controller
        elif "Sensor" in keys and "Controller" in keys:
            print("📁 Format : Sensor/Controller")
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

            print("  🧪 Sensor :")
            X = []

            for key in emg_keys:
                signal = np.array(sensor[key])
                print(f"    ➤ {key} | shape: {signal.shape}")
                print(f"      Extrait : {signal[:10]}")
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
                print(f"    ➤ {key} | shape: {imu_data.shape}")
                print(f"      Extrait : {imu_data[:10]}")
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

    print(f"✅ Données extraites : X.shape = {X_final.shape}, y.shape = {y_final.shape}")
    print(f"   ➤ Extrait y : {y_final[:10]}")

    return X_final, y_final

def create_sliding_windows(X, y, window_size, stride):
    """Create sliding windows for training"""
    X_windows = []
    y_windows = []
    
    for i in range(0, len(X) - window_size + 1, stride):
        window_x = X[i:i + window_size]
        window_y = y[i:i + window_size]
        
        # Label: 1 if any button press in window, 0 otherwise
        label = 1 if np.any(window_y > 0.5) else 0
        
        X_windows.append(window_x)
        y_windows.append(label)
    
    return np.array(X_windows), np.array(y_windows)

class CustomCallback(callbacks.Callback):
    def __init__(self, progress_signal):
        super().__init__()
        self.progress_signal = progress_signal
        
    def on_epoch_end(self, epoch, logs=None):
        # Émettre le signal de progression
        if self.progress_signal:
            progress = int((epoch + 1) / self.params['epochs'] * 100)
            self.progress_signal.emit(progress)

class TrainingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object, object, object)

    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test,
                 model_type, optimizer_name, loss_name, epochs, learning_rate,
                 batch_size=64, num_layers=2, sequence_length=None):
        super().__init__()
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.model_type = model_type
        self.optimizer_name = optimizer_name
        self.loss_name = loss_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length

    def run(self):
        try:
            # Set seeds for reproducibility
            tf.random.set_seed(42)
            np.random.seed(42)
            
            input_shape = (self.X_train.shape[1], self.X_train.shape[2])
            num_classes = len(np.unique(self.y_train))

            if self.loss_name in ["BCEWithLogitsLoss", "MSELoss", "SmoothL1Loss", "HuberLoss"]:
                num_classes = 1

            # Create model
            model = get_model(self.model_type, input_shape, num_classes, num_layers=self.num_layers)
            
            # Get optimizer and loss
            optimizer = get_optimizer(self.optimizer_name, self.learning_rate)
            
            # Calculate class weights
            class_counts = Counter(self.y_train)
            total = sum(class_counts.values())
            class_weights = {i: total / class_counts[i] for i in class_counts}
            
            loss_fn = get_loss_function(self.loss_name, class_weights)
            
            # Compile model
            if num_classes == 1:
                metrics = ['accuracy']
            else:
                metrics = ['sparse_categorical_accuracy']
                
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
            
            # Callbacks
            reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)
            early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            custom_callback = CustomCallback(self.progress)
            
            # Prepare data
            y_train = self.y_train
            y_val = self.y_val
            y_test = self.y_test
            
            if self.loss_name in ["BCEWithLogitsLoss", "MSELoss", "SmoothL1Loss", "HuberLoss"]:
                y_train = y_train.astype(np.float32)
                y_val = y_val.astype(np.float32)
                y_test = y_test.astype(np.float32)
            
            # Training
            history = model.fit(
                self.X_train, y_train,
                validation_data=(self.X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[reduce_lr, early_stop, custom_callback],
                verbose=1,
                class_weight=class_weights if num_classes > 1 else None
            )
            
            # Test evaluation
            test_predictions = model.predict(self.X_test)
            
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
            
            # Convert history to dict format compatible with plotting
            history_dict = {
                'train_loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'train_acc': history.history.get('accuracy', history.history.get('sparse_categorical_accuracy', [])),
                'val_acc': history.history.get('val_accuracy', history.history.get('val_sparse_categorical_accuracy', [])),
                'lr': [reduce_lr.get_lr() if hasattr(reduce_lr, 'get_lr') else self.learning_rate] * len(history.history['loss'])
            }
            
            self.finished.emit(model, history_dict, test_results, optimizer)
            
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()

class EMGIMUGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced EMG/IMU Button Classifier - Keras Version")
        self.setGeometry(100, 100, 1400, 900)
        
        # Variables
        self.X = None
        self.y = None
        self.model = None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.scaler = StandardScaler()
        self.history = None 
        self.test_results = None
        self.setup_ui()
        
    def setup_ui(self):
        # Boutons principaux
        self.layer_type_combos = []
        self.load_button = QPushButton("Load HDF5 File")
        self.train_button = QPushButton("Train Model")
        self.train_button.setEnabled(False)

        # Liste des fichiers
        layout = QVBoxLayout()
        self.file_list = QListWidget()
        layout.addWidget(QLabel("📁 Fichiers disponibles :"))
        layout.addWidget(self.file_list)

        # Champ de pourcentage + bouton de sélection auto
        hbox_auto = QHBoxLayout()
        self.training_percentage_input = QLineEdit("80")
        self.training_percentage_input.setMaximumWidth(60)
        hbox_auto.addWidget(QLabel("Training %:"))
        hbox_auto.addWidget(self.training_percentage_input)

        self.auto_select_button = QPushButton("🎯 Auto Select Files")
        self.auto_select_button.clicked.connect(self.auto_select_files)
        hbox_auto.addWidget(self.auto_select_button)

        layout.addLayout(hbox_auto)

        self.save_button = QPushButton("💾 Save Model")
        self.save_button.setEnabled(False) 
        self.save_button.clicked.connect(self.save_model)

        self.load_model_button = QPushButton("📂 Load Model")
        self.load_model_button.setEnabled(True)
        self.load_model_button.clicked.connect(self.load_model)

        # Groupe de sélection du modèle
        model_group = QGroupBox("Model Configuration")
        model_layout = QGridLayout()

        model_layout.addWidget(QLabel("Optimizer:"), 1, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "AdamW"])
        model_layout.addWidget(self.optimizer_combo, 1, 1)

        model_layout.addWidget(QLabel("Loss Function:"), 2, 0)
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["CrossEntropyLoss", "BCEWithLogitsLoss", "MSELoss", "SmoothL1Loss", "HuberLoss"])
        model_layout.addWidget(self.loss_combo, 2, 1)

        model_layout.addWidget(QLabel("Learning Rate:"), 3, 0)
        self.lr_combo = QComboBox()
        self.lr_combo.addItems(["0.01", "0.001", "0.0001", "0.00001"])
        self.lr_combo.setCurrentText("0.001")
        model_layout.addWidget(self.lr_combo, 3, 1)

        model_group.setLayout(model_layout)

        # Groupe des paramètres d'entraînement
        params_group = QGroupBox("Training Parameters")
        self.layer_group = QGroupBox("Layer Types Configuration")
        self.layer_layout = QVBoxLayout()
        self.add_layer_button = QPushButton("➕ Ajouter une couche")
        self.add_layer_button.clicked.connect(self.add_layer_selector)
        self.layer_layout.addWidget(self.add_layer_button)
        self.layer_group.setLayout(self.layer_layout)
        params_layout = QGridLayout()

        params_layout.addWidget(QLabel("Window Size:"), 0, 0)
        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(10, 100)
        self.window_size_spin.setValue(20)
        params_layout.addWidget(self.window_size_spin, 0, 1)

        params_layout.addWidget(QLabel("Stride:"), 1, 0)
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 20)
        self.stride_spin.setValue(5)
        params_layout.addWidget(self.stride_spin, 1, 1)

        params_layout.addWidget(QLabel("Epochs:"), 2, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 200)
        self.epochs_spin.setValue(50)
        params_layout.addWidget(self.epochs_spin, 2, 1)

        params_layout.addWidget(QLabel("Batch Size:"), 3, 0)
        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 512)
        self.batch_size_input.setValue(64)
        params_layout.addWidget(self.batch_size_input, 3, 1)

        params_layout.addWidget(QLabel("Sequence Length:"), 4, 0)
        self.sequence_length_spin = QSpinBox()
        self.sequence_length_spin.setRange(5, 200)
        self.sequence_length_spin.setValue(20)
        params_layout.addWidget(self.sequence_length_spin, 4, 1)

        params_group.setLayout(params_layout)

        # Statut et résultats
        self.status = QLabel("No file loaded")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)

        # Créer une grande figure pour les plots
        fig_width, fig_height = 10, 20
        dpi = 100
        self.canvas = FigureCanvas(Figure(figsize=(fig_width, fig_height), dpi=dpi))
        self.canvas.setMinimumSize(int(fig_width * dpi), int(fig_height * dpi))

        # Conteneur pour le canvas dans un layout
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout()
        canvas_layout.addWidget(self.canvas)
        canvas_container.setLayout(canvas_layout)

        # Scroll area pour le canvas
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(canvas_container)

        # Disposition principale
        layout.addWidget(self.load_button)
        layout.addWidget(model_group)
        layout.addWidget(params_group)
        layout.addWidget(self.layer_group)
        layout.addWidget(self.train_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.load_model_button)
        layout.addWidget(self.status)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.results_text)
        layout.addWidget(self.scroll_area)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Connexions
        self.load_button.clicked.connect(self.load_file)
        self.train_button.clicked.connect(self.train_model)

    def add_layer_selector(self):
        combo = QComboBox()
        combo.addItems(["LSTM", "GRU", "RNN", "Transformer", "TinyTransformer"])
        self.layer_layout.addWidget(combo)
        self.layer_type_combos.append(combo)

    def load_multiple_files(self, folder_path, window_size, stride, selected_keys):
        X_all, y_all = [], []

        # 🔽 Récupérer seulement les fichiers cochés
        selected_files = [
            self.file_list.item(i).text()
            for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == Qt.Checked
        ]

        if not selected_files:
            raise ValueError("Aucun fichier sélectionné dans la liste.")

        for fname in sorted(selected_files):
            file_path = os.path.join(folder_path, fname)
            try:
                X, y = load_and_preprocess_data(file_path, window_size, stride, selected_keys)
                if X.size > 0:
                    X_all.append(X)
                    y_all.append(y)
            except Exception as e:
                print(f"❌ Erreur avec {fname}: {e}")

        if not X_all:
            raise ValueError("Aucun fichier valide ou signal utilisable trouvé.")

        return np.concatenate(X_all), np.concatenate(y_all)

    def load_file(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder with .h5 Files")
        if folder_path:
            try:
                # 🔹 Mettre à jour la liste cochable des fichiers
                self.file_list.clear()
                for fname in sorted(os.listdir(folder_path)):
                    if fname.endswith(".h5"):
                        item = QListWidgetItem(fname)
                        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                        item.setCheckState(Qt.Checked)  # ou Qt.Unchecked
                        self.file_list.addItem(item)

                # 🔹 Paramètres de la fenêtre
                window_size = self.window_size_spin.value()
                stride = self.stride_spin.value()
                selected_keys = None  # ex: ["emgL3", "imu1"] si tu veux filtrer les capteurs

                # 🔹 Chargement des données
                self.X, self.y = self.load_multiple_files(folder_path, window_size, stride, selected_keys)

                # 🔹 Rebalancer les classes pour l'entraînement
                from sklearn.utils import resample, shuffle
                class_0 = self.X[self.y == 0]
                class_1 = self.X[self.y == 1]
                if len(class_0) > len(class_1):
                    class_1 = resample(class_1, replace=True, n_samples=len(class_0), random_state=42)
                else:
                    class_0 = resample(class_0, replace=True, n_samples=len(class_1), random_state=42)

                self.X = np.concatenate([class_0, class_1])
                self.y = np.array([0] * len(class_0) + [1] * len(class_1))
                self.X, self.y = shuffle(self.X, self.y, random_state=42)

                # 🔹 Affichage
                unique, counts = np.unique(self.y, return_counts=True)
                stats = f"✅ Données chargées : {self.X.shape[0]} échantillons\n"
                stats += f"📊 Classes : {dict(zip(unique, counts))}\n"
                stats += f"🧩 Shape : {self.X.shape}\n"
                stats += f"📈 Features : {self.X.shape[2]} (EMG + IMU détectés)"

                self.status.setText("✔️ Dossier chargé avec succès")
                self.results_text.setText(stats)
                self.train_button.setEnabled(True)

            except Exception as e:
                import traceback
                self.status.setText(f"❌ Erreur: {str(e)}")
                traceback.print_exc()


    def auto_select_files(self):
        """Auto-select a percentage of files for training using checkboxes."""
        try:
            percentage = float(self.training_percentage_input.text().strip().replace('%', ''))
            if not (0 < percentage <= 100):
                raise ValueError

            total = self.file_list.count()
            if total == 0:
                QMessageBox.information(self, "Info", "Aucun fichier listé.")
                return

            to_check = round((percentage / 100) * total)

            all_items = [self.file_list.item(i) for i in range(total)]
            selected = random.sample(all_items, to_check)

            for item in all_items:
                item.setCheckState(Qt.Checked if item in selected else Qt.Unchecked)

            self.status.setText(f"✅ {to_check} fichiers cochés automatiquement ({percentage:.0f}%).")
        except ValueError:
            QMessageBox.warning(
                self,
                "Entrée invalide",
                "Entrez un pourcentage entre 1 et 100."
        )

    def train_model(self):
        if self.X is None:
            return
        try:
            # Split train/val/test
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                self.X, self.y, test_size=0.15, random_state=42, stratify=self.y
            )
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
            )
            # Récupérer les paramètres sélectionnés
            layer_types = [combo.currentText() for combo in self.layer_type_combos]
            model_type = layer_types if len(set(layer_types)) > 1 else layer_types[0]
            optimizer_name = self.optimizer_combo.currentText()
            loss_name = self.loss_combo.currentText()
            learning_rate = float(self.lr_combo.currentText())
            epochs = self.epochs_spin.value()
            batch_size = self.batch_size_input.value()
            sequence_length = self.sequence_length_spin.value()
            num_layers = len(layer_types)
            # Lancer le thread d'entraînement Keras
            self.training_thread = TrainingThread(
                self.X_train, self.X_val, self.X_test,
                self.y_train, self.y_val, self.y_test,
                model_type, optimizer_name, loss_name, epochs, learning_rate,
                batch_size=batch_size, num_layers=num_layers,
                sequence_length=sequence_length
            )
            self.training_thread.progress.connect(self.update_progress)
            self.training_thread.finished.connect(self.training_finished)
            self.train_button.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.status.setText(f"Training {model_type} with {optimizer_name} optimizer...")
            self.training_thread.start()
        except Exception as e:
            import traceback
            self.status.setText(f"Error: {str(e)}")
            traceback.print_exc()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def training_finished(self, model, history, test_results, optimizer):
        self.model = model
        self.progress_bar.setVisible(False)
        self.train_button.setEnabled(True)
        self.history = history
        self.test_results = test_results

        # Affichage des résultats
        layer_types = [combo.currentText() for combo in self.layer_type_combos]
        model_type = f"Hybrid ({' → '.join(layer_types)})" if len(set(layer_types)) > 1 else layer_types[0]
        optimizer_name = self.optimizer_combo.currentText()
        loss_name = self.loss_combo.currentText()

        results_text = f"=== TRAINING RESULTS ===\n"
        results_text += f"Model: {model_type}\n"
        results_text += f"Optimizer: {optimizer_name}\n"
        results_text += f"Loss Function: {loss_name}\n"
        results_text += f"Learning Rate: {self.lr_combo.currentText()}\n\n"
        results_text += f"Final test accuracy: {test_results['accuracy']:.4f}\n"
        results_text += f"Best validation accuracy: {max(history['val_acc']):.4f}\n"
        results_text += f"Training epochs: {len(history['train_loss'])}\n\n"
        results_text += "Classification report:\n"
        results_text += test_results['report']

        self.results_text.setText(results_text)
        self.status.setText("Training completed successfully")
        self.save_button.setEnabled(True)

        # Afficher les courbes
        self.plot_results(history, test_results, optimizer)

    def save_model(self):
        if self.model is None or self.history is None or self.test_results is None:
            QMessageBox.warning(self, "Erreur", "Modèle, historique ou résultats de test manquants.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "Keras Model (*.h5)")
        if file_path:
            if not file_path.endswith(".h5"):
                file_path += ".h5"
            try:
                self.model.save(file_path)

                # 🔽 Déterminer le type de modèle utilisé
                layer_types = [combo.currentText() for combo in self.layer_type_combos]
                model_type_name = (
                    "Hybrid (" + " → ".join(layer_types) + ")"
                    if len(set(layer_types)) > 1 else layer_types[0]
                )

                # 🔽 Ajouter le nom du modèle dans les résultats
                self.test_results["model_type"] = model_type_name

                # 🔽 Sauvegarder history + test_results correctement
                base = file_path[:-3]
                import json
                import numpy as np

                def make_json_serializable(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (np.integer, np.floating)):
                        return obj.item()
                    return obj

                with open(base + "_history.json", "w") as f:
                    json.dump(self.history, f, default=make_json_serializable, indent=2)

                with open(base + "_results.json", "w") as f:
                    json.dump(self.test_results, f, default=make_json_serializable, indent=2)

                self.status.setText(f"✅ Modèle sauvegardé dans : {file_path}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.status.setText(f"❌ Erreur lors de la sauvegarde : {str(e)}")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Keras Model (*.h5)")
        if not file_path:
            return
        try:
            self.model = keras.models.load_model(file_path)

            # 🔽 Recharger les fichiers d'historique et de résultats
            base = file_path[:-3]
            import os
            import json
            import numpy as np
            hist_path = base + "_history.json"
            res_path = base + "_results.json"

            if os.path.exists(hist_path) and os.path.exists(res_path):
                with open(hist_path, "r") as f:
                    self.history = json.load(f)
                with open(res_path, "r") as f:
                    self.test_results = json.load(f)
                    print("📌 Modèle utilisé :", self.test_results.get("model_type", "inconnu"))

                # ✅ Conversion sécurisée de predictions / true_labels
                for key in ['true_labels', 'predictions']:
                    if key in self.test_results:
                        try:
                            raw = self.test_results[key]
                            if isinstance(raw, str):
                                import ast
                                raw = ast.literal_eval(raw)  # convertit la string en liste
                            arr = np.asarray(raw)
                            print(f"✅ {key} loaded with shape: {arr.shape}, dtype: {arr.dtype}")
                            if arr.ndim == 1:
                                self.test_results[key] = arr
                            elif arr.ndim == 2 and arr.shape[1] == 1:
                                self.test_results[key] = arr.flatten()
                            else:
                                print(f"⚠️ Format inattendu pour {key}: {arr.shape}")
                                self.test_results[key] = arr
                        except Exception as e:
                            print(f"❌ Conversion échouée pour {key} : {e}")
                            self.test_results[key] = np.array([])

                self.plot_results(self.history, self.test_results, optimizer=None)

                # Résumé texte
                results_text = f"=== MODÈLE CHARGÉ ===\n"
                results_text += f"Fichier : {os.path.basename(file_path)}\n\n"
                results_text += f"Modèle : {self.test_results.get('model_type', 'inconnu')}\n"
                results_text += f"Test accuracy : {self.test_results.get('accuracy', '?'):.4f}\n"
                results_text += "Classification report :\n"
                results_text += self.test_results.get('report', 'N/A')
                self.results_text.setText(results_text)
            else:
                self.history = None
                self.test_results = None
                self.results_text.setText("ℹ️ Modèle chargé, mais aucun historique trouvé.")

            self.status.setText(f"✅ Modèle chargé depuis : {file_path}")
            self.save_button.setEnabled(True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status.setText(f"❌ Erreur lors du chargement : {str(e)}")


    def plot_results(self, history, test_results, optimizer):
        self.canvas.figure.clear()
        ax1 = self.canvas.figure.add_subplot(5, 1, 1)
        ax2 = self.canvas.figure.add_subplot(5, 1, 2)
        ax3 = self.canvas.figure.add_subplot(5, 1, 3)
        ax4 = self.canvas.figure.add_subplot(5, 1, 4)
        ax5 = self.canvas.figure.add_subplot(5, 1, 5)

        # 1️⃣ Courbes de loss
        ax1.plot(history.get('train_loss', []), label='Train Loss', color='blue', linewidth=2)
        ax1.plot(history.get('val_loss', []), label='Val Loss', color='red', linewidth=2)
        ax1.set_title('Loss Evolution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2️⃣ Courbes d'accuracy
        ax2.plot(history.get('train_acc', []), label='Train Accuracy', color='blue', linewidth=2)
        ax2.plot(history.get('val_acc', []), label='Validation Accuracy', color='red', linewidth=2)
        ax2.set_title('Accuracy Evolution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3️⃣ Prédictions vs Vérité terrain
        preds = np.asarray(test_results.get('predictions', []))
        true = np.asarray(test_results.get('true_labels', []))
        if preds.size > 0 and true.size > 0:
            sample_size = min(200, len(preds))
            indices = np.random.choice(len(preds), sample_size, replace=False)
            ax3.scatter(indices, true[indices], alpha=0.6, label='True', color='blue', s=15)
            ax3.scatter(indices, preds[indices], alpha=0.6, label='Predicted', color='red', s=15)
            ax3.set_title('Predictions vs Ground Truth (sample)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Sample Index')
            ax3.set_ylabel('Class')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No predictions available', transform=ax3.transAxes,
                    ha='center', va='center', fontsize=12)

        # 4️⃣ Matrice de confusion
        if preds.size > 0 and true.size > 0:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true, preds)
            im = ax4.imshow(cm, interpolation='nearest', cmap='Blues')
            ax4.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Predicted')
            ax4.set_ylabel('True')
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax4.text(j, i, str(cm[i, j]), ha='center', va='center',
                            color='white' if cm[i, j] > cm.max() / 2 else 'black', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No data for confusion matrix', transform=ax4.transAxes,
                    ha='center', va='center', fontsize=12)

        # 5️⃣ Learning Rate
        if 'lr' in history:
            ax5.plot(history['lr'], label="Learning Rate", color="purple")
            ax5.set_title("Learning Rate Evolution", fontsize=14, fontweight='bold')
            ax5.set_xlabel("Epoch")
            ax5.set_ylabel("LR")
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No learning rate data', transform=ax5.transAxes,
                    ha='center', va='center', fontsize=12)

        self.canvas.figure.tight_layout(pad=2.0)
        self.canvas.draw()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = EMGIMUGUI()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Erreur lors du lancement de l'application: {e}")
        import traceback
        traceback.print_exc()