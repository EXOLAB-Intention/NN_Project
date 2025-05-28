# requirements: pyqt5, torch, numpy, h5py, matplotlib, scikit-learn
import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, 
    QWidget, QLabel, QComboBox, QSpinBox, QHBoxLayout, QTextEdit, QProgressBar,
    QGroupBox, QGridLayout, QDoubleSpinBox,QScrollArea,QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import math

class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size//2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size//2, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, (h_n, _) = self.lstm2(out)
        out = self.fc1(h_n[-1])
        out = self.relu(out)
        out = self.fc2(out)
        if out.shape[1] == 1:
            out = out.squeeze(1)  
        return out

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.rnn1 = nn.RNN(input_size, hidden_size, num_layers=num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.rnn2 = nn.RNN(hidden_size, hidden_size//2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size//2, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out, _ = self.rnn1(x)
        out = self.dropout(out)
        out, h_n = self.rnn2(out)
        out = self.fc1(h_n[-1])
        out = self.relu(out)
        out = self.fc2(out)
        if out.shape[1] == 1:
            out = out.squeeze(1)  
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out, h_n = self.gru(x) 
        out = self.dropout(h_n[-1])  
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        if self.num_classes == 1:
            out = out.squeeze(1)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=8, num_layers=2, num_classes=2, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        out = self.classifier(x)
        if self.num_classes == 1:
            out = out.squeeze(1)  
        return out

class TinyTransformerModel(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=4, num_layers=1, num_classes=2, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        out = self.classifier(x)
        if self.num_classes == 1:
            out = out.squeeze(1)  
        return out
    
class HybridNNModel(nn.Module):
    def __init__(self, input_size, layer_types, num_classes=2, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        in_size = input_size

        for layer_type in layer_types:
            if layer_type == "RNN":
                self.layers.append(nn.RNN(in_size, in_size, batch_first=True))
            elif layer_type == "GRU":
                self.layers.append(nn.GRU(in_size, in_size, batch_first=True))
            elif layer_type == "LSTM":
                self.layers.append(nn.LSTM(in_size, in_size, batch_first=True))
            elif layer_type == "Transformer":
                self.layers.append(
                    nn.TransformerEncoderLayer(d_model=in_size, nhead=2, batch_first=True)
                )
            elif layer_type == "TinyTransformer":
                self.layers.append(
                    nn.TransformerEncoderLayer(d_model=in_size, nhead=1, dim_feedforward=32, batch_first=True)
                )
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        self.classifier = nn.Sequential(
            nn.Linear(in_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            if isinstance(layer, (nn.RNN, nn.GRU, nn.LSTM)):
                out, _ = layer(out)
            else:
                out = layer(out)
        out = out[:, -1, :]  
        return self.classifier(out)

def get_model(model_type, input_size, num_classes=2, num_layers=2, sequence_length=None):
    if isinstance(model_type, list):  
        return HybridNNModel(input_size, model_type, num_classes)
    elif model_type == "LSTM":
        return ImprovedLSTMModel(input_size, num_layers=num_layers, num_classes=num_classes)
    elif model_type == "RNN":
        return RNNModel(input_size, num_layers=num_layers, num_classes=num_classes)
    elif model_type == "GRU":
        return GRUModel(input_size, num_layers=num_layers, num_classes=num_classes)
    elif model_type == "Transformer":
        return TransformerModel(input_size, num_layers=num_layers, num_classes=num_classes)
    elif model_type == "TinyTransformer":
        return TinyTransformerModel(input_size, num_layers=num_layers, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_optimizer(name, model_params, lr=0.001):
    """Factory function to create optimizers"""
    if name == "Adam":
        return torch.optim.Adam(model_params, lr=lr)
    elif name == "SGD":
        return torch.optim.SGD(model_params, lr=lr)
    elif name == "AdamW":
        return torch.optim.AdamW(model_params, lr=lr)
    else:
        raise ValueError("Optimiseur non supporté.")

def get_loss_function(loss_name):
    """Factory function to create loss functions"""
    if loss_name == "MSELoss":
        return nn.MSELoss()
    elif loss_name == "SmoothL1Loss":
        return nn.SmoothL1Loss()
    elif loss_name == "HuberLoss":
        return nn.HuberLoss()
    elif loss_name == "CrossEntropyLoss":
        weights = torch.tensor([1.0, 1.0])  
        return nn.CrossEntropyLoss(weight=weights)
    elif loss_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Loss non supportée.")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes=2, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def load_and_preprocess_data(path, window_size=20, stride=5, selected_keys=None):
    """Charge un fichier HDF5 (format trial_* ou Sensor/Controller) avec affichage des signaux."""
    X_all, y_all = [], []

    import os
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
            y = np.array(ctrl["button_ok"])  # ⚠️ PAS de [0] ici

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

class TrainingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object, object,object)  

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
        import random
        import torch
        import numpy as np

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            input_size = self.X_train.shape[2]
            num_classes = len(np.unique(self.y_train))

            if self.loss_name in ["BCEWithLogitsLoss", "MSELoss", "SmoothL1Loss", "HuberLoss"]:
                num_classes = 1

            model = get_model(self.model_type, input_size, num_classes, num_layers=self.num_layers)
            criterion = get_loss_function(self.loss_name)
            optimizer = get_optimizer(self.optimizer_name, model.parameters(), self.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

            model.to(device)

            X_train_tensor = torch.FloatTensor(self.X_train)
            X_val_tensor = torch.FloatTensor(self.X_val)
            X_test_tensor = torch.FloatTensor(self.X_test)

            if isinstance(criterion, (nn.BCEWithLogitsLoss, nn.MSELoss, nn.SmoothL1Loss, nn.HuberLoss)):
                y_train_tensor = torch.FloatTensor(self.y_train)
                y_val_tensor = torch.FloatTensor(self.y_val)
                y_test_tensor = torch.FloatTensor(self.y_test)
            else:
                y_train_tensor = torch.LongTensor(self.y_train)
                y_val_tensor = torch.LongTensor(self.y_val)
                y_test_tensor = torch.LongTensor(self.y_test)
                
            X_train_tensor = X_train_tensor.to(device)
            X_val_tensor = X_val_tensor.to(device)
            X_test_tensor = X_test_tensor.to(device)
            y_train_tensor = y_train_tensor.to(device)
            y_val_tensor = y_val_tensor.to(device)
            y_test_tensor = y_test_tensor.to(device)

            # Training history
            history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
            self.last_training_history = history
            best_val_acc = 0
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            for epoch in range(self.epochs):
                # Training
                model.train()
                train_loss = 0
                train_correct = 0
                
                # Mini-batch training
                batch_size = self.batch_size
                num_batches = 0
                for i in range(0, len(X_train_tensor), batch_size):
                    batch_x = X_train_tensor[i:i+batch_size]
                    batch_y = y_train_tensor[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)

                    # Corriger la forme si nécessaire
                    if isinstance(criterion, (nn.MSELoss, nn.SmoothL1Loss, nn.HuberLoss,nn.BCEWithLogitsLoss)):
                        if outputs.shape != batch_y.shape:
                            batch_y = batch_y.unsqueeze(1)
                    loss = criterion(outputs, batch_y)

                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    # Adapt output prediction logic based on loss type
                    if isinstance(criterion, nn.CrossEntropyLoss):
                        preds = outputs.argmax(1)
                    elif isinstance(criterion, nn.BCEWithLogitsLoss):
                        preds = (torch.sigmoid(outputs) > 0.5).long()
                    else:
                        preds = outputs.round().long()  # For regression-based binary classification

                    train_correct += (preds == batch_y.long()).sum().item()
                    num_batches += 1

                     # ✅ Affichage clair
                    batch_index = i // batch_size + 1
                    total_batches = len(X_train_tensor) // batch_size + (1 if len(X_train_tensor) % batch_size > 0 else 0)
                    print(f"Epoch [{epoch+1}/{self.epochs}], Batch [{batch_index}/{total_batches}], Loss: {loss.item():.4f}")
            
                # Validation
                model.eval()
                val_loss = 0
                val_correct = 0
                val_batches = 0

                with torch.no_grad():
                    for i in range(0, len(X_val_tensor), batch_size):
                        batch_x = X_val_tensor[i:i+batch_size]
                        batch_y = y_val_tensor[i:i+batch_size]

                        outputs = model(batch_x)

                        if isinstance(criterion, (nn.MSELoss, nn.SmoothL1Loss, nn.HuberLoss, nn.BCEWithLogitsLoss)):
                            if outputs.shape != batch_y.shape:
                                batch_y = batch_y.unsqueeze(1)

                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                        if isinstance(criterion, nn.CrossEntropyLoss):
                            preds = outputs.argmax(1)
                        elif isinstance(criterion, nn.BCEWithLogitsLoss):
                            preds = (torch.sigmoid(outputs) > 0.5).long()
                        else:
                            preds = outputs.round().long()

                        val_correct += (preds == batch_y.long()).sum().item()
                        val_batches += 1

                val_loss = val_loss / val_batches
                val_acc = val_correct / len(self.y_val)
            
                # Ajoute ceci :
                history['train_loss'].append(train_loss / num_batches)
                history['train_acc'].append(train_correct / len(self.y_train))
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                if 'lr' not in history:
                    history['lr'] = []
                history['lr'].append(optimizer.param_groups[0]['lr'])

            # Load best model
            model.load_state_dict(best_model_state)
            
            # Final test
            # Final test
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)

                # 🔧 Assure la compatibilité des dimensions avec la loss
                if isinstance(criterion, (nn.MSELoss, nn.SmoothL1Loss, nn.HuberLoss, nn.BCEWithLogitsLoss)):
                    if test_outputs.shape != y_test_tensor.shape:
                        y_test_tensor = y_test_tensor.unsqueeze(1)

                # 🔎 Générer les prédictions selon le type de tâche
                if isinstance(criterion, nn.CrossEntropyLoss):
                    test_preds = test_outputs.argmax(1).cpu().numpy()
                elif isinstance(criterion, nn.BCEWithLogitsLoss):
                    test_preds = (torch.sigmoid(test_outputs) > 0.5).long().cpu().numpy()
                else:
                    test_preds = test_outputs.round().long().cpu().numpy()

                test_targets = self.y_test  # déjà numpy
                test_acc = accuracy_score(test_targets, test_preds)
                test_report = classification_report(test_targets, test_preds, zero_division=0)

            test_results = {
                'accuracy': test_acc,
                'predictions': test_preds,
                'true_labels': test_targets,
                'report': test_report
            }

            self.finished.emit(model, history, test_results, optimizer)

        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()

class EMGIMUGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced EMG/IMU Button Classifier")
        self.setGeometry(100, 100, 1400, 900)
        
        # Variables
        self.X = None
        self.y = None
        self.model = None
        # Initialiser les datasets pour éviter les erreurs d'attribut
        self.X = self.y = None
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

        params_layout.addWidget(QLabel("Sequence Length:"), 5, 0)
        self.sequence_length_spin = QSpinBox()
        self.sequence_length_spin.setRange(5, 200)
        self.sequence_length_spin.setValue(20)
        params_layout.addWidget(self.sequence_length_spin, 5, 1)

        params_group.setLayout(params_layout)

        # Statut et résultats
        self.status = QLabel("No file loaded")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)

        # Créer une grande figure pour les plots
        from matplotlib.figure import Figure
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
        layout = QVBoxLayout()
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
        import os
        X_all, y_all = [], []

        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith(".h5"):
                continue

            file_path = os.path.join(folder_path, fname)

            # Étape 1 : filtre structurel
            try:
                with h5py.File(file_path, 'r') as h5f:
                    keys = list(h5f.keys())
                    if not any(k.startswith("trial_") or k in ["Sensor", "Controller"] for k in keys):
                        print(f"⚩️ Ignoré (pas un fichier de données valide) : {fname}")
                        continue
            except Exception as e:
                print(f"⚩️ Ignoré (fichier non lisible) : {fname} ({e})")
                continue

            # Étape 2 : traitement normal
            try:
                print(f"🔍 Lecture de : {fname}")
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
                window_size = self.window_size_spin.value()
                stride = self.stride_spin.value()

                selected_keys = None  # ou ex: ["emgL3", "imu1"] si tu veux filtrer

                self.X, self.y = self.load_multiple_files(folder_path, window_size, stride, selected_keys)

                print("Classes:", dict(zip(*np.unique(self.y, return_counts=True))))

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

                unique, counts = np.unique(self.y, return_counts=True)
                stats = f"Data loaded: {self.X.shape[0]} samples\n"
                stats += f"Classes: {dict(zip(unique, counts))}\n"
                stats += f"Shape: {self.X.shape}\n"
                stats += f"Features: {self.X.shape[2]} (EMG + IMU détectés)"

                self.status.setText("✔️ Dossier chargé avec succès")
                self.results_text.setText(stats)
                self.train_button.setEnabled(True)

            except Exception as e:
                import traceback
                self.status.setText(f"❌ Erreur: {str(e)}")
                traceback.print_exc()

                    
    def train_model(self):
        if self.X is None:
            return
        
        try:
            # Définir le device pour le GPU/CPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Train/val/test split
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                self.X, self.y, test_size=0.15, random_state=42, stratify=self.y
            )
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
            )

            from collections import Counter

            # Calcul des poids des classes
            class_counts = Counter(self.y_train)
            total = sum(class_counts.values())
            weights = [total / class_counts[i] for i in sorted(class_counts)]
            weights = torch.tensor(weights, dtype=torch.float).to(device)

            # Récupérer les paramètres sélectionnés
            layer_types = [combo.currentText() for combo in self.layer_type_combos]
            model_type =  model_type = layer_types if len(set(layer_types)) > 1 else layer_types[0]
            optimizer_name = self.optimizer_combo.currentText()
            loss_name = self.loss_combo.currentText()
            learning_rate = float(self.lr_combo.currentText())
            epochs = self.epochs_spin.value()
            batch_size = self.batch_size_input.value()
            sequence_length = self.sequence_length_spin.value()

            # Sélection de la fonction de perte
            if loss_name == "CrossEntropyLoss":
                criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                criterion = get_loss_function(loss_name)

            # Initialiser le thread d'entraînement
            self.training_thread = TrainingThread(
                self.X_train, self.X_val, self.X_test,
                self.y_train, self.y_val, self.y_test,
                model_type, optimizer_name, loss_name, epochs, learning_rate,
                batch_size=batch_size, num_layers=len(layer_types),
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
        self.last_training_history = history 
        self.test_results = test_results
        
        # Display results
        layer_types = [combo.currentText() for combo in self.layer_type_combos]
        model_type = f"Hybrid ({' → '.join(layer_types)})"
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
        
        # Plot results
        self.plot_results(history, test_results, optimizer)
    
    def save_model(self):
        import h5py
        import numpy as np
        from PyQt5.QtWidgets import QFileDialog

        if self.model is None or self.history is None or self.test_results is None:
            QMessageBox.warning(self, "Erreur", "Modèle, historique ou résultats de test manquants.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "HDF5 Files (*.h5)")
        if file_path:
            if not file_path.endswith(".h5"):
                file_path += ".h5"

            try:
                # 🔁 Reconstruire les infos nécessaires
                layer_types = [combo.currentText() for combo in self.layer_type_combos]
                model_type = layer_types if len(set(layer_types)) > 1 else layer_types[0]
                input_size = self.X.shape[2]
                num_classes = 1 if self.loss_combo.currentText() in [
                    "BCEWithLogitsLoss", "MSELoss", "SmoothL1Loss", "HuberLoss"
                ] else 2
                num_layers = len(layer_types)
                sequence_length = self.sequence_length_spin.value()

                with h5py.File(file_path, 'w') as f:
                    # Métadonnées
                    f.attrs['model_type'] = str(model_type)
                    f.attrs['input_size'] = input_size
                    f.attrs['num_classes'] = num_classes
                    f.attrs['num_layers'] = num_layers
                    f.attrs['sequence_length'] = sequence_length

                    # Poids du modèle
                    weights_group = f.create_group("weights")
                    for name, param in self.model.state_dict().items():
                        weights_group.create_dataset(name, data=param.detach().cpu().numpy())

                    # Historique d'entraînement
                    hist_group = f.create_group("history")
                    for k, v in self.history.items():
                        hist_group.create_dataset(k, data=np.array(v))

                    # Résultats de test
                    test_group = f.create_group("test_results")
                    test_group.create_dataset("predictions", data=np.array(self.test_results["predictions"]))
                    test_group.create_dataset("true_labels", data=np.array(self.test_results["true_labels"]))
                    test_group.attrs["accuracy"] = self.test_results["accuracy"]
                    test_group.attrs["report"] = str(self.test_results["report"])

                self.status.setText(f"✅ Modèle sauvegardé dans : {file_path}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.status.setText(f"❌ Erreur lors de la sauvegarde : {str(e)}")


    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "HDF5 Files (*.h5)")
        if not file_path:
            return

        import h5py
        with h5py.File(file_path, 'r') as f:
            model_type_str = f.attrs['model_type']
            model_type = eval(model_type_str) if model_type_str.startswith("[") else model_type_str
            input_size = int(f.attrs['input_size'])
            num_classes = int(f.attrs['num_classes'])
            num_layers = int(f.attrs['num_layers'])
            sequence_length = int(f.attrs['sequence_length'])

            model = get_model(model_type, input_size, num_classes, num_layers, sequence_length)

            # Charger les poids
            state_dict = {}
            for name in f["weights"]:
                state_dict[name] = torch.tensor(f["weights"][name][...])
            model.load_state_dict(state_dict)
            model.eval()

            # Charger historique
            history = {}
            if "history" in f:
                for k in f["history"]:
                    history[k] = list(f["history"][k][...])

            # Charger résultats
            test_results = {}
            if "test_results" in f:
                grp = f["test_results"]
                test_results["predictions"] = grp["predictions"][...]
                test_results["true_labels"] = grp["true_labels"][...]
                test_results["accuracy"] = grp.attrs["accuracy"]
                test_results["report"] = grp.attrs["report"]

        # Mettre à jour interface
        self.model = model
        self.history = history
        self.test_results = test_results
        self.plot_results(history, test_results, optimizer=None)
        self.status.setText(f"✅ Modèle chargé depuis : {file_path}")

    @staticmethod
    def infer_predictions(outputs, loss_name, num_classes):
        if loss_name == "CrossEntropyLoss":
            return outputs.argmax(1).numpy()
        elif loss_name == "BCEWithLogitsLoss":
            return (torch.sigmoid(outputs) > 0.5).int().numpy()
        elif loss_name in ["MSELoss", "SmoothL1Loss", "HuberLoss"]:
            if num_classes == 1:
                return outputs.round().int().numpy()
            else:
                return outputs.argmax(1).numpy()
        else:
            return outputs.round().int().numpy()


    def plot_results(self, history, test_results, optimizer):
        import numpy as np
        from sklearn.metrics import confusion_matrix

        self.canvas.figure.clear()

        # Création des sous-graphiques
        ax1 = self.canvas.figure.add_subplot(5, 1, 1)
        ax2 = self.canvas.figure.add_subplot(5, 1, 2)
        ax3 = self.canvas.figure.add_subplot(5, 1, 3)
        ax4 = self.canvas.figure.add_subplot(5, 1, 4)
        ax5 = self.canvas.figure.add_subplot(5, 1, 5)

        # --- Graphe 1 : Loss ---
        ax1.plot(history.get('train_loss', []), label='Train Loss', color='blue', linewidth=2)
        ax1.plot(history.get('val_loss', []), label='Val Loss', color='red', linewidth=2)
        ax1.set_title('Loss Evolution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Graphe 2 : Accuracy ---
        ax2.plot(history.get('train_acc', []), label='Train Accuracy', color='blue', linewidth=2)
        ax2.plot(history.get('val_acc', []), label='Validation Accuracy', color='red', linewidth=2)
        ax2.set_title('Accuracy Evolution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # === DEBUG test_results content ===
        print("[DEBUG] Predictions available:", len(test_results.get('predictions', [])))
        print("[DEBUG] True labels available:", len(test_results.get('true_labels', [])))

        # --- Graphe 3 : Prédictions vs Vérité terrain ---
        if len(test_results['predictions']) > 0:
            sample_size = min(200, len(test_results['predictions']))
            indices = np.random.choice(len(test_results['predictions']), sample_size, replace=False)
            ax3.scatter(indices, test_results['true_labels'][indices], alpha=0.6, label='True', color='blue', s=15)
            ax3.scatter(indices, test_results['predictions'][indices], alpha=0.6, label='Predicted', color='red', s=15)
            ax3.set_title('Predictions vs Ground Truth (sample)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Sample Index')
            ax3.set_ylabel('Class')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No predictions available', transform=ax3.transAxes,
                    ha='center', va='center', fontsize=12)

        # --- Graphe 4 : Matrice de confusion ---
        if len(test_results['predictions']) > 0:
            cm = confusion_matrix(test_results['true_labels'], test_results['predictions'])
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

        # --- Graphe 5 : Learning Rate ---
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
