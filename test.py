import os
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QComboBox, QPushButton, QFileDialog
)
import sys

# -------- Dataset ----------
class HDF5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5 = h5py.File(h5_path, 'r')
        self.emg_keys = ['emgL1', 'emgL2', 'emgL3', 'emgL4', 'emgR1', 'emgR2', 'emgR4']
        self.data = np.stack([self.h5[f'Sensor/EMG/{k}'][:] for k in self.emg_keys], axis=1)
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.data[idx + 1]
        return x, y

# -------- Modèle ----------
class SimpleRegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        return self.fc(x)

# -------- GUI ----------
class TrainingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Training Interface")

        layout = QVBoxLayout()

        # Optimizer selection
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD","AdamW"])
        layout.addWidget(QLabel("Choose Optimizer:"))
        layout.addWidget(self.optimizer_combo)

        # Loss selection
        self.loss_combo = QComboBox()
        self.loss_function_categories = {
            "Regression": [
                "MSELoss", "SmoothL1Loss", "HuberLoss"
            ],
            "Classification": [
                "CrossEntropyLoss", "BCEWithLogitsLoss"
            ]
        }
        for category, losses in self.loss_function_categories.items():
            for loss in losses:
                self.loss_combo.addItem(f"{category} - {loss}")

        layout.addWidget(QLabel("Choose Loss Function:"))
        layout.addWidget(self.loss_combo)

        # Button to train
        self.train_button = QPushButton("Train Model and Plot")
        self.train_button.clicked.connect(self.run_training)
        layout.addWidget(self.train_button)

        self.setLayout(layout)

    def get_loss_function(self, loss_name):
        if loss_name == "MSELoss":
            return nn.MSELoss()
        elif loss_name == "SmoothL1Loss":
            return nn.SmoothL1Loss()
        elif loss_name == "HuberLoss":
            return nn.HuberLoss()
        elif loss_name == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif loss_name == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Loss non supportée.")

    def get_optimizer(self, name, model_params, lr=0.001):
        if name == "Adam":
            return torch.optim.Adam(model_params, lr=lr)
        elif name == "SGD":
            return torch.optim.SGD(model_params, lr=lr)
        elif name == "AdamW":
            return torch.optim.AdamW(model_params, lr=lr)
        else:
            raise ValueError("Optimiseur non supporté.")

    def run_training(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "sensor_data", "data7.h5")

        dataset = HDF5Dataset(file_path)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)

        model = SimpleRegressor(input_size=7)
        optimizer = self.get_optimizer(self.optimizer_combo.currentText(), model.parameters())
        loss_text = self.loss_combo.currentText().split(" - ")[1]
        criterion = self.get_loss_function(loss_text)

        train_losses, val_losses = [], []

        for epoch in range(20):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
            train_loss = total_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    pred = model(x)
                    loss = criterion(pred, y)
                    total_val_loss += loss.item() * x.size(0)
            val_loss = total_val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        self.plot_losses(train_losses, val_losses)

    def plot_losses(self, train_losses, val_losses):
        plt.figure()
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='orange')
        plt.title("Training and Validation Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# -------- Exécution --------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = TrainingApp()
    win.resize(400, 200)
    win.show()
    sys.exit(app.exec_())