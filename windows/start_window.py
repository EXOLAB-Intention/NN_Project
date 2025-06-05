import os
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QFileDialog, QAction, QListWidgetItem, QMessageBox
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import QSize, Qt
from tensorflow.keras.models import load_model

from windows.dataset_builder_window import DatasetBuilderWindow
from windows.nn_designer_window import NeuralNetworkDesignerWindow  
from windows.nn_evaluator_window import NeuralNetworkEvaluator
from widgets.header import Header
import windows.progress_state as progress_state

class StartWindow(QMainWindow):
    def __init__(self):
        """
        Initialize the StartWindow, set up the UI and load recent folders.
        """
        super().__init__()
        self.setWindowTitle("Neural Network Trainer")
        self.setGeometry(100, 100, 600, 400)

        # Load recent folders from file
        self.recent_folders = self.load_recent_folders()

        # Build the UI
        self.init_ui()

        # Add a centered status bar at the bottom
        status_bar = self.statusBar()
        status_label = QLabel("Data Monitoring Software version 1.0.13")
        status_label.setAlignment(Qt.AlignCenter)  
        status_bar.addPermanentWidget(status_label, 1)  

    def init_ui(self):
        """
        Set up the main UI components: header, titles, buttons, recent folders list, and menu bar.
        """
        # Central widget and main layout (vertical)
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Add the custom header with navigation tabs
        header = Header(active_page="Start Window", parent_window=self)
        layout.addWidget(header)

        # Add the main title "Start"
        title_start = QLabel("Start")
        title_start.setFont(QFont("Arial", 42, QFont.Bold))
        title_start.setContentsMargins(10, 20, 0, 0)  
        layout.addWidget(title_start)

        # Add horizontal layout with "New Dataset" and "Open Dataset" buttons
        btn_layout = QHBoxLayout()
    
        btn_new = QPushButton("New Dataset…")
        btn_new.setIcon(QIcon("assets/new_data.png")) 

        btn_open = QPushButton("Open Dataset…")
        btn_open.setIcon(QIcon("assets/open_data.png"))

        btn_load = QPushButton("Load Model...")
        btn_load.setIcon(QIcon("assets/load.png"))

        btn_new.setIconSize(QSize(36, 36))  
        btn_open.setIconSize(QSize(38, 38))
        btn_load.setIconSize(QSize(33, 33))

        btn_new.setFixedWidth(450)
        btn_open.setFixedWidth(450)
        btn_load.setFixedWidth(450)

        btn_new.setStyleSheet("padding-top: 10px; padding-bottom: 10px;")
        btn_open.setStyleSheet("padding-top: 10px; padding-bottom: 10px;")
        btn_load.setStyleSheet("padding-top: 10px; padding-bottom: 10px;")

        btn_new.clicked.connect(self.create_new_dataset)
        btn_open.clicked.connect(self.open_dataset)
        btn_load.clicked.connect(self.load_existing_model)

        btn_layout.addWidget(btn_new)
        btn_layout.addWidget(btn_open)
        btn_layout.addWidget(btn_load)

        layout.addLayout(btn_layout)

        # Add the "Recent" title
        title_recent = QLabel("Recent")
        title_recent.setFont(QFont("Arial", 40, QFont.Bold))
        title_recent.setContentsMargins(10, 20, 0, 0)  
        layout.addWidget(title_recent)

        # Add the recent folders list widget
        self.recent_folders_list = QListWidget()
        self.recent_folders_list.setStyleSheet("border: 1px solid #dcdcdc; background-color: white;")
        self.recent_folders_list.itemDoubleClicked.connect(self.open_recent_folder)
        layout.addWidget(self.recent_folders_list)

        # Populate the recent folders list
        self.update_recent_folders_list()

        # Set the layout on the central widget
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Create the menu bar and "File" menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        # Add actions to the File menu
        action_create = QAction("Create new dataset", self)
        action_create.triggered.connect(self.create_new_dataset)

        action_load_dataset = QAction("Load existing dataset", self)
        action_load_dataset.triggered.connect(self.open_dataset)

        action_load_model = QAction("Load existing Model", self)
        action_load_model.triggered.connect(self.load_existing_model)

        file_menu.addAction(action_create)
        file_menu.addAction(action_load_dataset)
        file_menu.addAction(action_load_model)

    def create_new_dataset(self):
        """
        Open the DatasetBuilderWindow for creating a new dataset.
        """
        progress_state.dataset_built = False  # Reset progress state
        progress_state.nn_designed = False
        progress_state.training_started = False  # Reset training state
        saved_state = {"from_start_window": True}
        self.set_tabs_enabled(False)  # Disable tabs during dataset creation
        self.dataset_builder = DatasetBuilderWindow(start_window_ref=self)  
        self.dataset_builder.showMaximized()
        self.hide()
        self.set_tabs_enabled(True)  # Re-enable tabs after dataset creation

    def open_dataset(self):
        """
        Open a dialog to select a dataset folder, add it to recents, and open it in NeuralNetworkDesignerWindow.
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder_path:
            self.add_to_recent_folders(folder_path)
            self.open_neural_network_designer(folder_path)

    def open_recent_folder(self, item):
        """
        Open a recent folder in NeuralNetworkDesignerWindow if it exists, otherwise remove it from recents.
        """
        folder_path = item.text()
        if os.path.exists(folder_path):
            self.open_neural_network_designer(folder_path)
        else:
            self.recent_folders.remove(folder_path)
            self.save_recent_folders()
            self.update_recent_folders_list()

    def find_nn_project_root(self, path):
        while path and os.path.basename(path) != "NN_Project":
            parent = os.path.dirname(path)
            if parent == path:
                return None
            path = parent
        return path

    def open_neural_network_designer(self, folder_path):
        """
        Open the NeuralNetworkDesignerWindow with the given dataset folder.
        """
        self.hide()  
        progress_state.dataset_built = True

        # ➤ Liste des chemins relatifs complets
        project_root = self.find_nn_project_root(folder_path)
        all_files = [
            os.path.relpath(os.path.join(folder_path, f), project_root)
            for f in os.listdir(folder_path) if f.endswith('.h5')
        ]
        saved_state = {
            "dataset_path": folder_path,
            "all_files": all_files,
            "selected_files": []
        }
        self.nn_designer_window = NeuralNetworkDesignerWindow(
            dataset_path=folder_path,
            saved_state=saved_state
        )
        self.nn_designer_window.populate_file_list_with_paths(all_files, all_files)
        self.nn_designer_window.showMaximized()
        self.nn_designer_window.parent_window = self

    def load_recent_folders(self):
        """
        Load the list of recent folders from a text file.
        """
        recent_file = "recent_folders.txt"
        if os.path.exists(recent_file):
            with open(recent_file, "r") as file:
                return [line.strip() for line in file.readlines()]
        return []

    def save_recent_folders(self):
        """
        Save the current list of recent folders to a text file.
        """
        recent_file = "recent_folders.txt"
        with open(recent_file, "w") as file:
            file.writelines(f"{folder}\n" for folder in self.recent_folders)

    def update_recent_folders_list(self):
        """
        Update the QListWidget to display the current recent folders with increased font size and padding.
        """
        self.recent_folders_list.clear()
        self.recent_folders_list.setStyleSheet("""
            QListWidget::item {
                padding: 5px 30px;  
            }
        """)
        for folder in self.recent_folders:
            item = QListWidgetItem(folder)
            font = QFont("Arial", 16)  
            item.setFont(font)
            self.recent_folders_list.addItem(item)

    def add_to_recent_folders(self, folder):
        """
        Add a folder to the top of the recent folders list, ensuring no duplicates and a maximum of 10 items.
        """
        if folder in self.recent_folders:
            self.recent_folders.remove(folder)
        self.recent_folders.insert(0, folder)
        self.recent_folders = self.recent_folders[:10]
        self.save_recent_folders()
        self.update_recent_folders_list()

    def load_existing_model(self):
        """Charge un modèle existant depuis un fichier .h5"""
        try:
            file_path = QFileDialog.getOpenFileName(self, "Load Model", "", "H5 Files (*.h5)")[0]
            if not file_path:
                return

            import h5py
            import json
            import numpy as np
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
            import os  # <-- Assure l'import ici

            with h5py.File(file_path, 'r') as hf:
                # 1. Recréer le modèle manuellement depuis la config
                config = json.loads(hf['model_config'][()].decode('utf-8'))

                # Créer un modèle séquentiel
                model = Sequential()

                # Parcourir les couches dans la config
                for layer_config in config['layers']:
                    if layer_config['class_name'] == 'InputLayer':
                        input_shape = layer_config['config']['batch_input_shape'][1:]
                        model.add(InputLayer(input_shape=input_shape))

                    elif layer_config['class_name'] == 'LSTM':
                        lstm_config = layer_config['config']
                        model.add(LSTM(
                            units=lstm_config['units'],
                            activation=lstm_config['activation'],
                            return_sequences=lstm_config['return_sequences'],
                            dropout=lstm_config['dropout'],
                            recurrent_dropout=lstm_config['recurrent_dropout']
                        ))

                    elif layer_config['class_name'] == 'Dense':
                        dense_config = layer_config['config']
                        model.add(Dense(
                            units=dense_config['units'],
                            activation=dense_config['activation']
                        ))

                    elif layer_config['class_name'] == 'Dropout':
                        dropout_config = layer_config['config']
                        model.add(Dropout(rate=dropout_config['rate']))

                # 2. Restaurer les poids
                weights_group = hf['model_weights']
                for layer in model.layers:
                    if layer.name in weights_group:
                        layer_weights = []
                        layer_group = weights_group[layer.name]
                        weight_names = sorted([k for k in layer_group.keys()])
                        for name in weight_names:
                            weight = layer_group[name][:]
                            layer_weights.append(weight)
                        if layer_weights:
                            layer.set_weights(layer_weights)

                # 3. Charger les paramètres
                training_params = json.loads(hf['training_params'][()].decode('utf-8'))

                # 4. Charger l'historique et le formater pour correspondre à keras.callbacks.History
                class HistoryWrapper:
                    def __init__(self, history_dict):
                        self.history = history_dict

                history_data = {}
                for key in hf['history'].keys():
                    history_data[key] = hf['history'][key][:]
                history = HistoryWrapper(history_data)

                # 5. Charger les résultats
                test_results = {}
                for key in hf['test_results'].keys():
                    if key.endswith('_json'):
                        base_key = key[:-5]
                        test_results[base_key] = json.loads(hf[f'test_results/{key}'][()].decode('utf-8'))
                    else:
                        test_results[key] = hf[f'test_results/{key}'][:]

                # 6. Mettre à jour progress_state
                import windows.progress_state as progress_state
                progress_state.trained_model = model
                progress_state.training_history = history.history
                progress_state.test_results = test_results
                progress_state.dataset_built = True
                progress_state.nn_designed = True
                progress_state.training_started = True

                # 7. Restaurer la liste des fichiers
                try:
                    files_group = hf['files']
                    selected_files = []
                    if 'selected_files' in files_group:
                        selected_files_dataset = files_group['selected_files']
                        if isinstance(selected_files_dataset, h5py.Dataset):
                            selected_files_data = selected_files_dataset[()]
                            if isinstance(selected_files_data, np.ndarray):
                                selected_files = [
                                    f.decode('utf-8') if isinstance(f, bytes) else str(f)
                                    for f in selected_files_data
                                ]
                            else:
                                print("Warning: selected_files_data is not a numpy array")
                        else:
                            print("Warning: selected_files is not an h5py dataset")

                    checked_files = []
                    if 'checked_files' in files_group:
                        checked_files_dataset = files_group['checked_files']
                        if isinstance(checked_files_dataset, h5py.Dataset):
                            checked_files_data = checked_files_dataset[()]
                            if isinstance(checked_files_data, np.ndarray):
                                checked_files = [
                                    f.decode('utf-8') if isinstance(f, bytes) else str(f)
                                    for f in checked_files_data
                                ]
                            else:
                                print("Warning: checked_files_data is not a numpy array")
                        else:
                            print("Warning: checked_files is not an h5py dataset")

                    print(f"Debug - Selected files: {len(selected_files)}, Checked files: {len(checked_files)}")

                except Exception as e:
                    print(f"Error loading files: {e}")
                    selected_files = []
                    checked_files = []

                # 8. Trouver le dossier NN_Project à partir du chemin du modèle
                def find_nn_project_root(start_path):
                    path = os.path.abspath(start_path)
                    while path and os.path.basename(path) != "NN_Project":
                        parent = os.path.dirname(path)
                        if parent == path:
                            return None
                        path = parent
                    return path

                PROJECT_ROOT = find_nn_project_root(file_path)
                if PROJECT_ROOT is None:
                    QMessageBox.critical(self, "Erreur", "Impossible de trouver le dossier NN_Project à partir du modèle chargé.")
                    return

                saved_state = {
                    "hyperparameters": training_params.get("hyperparameters", {}),
                    "optimizer": training_params.get("optimizer", ""),
                    "loss_function": training_params.get("loss_function", ""),
                    "training_history": history,
                    "test_results": test_results,
                    "all_files": selected_files,
                    "filtered_files": selected_files,       # ← essentiels pour l'affichage dans restore_state()
                    "original_files": selected_files,      # tous les fichiers affichés
                    "selected_files": checked_files,  # fichiers cochés
                    "dataset_path": PROJECT_ROOT 
                }

                print("Debug - About to create NeuralNetworkDesignerWindow")
                nn_designer = NeuralNetworkDesignerWindow(dataset_path=PROJECT_ROOT, saved_state=saved_state)
                self.nn_designer_window = nn_designer  # AJOUTE CETTE LIGNE
                nn_designer.parent_window = self       # AJOUTE CETTE LIGNE 
                print("Debug - NeuralNetworkDesignerWindow created")

                nn_designer.trained_model = model
                nn_designer.training_history = history
                nn_designer.test_results = test_results
                nn_designer.training_completed = True
                nn_designer.training_stopped = False
                nn_designer.state["trained_model"] = model
                nn_designer.state["training_history"] = history
                nn_designer.state["test_results"] = test_results
                nn_designer.state["training_completed"] = True
                nn_designer.state["training_stopped"] = False

                checked_file_names = [os.path.basename(f) for f in checked_files]
                nn_designer.populate_file_list_with_paths(selected_files, checked_files)
                print("Debug - populate_file_list_with_paths done")

                # 10. Créer et configurer l'évaluateur
                from windows.nn_evaluator_window import NeuralNetworkEvaluator
                nn_evaluator = NeuralNetworkEvaluator(saved_state={
                    "model": model,
                    "test_results": test_results
                })

                # 11. Connecter les fenêtres et régénérer les plots
                self.nn_evaluator_window = nn_evaluator  # AJOUTE CETTE LIGNE
                nn_evaluator.parent_window = self   

                nn_designer.plot_training_curves(history)

                if test_results:
                    true_labels = test_results.get('y_true', [])
                    predicted_labels = test_results.get('y_pred', [])
                    if len(true_labels) > 0 and len(predicted_labels) > 0:
                        from matplotlib.figure import Figure
                        pred_fig = Figure(figsize=(8, 6))
                        scatter_fig = Figure(figsize=(8, 6))
                        nn_evaluator.plot_prediction_vs_ground_truth(
                            fig=pred_fig,
                            ground_truth=true_labels,
                            predicted=predicted_labels
                        )
                        nn_evaluator.prediction_scatter_plot_canvas(scatter_fig)
                        nn_evaluator.plot_confusion_matrix(true_labels, predicted_labels)

                # Ajouter avant la ligne "self.hide()"
                nn_designer.loaded_params = training_params  # Stocke les paramètres originaux
                nn_designer.loaded_files = checked_files    # Stocke les fichiers originaux

                # 12. Afficher NN Designer
                self.hide()
                nn_designer.showMaximized()
                QMessageBox.information(self, "Success", "Model and all related data loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def find_project_root_from_file(self, file_path):
        """Trouve le dossier NN_Project à partir du chemin du fichier modèle."""
        path = os.path.abspath(file_path)
        while path and os.path.basename(path) != "NN_Project":
            path = os.path.dirname(path)
        return path

    def set_tabs_enabled(self, enabled: bool):
        """Active ou désactive tous les onglets du header."""
        if hasattr(self, "header") and self.header is not None:
            for tab in self.header.tabs.values():
                tab.setEnabled(enabled)

