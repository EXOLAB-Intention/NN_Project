from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QGridLayout,
    QPushButton, QComboBox, QTextEdit, QListWidget, QListWidgetItem, QProgressDialog,
    QFrame, QMessageBox, QGroupBox, QScrollArea, QSizePolicy, QProgressBar,QStackedLayout, QFileDialog,
    QApplication
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIntValidator, QFont, QTextCursor, QIcon
import random
import os
import time
from widgets.header import Header
import windows.progress_state as progress_state
import numpy as np
import windows.progress_state as progress_state
from sklearn.model_selection import train_test_split
from training_core.training_utils import (
            load_and_preprocess_data,
            train_model,
            save_model_and_results
        )          
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import re
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from windows.nn_evaluator_window import NeuralNetworkEvaluator
class TrainingThread(QThread):
    training_finished = pyqtSignal(object, object, object)
    training_log = pyqtSignal(str)

    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, params, test_time=None, parent=None):
        super().__init__(parent)
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.params = params
        self._stop_flag = False
        self.test_time = test_time
        

    def stop(self):
        self._stop_flag = True

    def run(self):
        from training_core.training_utils import train_model

        def log_callback(msg):
            self.training_log.emit(msg)

        model, history, test_results, _ = train_model(
            self.X_train, self.X_val, self.X_test,
            self.y_train, self.y_val, self.y_test,
            model_type=self.params["layers"],
            optimizer_name=self.params["optimizer"],
            loss_name=self.params["loss_function"],
            epochs=self.params["epochs"],
            learning_rate=self.params["learning_rate"],
            batch_size=self.params["batch_size"],
            num_layers=len(self.params["layers"]),
            sequence_length=self.params["sequence_length"],
            verbose=1,
            stop_flag_getter=lambda: self._stop_flag,
            log_callback=log_callback,
            test_time=self.test_time
        )

        print("🧪 Test results keys:", test_results.keys())
        self.training_finished.emit(model, history, test_results)

        def __del__(self):
            """Ensure thread is stopped when deleted."""
            if self.isRunning():
                self.stop()
                self.wait()
    
class NeuralNetworkDesignerWindow(QMainWindow):
    def __init__(self, dataset_path=None, saved_state=None):
        super().__init__()
        if dataset_path is None:
            QMessageBox.critical(self, "Error", "dataset_path is None! Impossible d'ouvrir le designer sans dataset.")
            raise ValueError("dataset_path is None")
        self.dataset_path = dataset_path
        self.saved_state = saved_state
        # HARRY: Initialize header attribute first
        self.header = None
        self.dataset_path = dataset_path
        self.is_training = False
        self.training_stopped = False
        self.loaded_params = None  # Paramètres du modèle chargé
        self.loaded_files = None
        
        
        # HARRY: Register this window
        QApplication.instance().register_window(self)
        self.dataset_path = dataset_path
        self.setWindowTitle("Data Monitoring Software")
        self.resize(1000, 700)

        # Initialize state with defaults or saved values
        self.state = saved_state if saved_state else {
            "optimizer": "Adam",
            "loss_function": None,
            "hyperparameters": None,
            "selected_files": [],
            "dataset_path": dataset_path,
            "summary_text": "No parameters saved",
            "training_monitor_logs": "",
            "training_progress": 0,
            "eval_plot_index": 0,
            "training_history": None,
            "test_results": None,
            "trained_model": None
        }
        
        if saved_state and saved_state.get("training_started"):
            progress_state.training_started = True
            
        # Initialize layer-related attributes
        self.layer_type_combos = []  # Stores layer type combo boxes
        self.layer_combo_rows = []  # Stores layer rows
        self.layer_scroll_layout = None

        self.hyperparams_saved = saved_state is not None and saved_state.get("hyperparameters") is not None
        self.training_stopped = False
        
        self.init_ui()
        
        # HARRY: Ensure header is created after UI initialization
        if not hasattr(self, 'header') or self.header is None:
            self.create_header()
        self.restore_state()
        self.update_summary_display()

        # HARRY: Add training state flag
        self.is_training = False
        # HARRY: Add completion flag*

    def restore_state(self):
        """Restore all UI elements from self.state."""
        self.optimizer_combo.setCurrentText(self.state.get("optimizer", "Adam"))
        self.classification_loss_combo.setCurrentText(self.state.get("loss_function", "CrossEntropyLoss"))
        self.regression_loss_combo.setCurrentIndex(-1)  # Ensure regression loss is not selected by default

        # Ensure hyperparameters are initialized
        hp = self.state.get("hyperparameters", {})
        if hp is None:
            hp = {}
            self.state["hyperparameters"] = hp

        self.sequence_length_input.setText(str(hp.get("sequence_length", 50)))
        self.stride_input.setText(str(hp.get("stride", 5)))
        self.batch_size_input.setText(str(hp.get("batch_size", 32)))
        self.epoch_number_input.setText(str(hp.get("epoch_number", 10)))
        self.learning_rate_combo.setCurrentText(str(hp.get("learning_rate", 0.001)))

        self.clear_layer_combos()
        for layer_cfg in hp.get("layers", []):
            self.add_layer_combo_row(config=layer_cfg)

        # --- Ajout pour restaurer la liste des fichiers et les cases cochées ---
        all_files = self.state.get("filtered_files") or self.state.get("all_files") or []
        selected_files = self.state.get("selected_files", [])
        if all_files:
            self.populate_file_list_with_paths(all_files, selected_files)
        elif self.dataset_path:
            self.populate_file_list()
        QTimer.singleShot(100, self.restore_checkboxes)
        # ----------------------------------------------------------------------

        self.training_monitor_text.setText(self.state.get("training_monitor_logs", ""))
        self.training_progress_bar.setValue(self.state.get("training_progress", 0))
        self.nn_text_edit.setText(self.state.get("summary_text", "No hyperparameters saved"))  # Reset summary text
        self.eval_stack.setCurrentIndex(self.state.get("eval_plot_index", 0))

        # Restore evaluation plot if training history exists
        history = self.state.get("training_history", None)
        if history:
            self.plot_training_curves(history)
        if "current_page_index" in self.state:
            self.stacked_widget.setCurrentIndex(self.state["current_page_index"])
        if "eval_plot_state" in self.state:
            self.eval_stack.setCurrentIndex(self.state["eval_plot_state"])
        if "training_log" in self.state:
            self.training_monitor_text.setText(self.state["training_log"])
        if "progress_value" in self.state:
            self.training_progress_bar.setValue(self.state["progress_value"])
        # ✅ Restaure les objets pour pouvoir sauvegarder le modèle
        self.trained_model = self.state.get("trained_model", None)
        self.training_history = self.state.get("training_history", None)
        self.test_results = self.state.get("test_results", None)
        self.training_completed = self.state.get("training_completed", False)
        self.training_stopped = self.state.get("training_stopped", False)
        if self.training_completed:
            self.training_stopped = False


    def clear_layer_combos(self):
        """Remove all layer combo rows."""
        for row_widget in self.layer_combo_rows:
            row_widget.setParent(None)
            row_widget.deleteLater()
        self.layer_type_combos.clear()
        self.layer_combo_rows.clear()

    def add_layer_combo_row(self, config=None):
        # Create the layer row widget
        row_widget = QWidget()
        row_layout = QGridLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setHorizontalSpacing(10)
        row_layout.setVerticalSpacing(5)

        # Create layer number label
        layer_number = len(self.layer_combo_rows) + 1
        layer_label = QLabel(f"Layer {layer_number}")
        layer_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #333;")
        layer_label.setFixedHeight(20)

        # Layer type combobox
        combo = QComboBox()
        combo.addItems(["LSTM", "GRU", "RNN", "Transformer", "TinyTransformer"])
        combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #dcdcdc;
                padding: 4px;
                background-color: white;
                font-size: 12px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                border-left: 1px solid #cce0ff;
                background-color: #dceaf7;
                image: url(assets/arrow_down.png);
                min-width: 20px;
                min-height: 20px;
                border: none;
            }
        """)

        # Parameter widgets setup
        param_widgets = {}
        param_layout = QVBoxLayout()
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setSpacing(5)

        def create_param_input(label, default_val):
            lbl = QLabel(label)
            lbl.setStyleSheet("font-weight: bold; color: #333; font-size: 14px; border: none;")
            
            input_widget = None

            # Common combo box style (same as Layer Types selector)
            combo_style = """
                QComboBox {
                    border: 1px solid #dcdcdc;
                    padding: 4px;
                    background-color: white;
                    font-size: 12px;
                }
                QComboBox::drop-down {
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    border-left: 1px solid #cce0ff;
                    background-color: #dceaf7;
                    image: url(assets/arrow_down.png);
                    min-width: 20px;
                    min-height: 20px;
                    border: none;
                }
            """
            
            if label == "Activation":
                input_widget = QComboBox()
                input_widget.addItems(["tanh", "relu", "gelu", "sigmoid", "linear", "softmax"])
                input_widget.setCurrentText(str(default_val))
                input_widget.setStyleSheet(combo_style)  
            elif label == "Bidirectional":
                input_widget = QComboBox()
                input_widget.addItems(["False", "True"])
                input_widget.setCurrentText(str(default_val))
                input_widget.setStyleSheet(combo_style)  # Apply the same style

            else:
                input_widget = QLineEdit(str(default_val))
                input_widget.setStyleSheet("""
                    QLineEdit {
                        border: 1px solid #dcdcdc;
                        padding: 4px;
                        font-size: 12px;
                        color: #000;
                        background-color: #fff;
                    }
                """)

            input_widget.setFixedHeight(24)
            param_layout.addWidget(lbl)
            param_layout.addWidget(input_widget)
            param_widgets[label.lower().replace(" ", "_")] = input_widget

        def update_param_fields(layer_type):
            # Clear existing widgets
            for i in reversed(range(param_layout.count())):
                param_layout.itemAt(i).widget().setParent(None)
            param_widgets.clear()

            if layer_type in ["LSTM", "GRU", "RNN"]:
                create_param_input("Units", config.get("units", 64) if config else 64)
                create_param_input("Dropout", config.get("dropout", 0.3) if config else 0.3)
                create_param_input("Activation", config.get("activation", "tanh") if config else "tanh")
                create_param_input("Bidirectional", config.get("bidirectional", "False") if config else "False")
            elif layer_type in ["Transformer", "TinyTransformer"]:
                create_param_input("d_model", config.get("d_model", 64 if layer_type == "Transformer" else 32) if config else 64)
                create_param_input("num_heads", config.get("num_heads", 8 if layer_type == "Transformer" else 2) if config else 8)
                create_param_input("Dropout", config.get("dropout", 0.1) if config else 0.1)
                create_param_input("Attention Dropout", config.get("attention_dropout", 0.1) if config else 0.1)
                create_param_input("Activation", config.get("activation", "relu") if config else "relu")

        combo.currentTextChanged.connect(update_param_fields)
        combo.setCurrentText(config.get("type", "LSTM") if config else "LSTM")
        update_param_fields(combo.currentText())

        # Remove button
        remove_btn = QPushButton()
        remove_btn.setIcon(QIcon("assets/bin.png"))
        remove_btn.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: #f5f5f5;
            }
        """)
        remove_btn.setFixedSize(28, 28)
        remove_btn.clicked.connect(lambda: self.remove_layer_row(row_widget))

        # Add widgets to layout
        row_layout.addWidget(layer_label, 0, 0)
        row_layout.addWidget(combo, 1, 0)
        row_layout.addWidget(remove_btn, 1, 1)
        row_layout.addLayout(param_layout, 2, 0, 1, 2)

        # Add separator (except for first layer)
        if len(self.layer_combo_rows) > 0:
            separator = QWidget()
            sep_layout = QVBoxLayout(separator)
            sep_layout.setContentsMargins(0, 8, 0, 8)
            sep_layout.setSpacing(0)
            
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            line.setStyleSheet("background-color: #dcdcdc; border: none;")
            sep_layout.addWidget(line)
            
            self.layer_scroll_layout.addWidget(separator)
            row_widget.separator = separator

        self.layer_scroll_layout.addWidget(row_widget)
        
        # Store all layer components together
        self.layer_combo_rows.append({
            'widget': row_widget,
            'combo': combo,
            'param_widgets': param_widgets,
            'label': layer_label
        })

    def remove_layer_row(self, row_widget):
        """Remove a layer row and its separator"""
        for i, layer in enumerate(self.layer_combo_rows):
            if layer['widget'] == row_widget:
                # Remove separator if exists
                if hasattr(row_widget, 'separator'):
                    row_widget.separator.setParent(None)
                    row_widget.separator.deleteLater()
                
                # Remove the row widget
                row_widget.setParent(None)
                row_widget.deleteLater()
                
                # Remove from list
                self.layer_combo_rows.pop(i)
                
                # If we removed the first row, remove next row's separator
                if i == 0 and len(self.layer_combo_rows) > 0:
                    next_row = self.layer_combo_rows[0]['widget']
                    if hasattr(next_row, 'separator'):
                        next_row.separator.setParent(None)
                        next_row.separator.deleteLater()
                        delattr(next_row, 'separator')
                
                # Renumber remaining layers
                self.renumber_layer_labels()
                break

    def renumber_layer_labels(self):
        """Update all layer numbers"""
        for i, layer in enumerate(self.layer_combo_rows):
            layer['label'].setText(f"Layer {i+1}")

    def get_layer_configs(self):
        """Get configuration for all layers"""
        configs = []
        for layer in self.layer_combo_rows:
            config = {
                'type': layer['combo'].currentText(),
            }
            
            params = layer['param_widgets']
            layer_type = config['type']
            
            if layer_type in ["LSTM", "GRU", "RNN"]:
                config.update({
                    'units': int(params['units'].text()),
                    'dropout': float(params['dropout'].text()),
                    'activation': params['activation'].currentText(),
                    'bidirectional': params['bidirectional'].currentText() == "True"
                })
            elif layer_type in ["Transformer", "TinyTransformer"]:
                config.update({
                    'd_model': int(params['d_model'].text()),
                    'num_heads': int(params['num_heads'].text()),
                    'dropout': float(params['dropout'].text()),
                    'attention_dropout': float(params['attention_dropout'].text()),
                    'activation': params['activation'].currentText()
                })
            
            configs.append(config)
        return configs

    def get_saved_state(self):
        """Return the complete current state."""
        self.state["hyperparameters"] = {
            "layers": self.get_layer_configs(),
            "sequence_length": self.sequence_length_input.text(),
            "stride": self.stride_input.text(),
            "batch_size": self.batch_size_input.text(),
            "epoch_number": self.epoch_number_input.text(),
            "learning_rate": self.learning_rate_combo.currentText()
        }
        self.state["optimizer"] = self.optimizer_combo.currentText()
        if self.classification_loss_combo.currentIndex() >= 0:
            self.state["loss_function"] = self.classification_loss_combo.currentText()
        elif self.regression_loss_combo.currentIndex() >= 0:
            self.state["loss_function"] = self.regression_loss_combo.currentText()
        else:
            self.state["loss_function"] = None
        # Sauvegarde les chemins relatifs cochés
        self.state["selected_files"] = [
            self.file_list.item(i).data(Qt.UserRole)
            for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == Qt.Checked
        ]
        # Sauvegarde tous les fichiers affichés (pour restauration)
        self.state["all_files"] = [
            self.file_list.item(i).data(Qt.UserRole)
            for i in range(self.file_list.count())
        ]
        self.state["training_history"] = getattr(self, "training_history", None)
        self.state["test_results"] = getattr(self, "test_results", None)
        self.state["trained_model"] = getattr(self, "trained_model", None)
        self.state["training_monitor_logs"] = self.training_monitor_text.toPlainText()
        self.state["training_progress"] = self.training_progress_bar.value()
        self.state["summary_text"] = self.nn_text_edit.toPlainText()
        self.state["eval_plot_index"] = self.eval_stack.currentIndex()
        self.state["training_started"] = progress_state.training_started
        self.state["training_completed"] = getattr(self, "training_completed", False)
        self.state["training_stopped"] = getattr(self, "training_stopped", False)
        return self.state

    def update_summary_display(self):
        """Update the summary text area with current state."""
        if not self.hyperparams_saved or not self.state.get("hyperparameters"):
            self.nn_text_edit.setText("No hyperparameters saved")
            return
            
        hp = self.state["hyperparameters"]
        optimizer = self.state.get("optimizer", "Adam")
        loss_function = self.state.get("loss_function", "Not selected")
        layer_configs = hp.get("layers", [])
        layer_types = [layer["type"] for layer in layer_configs] if layer_configs else []
        
        summary = (
            f"Parameters saved:\n"
            f"• Optimizer: {optimizer}\n"
            f"• Loss Function: {loss_function}\n"
            f"• Layer Types: {', '.join(layer_types) if layer_types else 'None'}\n"
            f"• Sequence Length: {hp.get('sequence_length', 50)}\n"
            f"• Stride: {hp.get('stride', 5)}\n"
            f"• Batch Size: {hp.get('batch_size', 32)}\n"
            f"• Epoch Number: {hp.get('epoch_number', 10)}\n"
            f"• Learning Rate: {hp.get('learning_rate', 0.001)}"
        )
        self.nn_text_edit.setText(summary)

    def get_training_parameters(self):
        """Retourne tous les paramètres nécessaires à l'entraînement."""
        if "hyperparameters" not in self.state:
            QMessageBox.warning(self, "Missing Hyperparameters", "Please save your hyperparameters before starting training.")
            return None, None
        hp = self.state["hyperparameters"]
        params = {
            "optimizer": self.state["optimizer"],
            "loss_function": self.state["loss_function"],
            "loss_type": "Classification" if self.classification_loss_combo.currentIndex() >= 0 else "Regression",
            "layers": hp.get("layers", []),
            "sequence_length": int(hp.get("sequence_length", 50)),
            "stride": int(hp.get("stride", 5)),
            "batch_size": int(hp.get("batch_size", 32)),
            "epochs": int(hp.get("epoch_number", 10)),
            "learning_rate": float(hp.get("learning_rate", 0.001))
        }
        selected_files = [
            self.file_list.item(i).data(Qt.UserRole)
            for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == Qt.Checked
        ]

        return params, selected_files

    def save_hyperparameters(self):
        """Save the hyperparameters entered by the user and update state."""
        # Get values from input fields
        layer_configs = self.get_layer_configs()
        sequence_length = self.sequence_length_input.text()
        stride = self.stride_input.text()
        batch_size = self.batch_size_input.text()
        epoch_number = self.epoch_number_input.text()
        learning_rate = self.learning_rate_combo.currentText()
        optimizer = self.optimizer_combo.currentText().strip()
        classification_loss = self.classification_loss_combo.currentText().strip()
        regression_loss = self.regression_loss_combo.currentText().strip()

        # Validation
        if not layer_configs:
            QMessageBox.warning(self, "Missing Layers", "Please add at least one layer before saving hyperparameters.")
            return
        if not all([sequence_length, stride, batch_size, epoch_number, optimizer, learning_rate]):
            QMessageBox.warning(self, "Incomplete Hyperparameters", "Please fill in all hyperparameters before saving.")
            return
        if (not classification_loss and not regression_loss) or (classification_loss and regression_loss):
            QMessageBox.warning(self, "Invalid Selection", "Please select either a Classification or Regression loss.")
            return

        # Save in state
        self.state["optimizer"] = optimizer
        self.state["loss_function"] = classification_loss if classification_loss else regression_loss
        self.state["hyperparameters"] = {
            "layers": layer_configs,
            "sequence_length": sequence_length,
            "stride": stride,
            "batch_size": batch_size,
            "epoch_number": epoch_number,
            "learning_rate": learning_rate
        }
        
        # Mark as saved and update display
        self.hyperparams_saved = True
        self.update_summary_display()
        QMessageBox.information(self, "Parameters Saved", "The parameters have been saved successfully.")

    def init_ui(self):
        """Initialize all UI components of the window."""
        # Create scrollable container for responsiveness
        self.create_scrollable_container()
        
        # Create header with navigation
        self.create_header()
        
        # Create progress indicator
        self.create_progress_indicator()
        
        # Create main content sections
        self.create_top_controls()
        self.create_main_content()
        self.create_training_controls()
        self.create_training_monitor()
        
        # Create status bar
        self.create_status_bar()
        
        # Populate file list if dataset path was provided
        if self.dataset_path:
            self.populate_file_list()

    def create_scrollable_container(self):
        """Create the scrollable main container for responsive layout."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        container = QWidget()
        scroll.setWidget(container)
        
        self.main_layout = QVBoxLayout(container)
        self.main_layout.setContentsMargins(0, 0, 0, 0)  
        
        self.setCentralWidget(scroll)

    def create_header(self):
        """HARRY: Enhanced header creation with proper attribute setting"""
        try:
            header_container = QWidget()
            header_layout = QVBoxLayout(header_container)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(0)
            
            # Create and store header reference
            self.header = Header(active_page="Neural Network Designer", parent_window=self)
            header_layout.addWidget(self.header)
            
            # Add to main layout
            if hasattr(self, 'main_layout'):
                self.main_layout.addWidget(header_container, alignment=Qt.AlignTop)
            else:
                print("Warning: main_layout not found when creating header")
                
        except Exception as e:
            print(f"Error creating header: {e}")
            self.header = None

    def create_progress_indicator(self):
        """Create the progress indicator showing workflow steps."""
        # Top bar layout (back button, title, progress)
        top_bar_layout = QHBoxLayout()
        
        #  Left: Back button
        left_widget = QWidget()
        left_layout = QHBoxLayout()
        left_layout.setContentsMargins(0,0,0,0)
        back_btn = QPushButton("← Back")
        back_btn.setFixedSize(100, 30)
        back_btn.clicked.connect(self.go_back)
        left_layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        left_widget.setLayout(left_layout)

        # Center: Title
        center_widget = QWidget()
        center_layout = QHBoxLayout()
        center_layout.setContentsMargins(0,0,0,0)
        title_label = QLabel("Neural Network Designer")
        title_label.setStyleSheet("font-size: 30px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignLeft)
        center_layout.addWidget(title_label, alignment=Qt.AlignLeft)
        center_widget.setLayout(center_layout)

        # Right: Progress label
        right_widget = QWidget()
        right_layout = QHBoxLayout()
        right_layout.setContentsMargins(0,0,0,0)
        self.top_right_label = QLabel()
        font = QFont("Arial", 10)
        font.setItalic(True)
        self.top_right_label.setFont(font)
        self.top_right_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.top_right_label.setTextFormat(Qt.RichText)
        self.top_right_label.setStyleSheet("""
            background-color: rgba(200, 200, 200, 50);
            padding: 6px 12px;
            border-radius: 8px;
            margin: 5px;
        """)
        right_layout.addWidget(self.top_right_label, alignment=Qt.AlignRight)
        right_widget.setLayout(right_layout)

        # Add all sections to top bar
        top_bar_layout.addWidget(left_widget)
        top_bar_layout.addWidget(center_widget, stretch=1)
        top_bar_layout.addWidget(right_widget)

        self.main_layout.addLayout(top_bar_layout)
        
        # Initialize progress state
        self.update_progress(active_step="NN Designer", completed_steps=["Dataset Builder"])

    def create_top_controls(self):
        """Create the top row of controls (training set selection)."""
        top_row = QHBoxLayout()

        # Training set controls
        left_title_controls = QHBoxLayout()
        training_label = QLabel("Training/Test Set")
        training_label.setStyleSheet("font-weight: bold; font-style: italic;")
        left_title_controls.addWidget(training_label)
        
        # Training percentage input
        self.training_percentage_input = QLineEdit("20%")
        self.training_percentage_input.setMaximumWidth(60)
        left_title_controls.addWidget(self.training_percentage_input)
        
        # Auto-select button
        auto_select_button = QPushButton("Auto Select")
        auto_select_button.setStyleSheet("""
            background-color: #dceaf7; 
            font-size: 12px; 
            font-weight: bold; 
            border: 2px solid white; 
            padding: 4px 4px;
        """)
        auto_select_button.clicked.connect(self.auto_select_files)
        left_title_controls.addWidget(auto_select_button) 
        left_title_controls.addStretch()

        # Container for training controls
        training_container = QWidget()
        training_layout = QVBoxLayout(training_container)
        training_layout.setContentsMargins(7, 0, 0, 0)  
        training_layout.addLayout(left_title_controls)
        
        # Middle and right section labels
        middle_title = QLabel("NN Model")
        middle_title.setStyleSheet("font-weight: bold; font-style: italic;")
        
        right_title = QLabel("Evaluation Plot")
        right_title.setStyleSheet("font-weight: bold; font-style: italic;")

        # Add all to top row
        top_row.addWidget(training_container, 1)
        top_row.addWidget(middle_title, 1, alignment=Qt.AlignLeft)
        top_row.addWidget(right_title, 1, alignment=Qt.AlignLeft)

        self.main_layout.addLayout(top_row)

    def create_main_content(self):
        """Create the main content area with three panels."""
        content_layout = QHBoxLayout()
        
        # Left panel - File selection
        self.create_file_selection_panel(content_layout)
        
        # Middle panel - NN configuration
        self.create_nn_config_panel(content_layout)
        
        # Right panel - Evaluation plot
        self.create_evaluation_panel(content_layout)
        
        self.main_layout.addLayout(content_layout)

    def create_file_selection_panel(self, parent_layout):
        """
        Create the left panel for file selection.

        Args:
            parent_layout: The layout to add this panel to
        """
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel.setStyleSheet("background: white")
        left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)

        # Title
        left_layout.addWidget(QLabel("File name"))

        # File list widget
        self.file_list = QListWidget()
        self.file_list.setStyleSheet("""
            border: 1px solid #dcdcdc;
            background-color: white;
            padding: 4px;
        """)
        left_layout.addWidget(self.file_list)

        # Wrap in container for margins
        left_panel_container = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_container)
        left_panel_layout.setContentsMargins(7, 0, 0, 0)
        left_panel_layout.addWidget(left_panel)

        parent_layout.addWidget(left_panel_container, 1)


    def create_nn_config_panel(self, parent_layout):
        """
        Create the middle panel for NN configuration.

        Args:
            parent_layout: The layout to add this panel to
        """
        middle_panel = QFrame()
        middle_panel.setFrameShape(QFrame.StyledPanel)
        middle_panel.setStyleSheet("background: white")
        middle_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        middle_layout = QVBoxLayout(middle_panel)
        middle_layout.setContentsMargins(5, 5, 5, 5)  # Add some margins

        # Initialize optimizer combo box
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "AdamW"])

        # Initialize two loss combo boxes: one for Classification, one for Regression
        self.classification_loss_combo = QComboBox()
        self.regression_loss_combo = QComboBox()

        # Define loss function categories
        self.loss_function_categories = {
            "Classification": ["CrossEntropyLoss", "BCEWithLogitsLoss"],
            "Regression": ["MSELoss", "SmoothL1Loss", "HuberLoss"]
        }

        for loss in self.loss_function_categories["Classification"]:
            self.classification_loss_combo.addItem(loss)
        for loss in self.loss_function_categories["Regression"]:
            self.regression_loss_combo.addItem(loss)

        self.classification_loss_combo.currentIndexChanged.connect(
            lambda i: self.regression_loss_combo.setCurrentIndex(-1) if i != -1 else None
        )
        self.regression_loss_combo.currentIndexChanged.connect(
            lambda i: self.classification_loss_combo.setCurrentIndex(-1) if i != -1 else None
        )

        combo_style = """
            QComboBox { 
                border: 1px solid #dcdcdc; 
                padding: 4px; 
                background-color: white; 
                font-size: 12px; 
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                border-left: 1px solid #cce0ff;
                background-color: #dceaf7;
                image: url(assets/arrow_down.png);
                min-width: 20px;
                min-height: 20px;
                border: none;
            }
        """
        self.optimizer_combo.setStyleSheet(combo_style)
        self.classification_loss_combo.setStyleSheet(combo_style)
        self.regression_loss_combo.setStyleSheet(combo_style)

        middle_layout.addWidget(QLabel("<b><i>Optimizer</b></i>"))
        middle_layout.addWidget(self.optimizer_combo)
        middle_layout.addWidget(QLabel("<b><i>Classification Loss Function</b></i>"))
        middle_layout.addWidget(self.classification_loss_combo)
        middle_layout.addWidget(QLabel("<b><i>Regression Loss Function</b></i>"))
        middle_layout.addWidget(self.regression_loss_combo)

        # ===== Hyperparameters Section =====
        hyper_box = QGroupBox()
        hyper_box.setTitle("Hyperparameters")
        hyper_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-style: italic;
                border: 1px solid #dcdcdc;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: black;
            }
        """)
        hyper_layout = QGridLayout()
        hyper_layout.setContentsMargins(10, 15, 10, 10)
        hyper_layout.setVerticalSpacing(5)

        label_style = "font-weight: bold; color: #333; font-size: 14px; border: none;"
        input_style = """
            QLineEdit {
                border: 1px solid #dcdcdc;
                padding: 4px;
                font-size: 12px;
                color: #000;
                background-color: #fff;
            }
        """

        self.sequence_length_input = QLineEdit("50")
        self.stride_input = QLineEdit("5")
        self.batch_size_input = QLineEdit("32")
        self.epoch_number_input = QLineEdit("10")

        self.learning_rate_combo = QComboBox()
        self.learning_rate_combo.addItems(["0.01", "0.001", "0.0001", "0.00001"])
        self.learning_rate_combo.setCurrentText("0.001")
        self.learning_rate_combo.setStyleSheet(combo_style)

        for widget in [self.sequence_length_input, self.stride_input, self.batch_size_input, self.epoch_number_input]:
            widget.setStyleSheet(input_style)

        fields = [
            ("Sequence Length", self.sequence_length_input),
            ("Stride", self.stride_input),
            ("Batch Size", self.batch_size_input),
            ("Epoch Number", self.epoch_number_input),
            ("Learning Rate", self.learning_rate_combo)
        ]

        for row, (label_text, input_widget) in enumerate(fields):
            lbl = QLabel(label_text)
            lbl.setStyleSheet(label_style)
            input_widget.setFixedHeight(24)
            hyper_layout.addWidget(lbl, row, 0)
            hyper_layout.addWidget(input_widget, row, 1)

        hyper_box.setLayout(hyper_layout)
        middle_layout.addWidget(hyper_box)

        # ===== Layer Types Section =====
        layer_box = QGroupBox()
        layer_box.setTitle("Layer Types")
        layer_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-style: italic;
                border: 1px solid #dcdcdc;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: black;
            }
        """)

        # Create a scroll area for the layer types
        layer_scroll = QScrollArea()
        layer_scroll.setWidgetResizable(True)
        layer_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #dceaf7;
                min-height: 20px;
            }
        """)
        layer_scroll.setMinimumHeight(150)  

        layer_scroll_content = QWidget()
        self.layer_scroll_layout = QVBoxLayout(layer_scroll_content)
        self.layer_scroll_layout.setContentsMargins(0, 10, 8, 10)
        self.layer_scroll_layout.setSpacing(5)

        # Add "Add Layer" button
        add_layer_button = QPushButton("Add Layer")
        add_layer_button.setStyleSheet("""
            QPushButton {
                background-color: #dceaf7;
                font-weight: bold;
                font-size: 12px;
                border: 2px solid #dcdcdc;
                padding: 5px 0px;
                border-radius: 20px;
                min-width: 100px;
                max-width: 100px;
            }
            QPushButton:hover {
                background-color: #cce0f5;
            }
        """)
        add_layer_button.clicked.connect(self.add_layer_combo_row)

        # Set the scroll area content
        layer_scroll.setWidget(layer_scroll_content)

        # Add the scroll area and button to the group box
        layer_box_layout = QVBoxLayout()
        layer_box_layout.addWidget(layer_scroll)
        layer_box_layout.addWidget(add_layer_button, 0, Qt.AlignCenter)  
        layer_box.setLayout(layer_box_layout)

        middle_layout.addWidget(layer_box)

        # Save button
        save_btn = QPushButton("Save Parameters")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #dceaf7;
                font-weight: bold;
                font-size: 12px;
                border: 2px solid #dcdcdc;
                padding: 8px 0px;
            }
            QPushButton:hover {
                background-color: #cce0f5;
            }
        """)
        save_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        save_btn.clicked.connect(self.save_hyperparameters)
        middle_layout.addWidget(save_btn)

        # Summary text area
        self.nn_text_edit = QTextEdit()
        self.nn_text_edit.setReadOnly(True)
        self.nn_text_edit.setStyleSheet("border: 1px solid #dcdcdc; background-color: white; padding: 4px;")
        middle_layout.addWidget(self.nn_text_edit, 1)

        parent_layout.addWidget(middle_panel, 1)

    def create_evaluation_panel(self, parent_layout):
        """Crée le panneau d’évaluation avec placeholder et figure canvas."""

        # ➤ Placeholder pour l'état initial
        self.eval_plot_placeholder = QLabel("Plot will appear here")
        self.eval_plot_placeholder.setAlignment(Qt.AlignCenter)
        self.eval_plot_placeholder.setStyleSheet("""
            background: white;
            border: 1px solid #dcdcdc;
            color: gray;
            font-style: italic;
        """)

        # ➤ Canvas pour afficher les plots (taille augmentée)
        self.eval_canvas = FigureCanvas(Figure(figsize=(6, 5)))  # ✅ Taille plus grande
        self.eval_canvas.setStyleSheet("background: white; border: 1px solid #dcdcdc;")

        # ➤ Stack layout pour alterner placeholder / canvas
        self.eval_stack = QStackedLayout()
        self.eval_stack.addWidget(self.eval_plot_placeholder)  # index 0
        self.eval_stack.addWidget(self.eval_canvas)            # index 1
        self.eval_stack.setCurrentIndex(0)  # Par défaut = placeholder

        # ➤ Conteneur pour empiler dans le layout principal
        container = QWidget()
        container.setLayout(self.eval_stack)

        # ➤ Ajoute à la colonne de droite
        parent_layout.addWidget(container, 1)

        # ➤ Optionnel : préparer layout plus propre
        self.eval_canvas.figure.tight_layout()


    def create_training_controls(self):
        """Create the training control buttons at the bottom."""

        proc_row = QHBoxLayout()

        # ✅ Processor label (dynamique)
        self.processor_label = QLabel(f"Current processor: {self.get_processor_name()}")
        self.processor_label.setStyleSheet("font-size: 8pt; font-family: 'Segoe UI'; color: #222;")
        proc_row.addWidget(self.processor_label, alignment=Qt.AlignLeft)

        # ➤ Training control buttons
        button_layout = QHBoxLayout()
        for name in ["Start", "Stop", "Save Model"]:
            btn = QPushButton(name)
            btn.setStyleSheet("""
                background-color: #dceaf7; 
                border: 2px solid white; 
                font-weight: bold;
            """)
            btn.setMinimumWidth(100)
            button_layout.addWidget(btn)

            # ➤ Connect buttons
            if name == "Start":
                btn.clicked.connect(self.start_training)
            elif name == "Stop":
                btn.clicked.connect(self.stop_training)
            elif name == "Save Model":
                btn.clicked.connect(self.save_model)  # Facultatif si défini

        proc_row.addStretch()
        proc_row.addLayout(button_layout)
        proc_row.addStretch()

        # ➤ Final container with margins
        proc_container = QWidget()
        proc_layout = QVBoxLayout(proc_container)
        proc_layout.setContentsMargins(7, 0, 0, 0) 
        proc_layout.addLayout(proc_row)

        self.main_layout.addWidget(proc_container)

    def get_processor_name(self):
        """Retourne 'CUDA~~' si GPU détecté, sinon 'CPU~~'."""
        import tensorflow as tf
        return "CUDA~~" if tf.config.list_physical_devices('GPU') else "CPU~~"
    
    def create_training_monitor(self):
        """Create the training progress monitor section."""

        monitor_frame = QFrame()
        monitor_frame.setStyleSheet("""
            background: white;
            border: 1px solid #dcdcdc;
        """)
        monitor_layout = QVBoxLayout(monitor_frame)

        # ✅ Zone texte pour les logs
        self.training_monitor_text = QTextEdit()
        self.training_monitor_text.setReadOnly(True)
        self.training_monitor_text.setStyleSheet("""
            font-family: Consolas;
            font-size: 11px;
            border: none;
            background-color: #f9f9f9;
        """)
        monitor_layout.addWidget(self.training_monitor_text)

        # ✅ Barre de progression
        self.training_progress_bar = QProgressBar()
        self.training_progress_bar.setValue(0)
        self.training_progress_bar.setTextVisible(True)
        self.training_progress_bar.setFormat("Loading bar (~%)")
        monitor_layout.addWidget(self.training_progress_bar)

        # Titre
        training_monitor_container = QWidget()
        training_monitor_layout = QVBoxLayout(training_monitor_container)
        training_monitor_layout.setContentsMargins(7, 0, 0, 0)
        training_monitor_layout.addWidget(
            QLabel("Training Monitor", parent=self, styleSheet="font-weight: bold; font-style: italic;")
        )

        self.main_layout.addWidget(training_monitor_container)

        monitor_container = QFrame()
        monitor_container_layout = QVBoxLayout(monitor_container)
        monitor_container_layout.setContentsMargins(7, 0, 7, 7)
        monitor_container_layout.addWidget(monitor_frame)

        self.main_layout.addWidget(monitor_container)

    def create_status_bar(self):
        """Create and configure the status bar."""
        status_label = QLabel("Data Monitoring Software version 1.0.13")
        status_label.setAlignment(Qt.AlignCenter)
        status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.statusBar().addPermanentWidget(status_label, 1)

    def update_progress(self, active_step, completed_steps=None):
        """
        Update the progress indicator showing workflow steps.
        
        Args:
            active_step (str): The currently active step (highlighted in orange)
            completed_steps (list): List of completed steps (shown in green)
        """
        steps = ["Dataset Builder", "NN Designer", "NN Evaluator"]
        completed_steps = completed_steps or []

        parts = []
        for step in steps:
            if step in completed_steps:
                # Completed steps in green bold
                parts.append(f'<span style="color: green; font-weight:bold;">{step}</span>')
            elif step == active_step:
                # Active step in orange bold
                parts.append(f'<span style="color: orange; font-weight:bold;">{step}</span>')
            else:
                # Other steps in normal style
                parts.append(step)

        text = " → ".join(parts)
        self.top_right_label.setText(f"Progress Statement : {text}")

    def go_back(self):
        """HARRY: Enhanced go_back method with proper parameter passing"""
        try:
            # Stop any running training thread
            if hasattr(self, "train_thread") and self.train_thread and self.train_thread.isRunning():
                self.train_thread.stop()
                self.train_thread.wait()

            # Save current state before navigating
            current_state = self.get_saved_state()

            # Import here to avoid circular imports
            from windows.dataset_builder_window import DatasetBuilderWindow

            # Create dataset builder window with both state and start_window_ref
            self.dataset_builder_window = DatasetBuilderWindow(
                start_window_ref=None,  # HARRY: Pass None since we don't need it for back navigation
                saved_state=current_state
            )
            
            # Hide current window and show new one
            self.hide()
            self.dataset_builder_window.showMaximized()

        except Exception as e:
            QMessageBox.critical(self, "Navigation Error", f"Failed to navigate back: {str(e)}")
            print(f"Navigation error details: {e}")  # HARRY: Added for debugging
            self.show()  # Show the current window again if navigation failed

    def start_training(self):
        """HARRY: Enhanced training start with better state and thread management"""
        try:
            # Check if training is already running
            if self.is_training:
                QMessageBox.warning(self, "Training in Progress", "Training is already running!")
                return

            # Handle previous thread cleanup
            if hasattr(self, "train_thread") and self.train_thread and self.train_thread.isRunning():
                self.train_thread.stop()
                if not self.train_thread.wait(1000):  # Wait 1 second
                    QMessageBox.warning(self, "Warning", "Previous training is still stopping, please wait")
                    return

            # Check if retraining
            if progress_state.nn_designed:
                reply = QMessageBox.question(
                    self,
                    "Retrain Model?",
                    "You have already trained the model. Restarting training will reset evaluator progress. Do you want to continue?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
                if hasattr(self, "eval_stack"):
                    self.eval_stack.setCurrentIndex(0)

            # Reset training state
            self.is_training = True
            self.training_completed = False
            self.training_stopped = False

            # Validate layers and hyperparameters
            params, selected_files = self.get_training_parameters()
            if params is None or not params["layers"]:
                QMessageBox.warning(self, "Missing Layers", "Please add at least one layer before starting training.")
                self.is_training = False
                return

            if not hasattr(self, "hyperparams_saved") or not self.hyperparams_saved:
                QMessageBox.warning(self, "Missing Parameters", "Please save your hyperparameters before starting training.")
                self.is_training = False
                return

            if not selected_files:
                QMessageBox.warning(self, "No Files Selected", "Please select at least one .h5 file to train on.")
                self.is_training = False
                return

            # Clear previous UI state
            self.training_monitor_text.clear()
            self.training_progress_bar.setValue(0)

            # Load and prepare data
            X_all, y_all, time_all = [], [], []
            for fname in selected_files:
                print(f"[DEBUG] About to process: {fname}")
                # Utilise le mapping pour obtenir le chemin absolu correct
                abs_path = self.file_name_to_path.get(fname)
                print(f"[DEBUG] Absolute path: {abs_path}")
                print(f"[DEBUG] File exists: {os.path.exists(abs_path) if abs_path else 'N/A'}")
                try:
                    X, y, full_time = load_and_preprocess_data(
                        abs_path,
                        window_size=params["sequence_length"],
                        stride=params["stride"]
                    )
                    print(f"[DEBUG] Loaded: {abs_path} X={X.shape if X is not None else None}")
                except Exception as e:
                    print(f"[EXCEPTION] Error loading {abs_path}: {e}")
                    continue

                # Validate data
                if X is None or y is None or X.size == 0 or y.size == 0:
                    print(f"⛔ Empty data ignored: {fname}")
                    continue
                if full_time is None or not hasattr(full_time, "shape") or full_time.size == 0:
                    print(f"⛔ Missing or empty time data for: {fname}")
                    continue
                if X.ndim < 2 or y.ndim < 1:
                    print(f"⛔ Incorrect dimensions: {fname}")
                    continue

                # Add valid data
                X_all.append(X)
                y_all.append(y)
                time_all.append(full_time)
                print(f"✅ Loaded: {fname} → X={X.shape}, y={y.shape}, time={full_time.shape}")

            # Validate loaded data
            if not X_all or not y_all:
                self.is_training = False
                raise ValueError("No files could be loaded correctly. Check their content.")

            # Combine data
            X = np.concatenate(X_all)
            y = np.concatenate(y_all)
            clean_time_all = [t.squeeze() if t.ndim > 1 else t for t in time_all if t is not None]
            time = np.concatenate(clean_time_all) if clean_time_all else np.arange(len(y))

            # Verify data shapes
            print("Shapes → X:", X.shape, "| y:", y.shape, "| time:", time.shape)
            assert X.shape[0] == y.shape[0] == time.shape[0], "❌ Inconsistency between X, y and time"

            # Split data
            x_train_val, X_test, y_train_val, y_test, time_train_val, test_time = train_test_split(
                X, y, time, test_size=0.2, random_state=42, shuffle=False
            )
            X_train, X_val, y_train, y_val, time_train, time_val = train_test_split(
                x_train_val, y_train_val, time_train_val, test_size=0.25, random_state=42, shuffle=False
            )

            # Initialize and start training thread
            self.train_thread = TrainingThread(
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                params=params,
                test_time=test_time
            )

            # Connect signals
            self.train_thread.setParent(self)
            self.train_thread.training_finished.connect(self.on_training_finished)
            self.train_thread.training_log.connect(self.append_training_log)
            self.train_thread.finished.connect(self.on_thread_finished)

            # Start training
            progress_state.nn_designed = False
            self.train_thread.start()
            self.nn_text_edit.setText("🧠 Training started...")

        except Exception as e:
            self.is_training = False
            QMessageBox.critical(self, "Training Error", str(e))

    def append_training_log(self, message):
        self.training_monitor_text.append(message)
        self.training_monitor_text.moveCursor(QTextCursor.End)

        # ✅ Met à jour la barre de progression si possible
        match = re.search(r"Epoch\s+\[(\d+)/(\d+)\]", message)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            percent = int((current / total) * 100)
            self.training_progress_bar.setValue(percent)

    def on_training_finished(self, model, history, test_results):
        """HARRY: Enhanced training completion handler with robust error handling and safe header management"""
        try:
            # Handle stopped training case first
            if self.training_stopped:
                self.training_progress_bar.setValue(0)
                self.nn_text_edit.setText("🛑 Training was manually stopped.")
                progress_state.nn_designed = False
                self.eval_stack.setCurrentIndex(0)  # Return to placeholder
                # Reset states
                self.is_training = False
                self.training_completed = False
                return  # Return immediately if training was stopped
            # Safely store results first
            self.last_training_params, self.last_training_files = self.get_training_parameters()
            self.trained_model = model
            self.training_history = history
            self.test_results = test_results
            self.training_completed = True
            self.training_stopped = False  # <-- AJOUTE CETTE LIGNE
            self.state["training_completed"] = True
            self.state["training_stopped"] = False
            
            # Then update flags and UI
            self.is_training = False
            self.training_progress_bar.setValue(100)

            # Now handle thread cleanup safely
            if hasattr(self, "train_thread") and self.train_thread is not None:
                try:
                    if self.train_thread.isRunning():
                        print("Cleaning up training thread...")
                        self.train_thread.quit()
                        self.train_thread.wait()
                except Exception as thread_e:
                    print(f"Thread cleanup error: {thread_e}")
                finally:
                    self.train_thread = None

            # Update UI with results
            try:
                acc = test_results["accuracy"]
                report = test_results["report"]
                self.nn_text_edit.setText(f"✅ Training complete\n\nAccuracy: {acc:.3f}\n\n{report}")
                self.plot_training_curves(history)

                # Update global progress state
                progress_state.nn_designed = True
                progress_state.trained_model = model
                progress_state.training_history = history
                progress_state.test_results = test_results
                progress_state.training_started = True
                self.state["training_started"] = True

                # Save test results
                if test_results and "true_labels" in test_results and "predictions" in test_results:
                    progress_state.test_results["y_true"] = test_results["true_labels"]
                    progress_state.test_results["y_pred"] = test_results["predictions"]
                else:
                    progress_state.test_results["y_true"] = []
                    progress_state.test_results["y_pred"] = []

                # Show completion message
                QMessageBox.information(self, "Training Complete", 
                    f"Model trained successfully.\nTest accuracy: {acc:.2%}\n\n"
                    f"You can now view detailed results in the 'NN Evaluator' tab.")

            except Exception as ui_e:
                print(f"UI update error: {ui_e}")
                raise

            # Enable tabs only if header exists
            if hasattr(self, 'header') and self.header is not None:
                self.header.update_active_tab()
            else:
                print("Warning: Header not available")

        except Exception as e:
            print(f"Error in training completion handler: {e}")
            self.is_training = False
            QMessageBox.warning(self, "Training Error", 
                "Error processing training results. Some features may be unavailable.")
            
            # Try to recover the header if that's the issue
            if not hasattr(self, 'header') or self.header is None:
                try:
                    self.create_header()
                except Exception as header_e:
                    print(f"Failed to recreate header: {header_e}")

    def on_thread_finished(self):
        """HARRY: Handle thread completion"""
        self.is_training = False
        if not self.training_completed:  # If training was stopped prematurely
            self.append_training_log("Training stopped.")
            self.training_progress_bar.setValue(0)

    def stop_training(self):
        """HARRY: Enhanced training stop with proper state management"""
        if not self.is_training:
            return
        
        reply = QMessageBox.question(
            self, 
            "Stop Training", 
            "Are you sure you want to stop the training?",
            QMessageBox.Yes | QMessageBox.No,
            defaultButton=QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                self.training_stopped = True
                self.cleanup_training_thread()
                self.is_training = False
                self.training_completed = False
                
                # Update UI
                self.training_progress_bar.setValue(0)
                self.append_training_log("Training stopped by user.")
                self.nn_text_edit.setText("🛑 Training was manually stopped.")
                
                QMessageBox.information(self, "Training Stopped", 
                    "Training process has been stopped.")
                    
            except Exception as e:
                print(f"Error stopping training: {e}")
                QMessageBox.warning(self, "Error", 
                    f"Error while stopping training: {str(e)}")

    def auto_select_files(self):
        """Automatically select a percentage of files for training."""
        try:
            # Parse percentage input
            percentage = float(self.training_percentage_input.text().strip().replace('%', ''))
            if not (0 < percentage <= 100):
                raise ValueError
                
            # Calculate number of files to select
            total = self.file_list.count()
            to_check = round((percentage / 100) * total)
            
            # Get all items and randomly select the specified number
            all_items = [self.file_list.item(i) for i in range(total)]
            selected = random.sample(all_items, to_check)
            
            # Update check states
            for item in all_items:
                item.setCheckState(Qt.Checked if item in selected else Qt.Unchecked)
                
        except ValueError:
            QMessageBox.warning(
                self, 
                "Invalid Input", 
                "Please enter a valid percentage between 0 and 100."
            )
    def get_selected_files(self):
        """Retourne la liste des chemins absolus des fichiers sélectionnés."""
        selected_files = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.checkState() == Qt.Checked:
                filename = item.text()
                abs_path = os.path.join(self.dataset_path, filename)
                selected_files.append(abs_path)
        return selected_files

    def populate_file_list(self):
        self.file_list.clear()
        self.file_name_to_path = {}
        if os.path.exists(self.dataset_path):
            # Trouver la racine NN_Project
            def find_nn_project_root(path):
                path = os.path.abspath(path)
                while path and os.path.basename(path) != "NN_Project":
                    parent = os.path.dirname(path)
                    if parent == path:
                        return None
                    path = parent
                return path
            project_root = find_nn_project_root(self.dataset_path)
            if not project_root:
                QMessageBox.critical(self, "Erreur", "Impossible de trouver le dossier NN_Project.")
                return
            for filename in os.listdir(self.dataset_path):
                if filename.endswith(".h5") and filename.startswith("filtered_"):
                    abs_path = os.path.join(self.dataset_path, filename)
                    rel_path = os.path.relpath(abs_path, project_root)
                    item = QListWidgetItem(filename)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Unchecked)
                    item.setData(Qt.UserRole, rel_path)  # clé = chemin relatif
                    self.file_list.addItem(item)
                    self.file_name_to_path[rel_path] = abs_path  # clé = chemin relatif

    def populate_file_list_with_paths(self, file_paths, checked_files=None):
        self.file_list.clear()
        print("Files received in NN Designer:", file_paths)
        checked_files_set = set([os.path.normpath(f) for f in (checked_files or [])])
        self.file_name_to_path = {}

        # 1. Trouver la racine du projet
        def find_nn_project_root(path):
            path = os.path.abspath(path)
            while path and os.path.basename(path) != "NN_Project":
                parent = os.path.dirname(path)
                if parent == path:
                    return None
                path = parent
            return path

        project_root = find_nn_project_root(self.dataset_path)
        if not project_root:
            QMessageBox.critical(self, "Erreur", "Impossible de trouver le dossier NN_Project.")
            return

        for rel_path in file_paths:
            rel_path = str(rel_path) 
            # rel_path peut être 'filtered_xxx.h5' ou 'subfolder/filtered_xxx.h5'
            abs_path = os.path.normpath(os.path.join(project_root, rel_path))
            filename = os.path.basename(rel_path)
            item = QListWidgetItem(filename)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setData(Qt.UserRole, rel_path)  # On stocke le chemin relatif pour la logique
            # On check si le chemin relatif est dans checked_files_set
            item.setCheckState(Qt.Checked if os.path.normpath(rel_path) in checked_files_set else Qt.Unchecked)
            self.file_list.addItem(item)
            self.file_name_to_path[rel_path] = abs_path  # clé = chemin relatif

    def restore_checkboxes(self):
        """Restore checkbox states for file list."""
        selected_files_set = set(self.state.get("selected_files", []))
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            rel_path = item.data(Qt.UserRole)
            item.setCheckState(Qt.Checked if rel_path in selected_files_set else Qt.Unchecked)
    
    def plot_training_curves(self, history):
            self.eval_stack.setCurrentIndex(1)

            self.eval_canvas.figure.clf()
            ax1 = self.eval_canvas.figure.add_subplot(2, 1, 1)
            ax2 = self.eval_canvas.figure.add_subplot(2, 1, 2)

            # Access history through the history attribute
            history_dict = history.history

            # Loss
            ax1.plot(history_dict['loss'], label='Train Loss', color='blue')
            ax1.plot(history_dict['val_loss'], label='Validation Loss', color='orange')
            ax1.set_title("Training and Validation Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True)

            # Accuracy (handle both 'accuracy' and 'sparse_categorical_accuracy')
            acc_key = None
            val_acc_key = None
            if 'accuracy' in history_dict:
                acc_key = 'accuracy'
                val_acc_key = 'val_accuracy'
            elif 'sparse_categorical_accuracy' in history_dict:
                acc_key = 'sparse_categorical_accuracy'
                val_acc_key = 'val_sparse_categorical_accuracy'

            if acc_key and val_acc_key and acc_key in history_dict and val_acc_key in history_dict:
                ax2.plot(history_dict[acc_key], label='Train Accuracy', color='blue')
                ax2.plot(history_dict[val_acc_key], label='Validation Accuracy', color='orange')
                ax2.set_title("Training and Validation Accuracy")
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Accuracy")
                ax2.legend()
                ax2.grid(True)
            else:
                ax2.text(0.5, 0.5, "No accuracy metric available in history.",
                        ha='center', va='center', fontsize=12)
                ax2.set_axis_off()

            self.eval_canvas.figure.tight_layout()
            self.eval_canvas.draw()

    def save_model(self):
        """Sauvegarde le modèle et toutes les données dans un fichier .h5"""
        try:
            import h5py
            import json
            import numpy as np
            import os
            print("trained_model:", self.trained_model)
            print("training_history:", self.training_history)
            print("training_completed:", self.training_completed)
            # Vérifier si on a un modèle entraîné
            if (
                not hasattr(self, "trained_model") or self.trained_model is None
                or not hasattr(self, "training_history") or self.training_history is None
                or not getattr(self, "training_completed", False)
            ):
                QMessageBox.warning(self, "Nothing to Save", "No trained model found or training not completed. Please train before saving.")
                return
                    
            # Obtenir les paramètres actuels
            current_params, current_files = self.get_training_parameters()
            
            if hasattr(self, 'last_training_params'):  # Si on a entraîné
                reference_params = self.last_training_params
                reference_files = self.last_training_files
            elif hasattr(self, 'loaded_params'):  # Si on a chargé un modèle
                reference_params = self.loaded_params
                reference_files = self.loaded_files

            if reference_params and reference_files:
                params_changed = current_params != reference_params
                files_changed = set(current_files) != set(reference_files)
            
                if params_changed or files_changed:
                    reply = QMessageBox.warning(
                        self,
                        "Parameters Modified",
                        "The current parameters or file selection differ from those used for training.\n"
                        "The saved model will not reflect these changes.\n\n"
                        "Do you want to:\n"
                        "• Save anyway (with old parameters)\n"
                        "• Train new model first",
                        QMessageBox.Save | QMessageBox.Cancel | QMessageBox.Retry,
                        QMessageBox.Retry
                    )
                    
                    if reply == QMessageBox.Retry:
                        self.start_training()
                        return
                    elif reply == QMessageBox.Cancel:
                        return
                    # Si Save, continuer avec la sauvegarde

            save_path = QFileDialog.getSaveFileName(self, "Save Model", "", "H5 Files (*.h5)")[0]
            if not save_path:
                return

            if not save_path.endswith('.h5'):
                save_path += '.h5'

            # Vérifier les permissions d'écriture
            save_dir = os.path.dirname(save_path)
            if not os.access(save_dir, os.W_OK):
                QMessageBox.critical(self, "Error", f"No write permission in directory:\n{save_dir}")
                return

            with h5py.File(save_path, 'w') as hf:
                # 1. Sauvegarder la configuration du modèle
                model_config = self.trained_model.get_config()
                hf.create_dataset('model_config', data=json.dumps(model_config).encode('utf-8'))
                
                # 2. Sauvegarder les poids du modèle
                weights_group = hf.create_group('model_weights')
                for layer in self.trained_model.layers:
                    layer_group = weights_group.create_group(layer.name)
                    weights = layer.get_weights()
                    for i, weight in enumerate(weights):
                        layer_group.create_dataset(f'weight_{i}', data=weight)

                # 3. Sauvegarder les paramètres d'entraînement
                training_params = {
                    'optimizer': self.state.get('optimizer', ''),
                    'loss_function': self.state.get('loss_function', ''),
                    'hyperparameters': self.state.get('hyperparameters', {})
                }
                hf.create_dataset('training_params', data=json.dumps(training_params).encode('utf-8'))

                # 4. Sauvegarder l'historique
                history_group = hf.create_group('history')
                for key, value in self.training_history.history.items():
                    history_group.create_dataset(key, data=np.array(value))

                # 5. Sauvegarder les résultats de test
                results_group = hf.create_group('test_results')
                for key, value in self.test_results.items():
                    if isinstance(value, (list, np.ndarray)):
                        results_group.create_dataset(key, data=np.array(value))
                    else:
                        results_group.create_dataset(f"{key}_json", 
                                                  data=json.dumps(value).encode('utf-8'))

                # 6. Sauvegarder la liste des fichiers
                files_group = hf.create_group('files')
                selected_files = []
                checked_files = []
                # ➤ Trouver la racine NN_Project
                project_root = self.find_project_root()
                for i in range(self.file_list.count()):
                    item = self.file_list.item(i)
                    abs_path = os.path.abspath(os.path.join(self.dataset_path, item.text()))
                    # ➤ Chemin relatif à NN_Project
                    rel_path = os.path.relpath(abs_path, project_root)
                    selected_files.append(rel_path)
                    if item.checkState() == Qt.Checked:
                        checked_files.append(rel_path)
                files_group.create_dataset('selected_files', data=[str(f).encode('utf-8') for f in selected_files])
                files_group.create_dataset('checked_files', data=[str(f).encode('utf-8') for f in checked_files])

            QMessageBox.information(self, "Success", 
                                  f"Model and all related data saved successfully to:\n{save_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save model and data:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def find_project_root(self):
        """Trouve le dossier NN_Project à partir du chemin courant."""
        path = os.path.abspath(self.dataset_path)
        while path and os.path.basename(path) != "NN_Project":
            parent = os.path.dirname(path)
            if parent == path:  # On est à la racine
                print("NN_Project folder not found in path hierarchy!")
                return None
            path = parent
        return path

    def cleanup_training_thread(self):
        """HARRY: Enhanced cleanup with comprehensive state management and error handling"""
        if hasattr(self, "train_thread") and self.train_thread is not None:
            try:
                # First attempt graceful stop
                self.train_thread.stop()
                
                if self.train_thread.isRunning():
                    # Request thread termination
                    self.train_thread.quit()
                    
                    # Wait with timeout (5 seconds)
                    if not self.train_thread.wait(5000):
                        print("Warning: Training thread timeout - forcing termination")
                        self.train_thread.terminate()
                        self.train_thread.wait(1000)  # Short wait after force
                
                # Cleanup Qt object
                self.train_thread.deleteLater()
                self.train_thread = None
                
                # Reset state flags
                self.is_training = False
                if not self.training_completed:
                    self.training_stopped = True
                        
            except Exception as e:
                print(f"Error during thread cleanup: {e}")
                # Try force cleanup in case of error
                try:
                    if self.train_thread and self.train_thread.isRunning():
                        self.train_thread.terminate()
                        self.train_thread = None
                except:
                    pass

    def closeEvent(self, event):
        """HARRY: Enhanced window closing handler"""
        # Clean up training thread
        self.cleanup_training_thread()
        
        # Unregister from active windows
        QApplication.instance().unregister_window(self)
        
        # Accept the close event
        event.accept()

    def restore_checkboxes(self):
        """Restore checkbox states for file list."""
        checked_files_set = set(self.state.get("selected_files", []))
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            rel_path = item.data(Qt.UserRole)
            item.setCheckState(Qt.Checked if rel_path in checked_files_set else Qt.Unchecked)
    def set_tabs_enabled(self, enabled: bool):
        if hasattr(self, "header"):
            for tab in self.header.tabs.values():
                tab_label = tab.layout().itemAt(0).widget()
                tab_label.setEnabled(enabled)

    def preserve_state_on_navigation(self):
        """Préserve l'état actuel lors de la navigation."""
        current_state = self.get_saved_state()
        current_state.update({
            "current_page_index": self.stacked_widget.currentIndex(),
            "eval_plot_state": self.eval_stack.currentIndex(),
            "training_log": self.training_monitor_text.toPlainText(),
            "progress_value": self.training_progress_bar.value()
        })
        return current_state

    def cleanup_training_thread(self):
        """
        HARRY: Enhanced cleanup of training thread
        Ensures thread is properly stopped and cleaned up
        """
        if hasattr(self, "train_thread") and self.train_thread is not None:
            try:
                # Stop the training
                self.train_thread.stop()
                
                # Force quit if still running
                if self.train_thread.isRunning():
                    self.train_thread.quit()
                
                # Wait with timeout
                if not self.train_thread.wait(3000):  # 3 second timeout
                    print("Warning: Training thread did not stop properly")
                
                # Ensure thread is finished
                self.train_thread.terminate()
                self.train_thread = None
            except Exception as e:
                print(f"Error during thread cleanup: {e}")

def get_project_root():
    # Cherche le dossier NN_Project à partir du fichier courant
    path = os.path.abspath(__file__)
    while path and os.path.basename(path) != "NN_Project":
        path = os.path.dirname(path)
    return path

def get_absolute_paths(rel_paths):
    project_root = get_project_root()
    return [os.path.join(project_root, rel_path) for rel_path in rel_paths]



