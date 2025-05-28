from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QGridLayout,
    QPushButton, QComboBox, QTextEdit, QListWidget, QListWidgetItem, QProgressDialog,
    QFrame, QMessageBox, QGroupBox, QScrollArea, QSizePolicy, QProgressBar,QStackedLayout, QFileDialog, 

)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIntValidator, QFont, QTextCursor
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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from windows.nn_evaluator_window import NeuralNetworkEvaluator
class TrainingThread(QThread):
    training_finished = pyqtSignal(object, object, object)
    training_log = pyqtSignal(str)

    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, params, parent=None):
        super().__init__(parent)
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.params = params
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        from training_core.training_utils import train_model

        def log_callback(msg):
            self.training_log.emit(msg)

        model, history, test_results, _ = train_model(
            self.X_train, self.X_val, self.X_test,
            self.y_train, self.y_val, self.y_test,
            model_type=self.params["layer_types"],
            optimizer_name=self.params["optimizer"],
            loss_name=self.params["loss_function"],
            epochs=self.params["epochs"],
            learning_rate=self.params["learning_rate"],
            batch_size=self.params["batch_size"],
            num_layers=len(self.params["layer_types"]),
            sequence_length=self.params["sequence_length"],
            verbose=1,
            stop_flag_getter=lambda: self._stop_flag,
            log_callback=log_callback
        )
        self.training_finished.emit(model, history, test_results)
    
class NeuralNetworkDesignerWindow(QMainWindow):
    def __init__(self, dataset_path=None, saved_state=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.setWindowTitle("Data Monitoring Software")
        self.resize(1000, 700)
        # --- STATE MANAGEMENT ---
        self.state = {
            "optimizer": "Adam",
            "loss_function": None,
            "hyperparameters": {
                "layer_types": [],
                "sequence_length": "50",
                "stride": "5",
                "batch_size": "32",
                "epoch_number": "50",
                "learning_rate": "0.001"
            },
            "selected_files": [],
            "dataset_path": dataset_path,
            "summary_text": "No parameters saved yet"
        }

        self.init_ui()
        # À LA FIN SEULEMENT :
        if saved_state:
            self.state = saved_state
        self.restore_state()
        self.update_summary_display()

    def restore_state(self):
        """Restore all UI elements from self.state."""
        self.optimizer_combo.setCurrentText(self.state.get("optimizer", "Adam"))
        self.classification_loss_combo.setCurrentText(self.state.get("loss_function", "CrossEntropyLoss"))
        self.regression_loss_combo.setCurrentText(self.state.get("loss_function", "HuberLoss"))
        hp = self.state.get("hyperparameters", {})
        self.sequence_length_input.setText(str(hp.get("sequence_length", 50)))
        self.stride_input.setText(str(hp.get("stride", 5)))
        self.batch_size_input.setText(str(hp.get("batch_size", 32)))
        self.epoch_number_input.setText(str(hp.get("epoch_number", 50)))
        self.learning_rate_combo.setCurrentText(str(hp.get("learning_rate", 0.001)))
        # Restore layer types
        self.clear_layer_combos()
        for layer in hp.get("layer_types", []):
            self.add_layer_combo_row(layer)
        # Restore optimizer
        if self.optimizer_combo.findText(self.state.get("optimizer", "Adam")) >= 0:
            self.optimizer_combo.setCurrentText(self.state.get("optimizer", "Adam"))
        # Restore loss function
        if self.state.get("loss_function"):
            if self.classification_loss_combo.findText(self.state["loss_function"]) >= 0:
                self.classification_loss_combo.setCurrentText(self.state.get("loss_function", "CrossEntropyLoss"))
                self.regression_loss_combo.setCurrentIndex(-1)
            elif self.regression_loss_combo.findText(self.state["loss_function"]) >= 0:
                self.regression_loss_combo.setCurrentText(self.state.get("loss_function", "MSELoss"))
                self.classification_loss_combo.setCurrentIndex(-1)
        QTimer.singleShot(100, self.restore_checkboxes)

        # Restaure les plots si l'historique d'entraînement est présent
        if hasattr(self, "training_history") and self.training_history:
            self.plot_training_curves(self.training_history)
        elif "training_history" in self.state and self.state["training_history"]:
            self.training_history = self.state["training_history"]
            self.plot_training_curves(self.training_history)
        # Tu peux aussi restaurer self.trained_model et self.test_results si tu veux les réutiliser

    def clear_layer_combos(self):
        """Remove all layer combo rows."""
        for row_layout in getattr(self, "layer_combo_rows", []):
            for i in reversed(range(row_layout.count())):
                widget = row_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            self.layer_combo_layout.removeItem(row_layout)
        self.layer_type_combos = []
        self.layer_combo_rows = []

    def add_layer_combo_row(self, preset_value=None):
        row_layout = QHBoxLayout()
        combo = QComboBox()
        combo.addItems(["LSTM", "GRU", "RNN", "Transformer", "TinyTransformer"])
        if preset_value and combo.findText(preset_value) >= 0:
            combo.setCurrentText(preset_value)
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

        remove_btn = QPushButton("🗑")
        remove_btn.setFixedWidth(30)
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #f9dcdc;
                border: 1px solid #dcdcdc;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f5bebe;
            }
        """)

        def remove_row():
            self.layer_type_combos.remove(combo)
            self.layer_combo_rows.remove(row_layout)
            for i in reversed(range(row_layout.count())):
                widget = row_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            self.layer_combo_layout.removeItem(row_layout)

        remove_btn.clicked.connect(remove_row)

        row_layout.addWidget(combo)
        row_layout.addWidget(remove_btn)

        self.layer_type_combos.append(combo)
        self.layer_combo_rows.append(row_layout)
        self.layer_combo_layout.addLayout(row_layout)

    def get_layer_types(self):
        return [combo.currentText() for combo in self.layer_type_combos]

    def get_saved_state(self):
        """Return the complete current state."""
        self.state["hyperparameters"] = {
            "layer_types": self.get_layer_types(),
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
        self.state["selected_files"] = [
            self.file_list.item(i).text()
            for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == Qt.Checked
        ]
        self.state["training_history"] = getattr(self, "training_history", None)
        self.state["test_results"] = getattr(self, "test_results", None)
        self.state["trained_model"] = getattr(self, "trained_model", None)
        return self.state

    def update_summary_display(self):
        """Update the summary text area with current state."""
        hp = self.state.get("hyperparameters", {})
        optimizer = self.state.get("optimizer", "Adam")
        loss_function = self.state.get("loss_function", "HuberLoss")
        layer_types = hp.get("layer_types", [])
        sequence_length = hp.get("sequence_length", 50)
        stride = hp.get("stride", 5)
        batch_size = hp.get("batch_size", 32)
        epoch_number = hp.get("epoch_number", 50)
        learning_rate = hp.get("learning_rate", 0.001)

        summary = (
            f"Hyperparameters saved:\n"
            f"• Optimizer: {optimizer}\n"
            f"• Loss Function: {loss_function}\n"
            f"• Layer Types: {', '.join(layer_types) if layer_types else 'None'}\n"
            f"• Sequence Length: {sequence_length}\n"
            f"• Stride: {stride}\n"
            f"• Batch Size: {batch_size}\n"
            f"• Epoch Number: {epoch_number}\n"
            f"• Learning Rate: {learning_rate}"
        )
        self.nn_text_edit.setText(summary)

    def get_training_parameters(self):
        """Retourne tous les paramètres nécessaires à l'entraînement."""
        hp = self.state["hyperparameters"]
        print("DEBUG layer_types:", hp.get("layer_types", []))  # Ajoute cette ligne
        params = {
            "optimizer": self.state["optimizer"],
            "loss_function": self.state["loss_function"],
            "loss_type": "Classification" if self.classification_loss_combo.currentIndex() >= 0 else "Regression",
            "layer_types": hp.get("layer_types", []),
            "sequence_length": int(hp.get("sequence_length", 50)),
            "stride": int(hp.get("stride", 5)),
            "batch_size": int(hp.get("batch_size", 32)),
            "epochs": int(hp.get("epoch_number", 50)),
            "learning_rate": float(hp.get("learning_rate", 0.001))
        }
        selected_files = [
            self.file_list.item(i).text()
            for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == Qt.Checked
        ]
        return params, selected_files

    def save_hyperparameters(self):
        """Save the hyperparameters entered by the user and update state."""
        # Get values from input fields
        layer_types = self.get_layer_types()  # <-- Assure-toi que ça retourne la bonne liste !
        sequence_length = self.sequence_length_input.text()
        stride = self.stride_input.text()
        batch_size = self.batch_size_input.text()
        epoch_number = self.epoch_number_input.text()
        learning_rate = self.learning_rate_combo.currentText()
        optimizer = self.optimizer_combo.currentText().strip()
        classification_loss = self.classification_loss_combo.currentText().strip()
        regression_loss = self.regression_loss_combo.currentText().strip()

        # Validation
        if not layer_types:
            QMessageBox.warning(self, "Missing Layers", "Please add at least one layer before saving hyperparameters.")
            return
        if not all([sequence_length, stride, batch_size, epoch_number, optimizer, learning_rate]):
            QMessageBox.warning(self, "Incomplete Hyperparameters", "Please fill in all hyperparameters before saving.")
            return
        if (not classification_loss and not regression_loss) or (classification_loss and regression_loss):
            QMessageBox.warning(self, "Invalid Selection", "Please select either a Classification or Regression loss.")
            return

        loss_function = classification_loss if classification_loss else regression_loss
        loss_type = "Classification" if classification_loss else "Regression"

        # Save in state
        self.state["optimizer"] = optimizer
        self.state["loss_function"] = loss_function
        self.state["hyperparameters"] = {
            "layer_types": layer_types,  # <-- C'est cette ligne qui doit contenir la vraie liste
            "sequence_length": sequence_length,
            "stride": stride,
            "batch_size": batch_size,
            "epoch_number": epoch_number,
            "learning_rate": learning_rate
        }

        summary = (
            f"Hyperparameters saved:\n"
            f"• Optimizer: {optimizer}\n"
            f"• Loss Function: {loss_function} ({loss_type})\n"
            f"• Layer Types: {', '.join(layer_types)}\n"
            f"• Sequence Length: {sequence_length}\n"
            f"• Stride: {stride}\n"
            f"• Batch Size: {batch_size}\n"
            f"• Epoch Number: {epoch_number}\n"
            f"• Learning Rate: {learning_rate}"
        )
        self.nn_text_edit.setText(summary)
        self.hyperparams_saved = True
        QMessageBox.information(self, "Hyperparameters Saved", "The hyperparameters have been saved successfully.")

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
        """Create the application header with navigation."""
        header_container = QWidget()
        header_container_layout = QVBoxLayout(header_container)
        header_container_layout.setContentsMargins(0, 0, 0, 0)  
        header_container_layout.setSpacing(0)

        # Add the header widget
        header = Header(active_page="Neural Network Designer", parent_window=self)
        header_container_layout.addWidget(header)

        self.main_layout.addWidget(header_container, alignment=Qt.AlignTop)

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
        back_btn.clicked.connect(self.go_back_to_start)
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
            QComboBox { border: 1px solid #dcdcdc; padding: 4px; background-color: white; }
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
        middle_layout.addWidget(QLabel("<b><i>Neural Network Model</b></i>"))

        # Hyperparameters
        hyper_box = QGroupBox("Hyperparameters")
        hyper_box.setStyleSheet("border: 1px solid #dcdcdc; border-radius: 5px; padding: 15px;")
        hyper_layout = QGridLayout()
        hyper_layout.setContentsMargins(0, 10, 10, 10)
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
        self.epoch_number_input = QLineEdit("50")

        self.learning_rate_combo = QComboBox()
        self.learning_rate_combo.addItems(["0.01", "0.001", "0.0001", "0.00001"])
        self.learning_rate_combo.setCurrentText("0.001")
        self.learning_rate_combo.setStyleSheet("""
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

        # === Layer Types Section ===
        self.layer_type_combos = []  # liste des QComboBox
        self.layer_combo_rows = []   # liste des layouts horizontaux

        layer_section_label = QLabel("<b><i>Layer Types</b></i>")
        layer_section_label.setStyleSheet("font-size: 14px;")
        middle_layout.addWidget(layer_section_label)

        self.layer_combo_layout = QVBoxLayout()
        middle_layout.addLayout(self.layer_combo_layout)

        # Bouton ajouter une couche
        add_layer_button = QPushButton("Add Layer")
        add_layer_button.setStyleSheet("""
            QPushButton {
                background-color: #dceaf7;
                font-weight: bold;
                font-size: 12px;
                border: 1px solid #dcdcdc;
                padding: 4px;
            }
            QPushButton:hover {
                background-color: #cce0f5;
            }
        """)
        add_layer_button.clicked.connect(self.add_layer_combo_row)
        middle_layout.addWidget(add_layer_button)

        # Save button
        save_btn = QPushButton("Save Hyperparameters")
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
        save_btn_layout = QVBoxLayout()
        save_btn_layout.addWidget(save_btn)
        middle_layout.addLayout(save_btn_layout)

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

    def go_back_to_start(self):
        reply = QMessageBox.question(
            self,
            "Confirm Return",
            "Are you sure you want to return to the start? All unsaved progress in Neural Network Designer will be lost.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            return  # Annule le retour

        self.hide()
        if hasattr(self, 'parent_window') and self.parent_window is not None:
            self.parent_window.showMaximized()
        else:
            from windows.start_window import StartWindow
            self.start_window = StartWindow()
            self.start_window.showMaximized()

    def start_training(self):
        """Lance l'entraînement réel du modèle dans un thread."""
        try:
            
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
                
            # 🛑 Vérifie si au moins une couche est définie
            params, selected_files = self.get_training_parameters()
            if params is None or not params["layer_types"]:
                QMessageBox.warning(self, "Missing Layers", "Please add at least one layer before starting training.")
                return

            # 🛑 Vérifie si les hyperparamètres sont définis
            if not hasattr(self, "hyperparams_saved") or not self.hyperparams_saved:
                QMessageBox.warning(self, "Missing Parameters", "Please save your hyperparameters before starting training.")
                return


            # Si un training est déjà en cours, demander confirmation
            if hasattr(self, "train_thread") and self.train_thread.isRunning():
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Confirm Restart")
                msg_box.setText("Training is already running. Do you want to restart it?")
                msg_box.setIcon(QMessageBox.Warning)

                cancel_btn = msg_box.addButton("Cancel", QMessageBox.RejectRole)
                restart_btn = msg_box.addButton("Restart", QMessageBox.AcceptRole)
                msg_box.setDefaultButton(cancel_btn)

                msg_box.exec_()
                if msg_box.clickedButton() != restart_btn:
                    return

                self.train_thread.stop()
                self.train_thread.wait()

            # Récupère les paramètres + fichiers cochés
            params, selected_files = self.get_training_parameters()
            if params is None or selected_files is None or not selected_files:
                QMessageBox.warning(self, "No Files Selected", "Please select at least one .h5 file to train on.")
                return

            # Chargement des données
            X_all, y_all = [], []
            for fname in selected_files:
                try:
                    full_path = os.path.join(self.dataset_path, fname)
                    X, y = load_and_preprocess_data(
                        full_path,
                        window_size=params["sequence_length"],
                        stride=params["stride"]
                    )
                    X_all.append(X)
                    y_all.append(y)
                except Exception as e:
                    QMessageBox.warning(self, "Error Loading File", f"Error in {fname}:\n{str(e)}")
                    return

            X = np.concatenate(X_all)
            y = np.concatenate(y_all)

            from sklearn.model_selection import train_test_split
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            self.set_tabs_enabled(False)
            # Lance l'entraînement dans un thread
            self.training_monitor_text.clear()  # ✅ Réinitialise le log visuel
            self.training_progress_bar.setValue(0) 
            self.training_stopped = False
            self.train_thread = TrainingThread(X_train, X_val, X_test, y_train, y_val, y_test, params)
            self.train_thread.setParent(self)
            self.train_thread.training_finished.connect(self.on_training_finished)
            self.train_thread.training_log.connect(self.append_training_log)
            progress_state.nn_designed = False
            self.train_thread.start()

            self.nn_text_edit.setText("🧠 Training started...")
            self.is_training = True  # Indique qu'un entraînement est en cours

        except Exception as e:
            QMessageBox.critical(self, "Erreur durant l'entraînement", str(e))

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
        """
        Callback when training is finished.
        
        Args:
            model: The trained model
            history: Training history
            test_results: Dictionary with test results (accuracy, report)
        """
        if hasattr(self, "train_thread") and self.train_thread.isRunning():
            print("Cleaning up training thread...")
            self.train_thread.quit()
            self.train_thread.wait()

        self.is_training = False  # Indique qu'aucun entraînement n'est en cours
        self.set_tabs_enabled(True)  # Réactive les onglets

        if self.training_stopped:
            self.nn_text_edit.setText("🛑 Training was manually stopped.")
            progress_state.nn_designed = False
            self.eval_stack.setCurrentIndex(0)  # Affiche placeholder (pas de plot)
        else:
            acc = test_results["accuracy"]
            report = test_results["report"]
            self.nn_text_edit.setText(f"✅ Training complete\n\nAccuracy: {acc:.3f}\n\n{report}")
            QMessageBox.information(self, "Training Done", f"Model trained.\nTest accuracy: {acc:.2%}")
            progress_state.nn_designed = True
            self.trained_model = model
            self.training_history = history
            self.test_results = test_results
            print("DEBUG history keys:", history.keys() if history else "None")
            self.plot_training_curves(history)  # ✅ Affiche les courbes

            # ✅ STOP PROPRE DU THREAD !
            if hasattr(self, "train_thread") and self.train_thread.isRunning():
                self.train_thread.quit()
                self.train_thread.wait()

            progress_state.trained_model = model
            progress_state.training_history = history
            progress_state.test_results = test_results
            progress_state.training_started = True
            if test_results and "true_labels" in test_results and "predictions" in test_results:
                progress_state.test_results["y_true"] = test_results["true_labels"]
                progress_state.test_results["y_pred"] = test_results["predictions"]
            else:
                progress_state.test_results["y_true"] = []
                progress_state.test_results["y_pred"] = []
            QMessageBox.information(self, "Training Complete", "Training is finished. You can now view results in the 'NN Evaluator' tab.")


    def stop_training(self):
        """Stop training with user confirmation."""
        if hasattr(self, "train_thread") and self.train_thread.isRunning():
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Confirm Stop")
            msg_box.setText("Are you sure you want to stop training?")
            msg_box.setIcon(QMessageBox.Question)

            # Boutons personnalisés
            discard_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)
            yes_button = msg_box.addButton("Yes", QMessageBox.YesRole)
            msg_box.setDefaultButton(discard_button)

            msg_box.exec_()

            if msg_box.clickedButton() == yes_button:
                self.train_thread.stop()
                self.training_stopped = True
                QMessageBox.information(self, "Training Stopped", "Training was manually stopped.")
                self.nn_text_edit.setText("Training was manually stopped.")
            else:
                print("Training stop cancelled.")

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

    def populate_file_list(self):
        """
        Populate the file list with .h5 files from the dataset folder.
        Only called if dataset_path was provided during initialization.
        """
        self.file_list.clear()
        if os.path.exists(self.dataset_path):
            for filename in os.listdir(self.dataset_path):
                if filename.endswith(".h5"):
                    item = QListWidgetItem(filename)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Unchecked)
                    self.file_list.addItem(item)

    def populate_file_list_with_paths(self, file_paths):
        """
        Populate the file list with the provided file paths.
        
        Args:
            file_paths (list): List of file paths to display in the file list.
        """
        self.file_list.clear()
        for file_path in file_paths:
            item = QListWidgetItem(os.path.basename(file_path))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.file_list.addItem(item)
    
    def get_training_parameters(self):
        """Retourne tous les paramètres nécessaires à l'entraînement."""
        hp = self.state["hyperparameters"]
        print("DEBUG layer_types:", hp.get("layer_types", []))  # Ajoute cette ligne
        params = {
            "optimizer": self.state["optimizer"],
            "loss_function": self.state["loss_function"],
            "loss_type": "Classification" if self.classification_loss_combo.currentIndex() >= 0 else "Regression",
            "layer_types": hp.get("layer_types", []),
            "sequence_length": int(hp.get("sequence_length", 50)),
            "stride": int(hp.get("stride", 5)),
            "batch_size": int(hp.get("batch_size", 32)),
            "epochs": int(hp.get("epoch_number", 50)),
            "learning_rate": float(hp.get("learning_rate", 0.001))
        }
        selected_files = [
            self.file_list.item(i).text()
            for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == Qt.Checked
        ]
        return params, selected_files
    
    def add_layer_combo_row(self, preset_value=None):
        row_layout = QHBoxLayout()
        combo = QComboBox()
        combo.addItems(["LSTM", "GRU", "RNN", "Transformer", "TinyTransformer"])
        if preset_value and combo.findText(preset_value) >= 0:
            combo.setCurrentText(preset_value)
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

        remove_btn = QPushButton("🗑")
        remove_btn.setFixedWidth(30)
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #f9dcdc;
                border: 1px solid #dcdcdc;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f5bebe;
            }
        """)

        def remove_row():
            self.layer_type_combos.remove(combo)
            self.layer_combo_rows.remove(row_layout)
            for i in reversed(range(row_layout.count())):
                widget = row_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            self.layer_combo_layout.removeItem(row_layout)

        remove_btn.clicked.connect(remove_row)

        row_layout.addWidget(combo)
        row_layout.addWidget(remove_btn)

        self.layer_type_combos.append(combo)
        self.layer_combo_rows.append(row_layout)
        self.layer_combo_layout.addLayout(row_layout)

    def get_layer_types(self):
        return [combo.currentText() for combo in self.layer_type_combos]

    def plot_training_curves(self, history):
        self.eval_stack.setCurrentIndex(1)

        self.eval_canvas.figure.clf()
        ax1 = self.eval_canvas.figure.add_subplot(2, 1, 1)
        ax2 = self.eval_canvas.figure.add_subplot(2, 1, 2)

        # Loss
        ax1.plot(history['train_loss'], label='Train Loss', color='blue')
        ax1.plot(history['val_loss'], label='Validation Loss', color='orange')
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy
        ax2.plot(history['train_acc'], label='Train Accuracy', color='blue')
        ax2.plot(history['val_acc'], label='Validation Accuracy', color='orange')
        ax2.set_title("Training and Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)

        self.eval_canvas.figure.tight_layout()  # ✅ Résout le chevauchement
        self.eval_canvas.draw()

    def save_model(self):
        """Enregistre le modèle entraîné + historique + résultats dans un fichier .h5 + .json."""

        model = getattr(self, "trained_model", None)
        history = getattr(self, "training_history", None)
        results = getattr(self, "test_results", None)

        if model is None or history is None or results is None:
            QMessageBox.warning(self, "Nothing to Save", "No trained model found. Please train before saving.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "H5 Files (*.h5)")
        if not file_path:
            return

        try:
            save_model_and_results(model, history, results, file_path)
            QMessageBox.information(self, "Success", "Model and results saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save model:\n{str(e)}")

    def closeEvent(self, event):
        if hasattr(self, "train_thread") and self.train_thread.isRunning():
            print("Closing event: stopping train thread")
            self.train_thread.quit()
            self.train_thread.wait()
        event.accept()


    def restore_checkboxes(self):
        """Restore checkbox states for file list."""
        selected_files_set = set(self.state.get("selected_files", []))
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            item.setCheckState(Qt.Checked if item.text() in selected_files_set else Qt.Unchecked)

    def set_tabs_enabled(self, enabled: bool):
        if hasattr(self, "header"):
            for tab in self.header.tabs.values():
                tab_label = tab.layout().itemAt(0).widget()
                tab_label.setEnabled(enabled)



