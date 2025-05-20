import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QGridLayout, QLabel, QWidget, QSpacerItem, QSizePolicy, QFileDialog, QPushButton, QHBoxLayout, QStackedWidget, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from windows.dataset_builder_window import DatasetBuilderWindow
from widgets.header import Header 


class NeuralNetworkEvaluator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Monitoring Software")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget and layout
        central_widget = QWidget()
        parent_layout = QVBoxLayout()  # Layout principal parent
        parent_layout.setContentsMargins(0, 0, 0, 0)  # Pas de marges globales

        # Header Section
        header_container = QWidget()
        header_container_layout = QVBoxLayout()

        # Supprimer les marges et espacement pour le conteneur du header
        header_container_layout.setContentsMargins(0, 0, 0, 0)
        header_container_layout.setSpacing(0)

        header = Header(active_page="Neural Network Evaluator", parent_window=self)
        header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        header_container_layout.addWidget(header)
        header_container.setLayout(header_container_layout)
        header_container.setContentsMargins(0, 0, 0, 0)  # Pas de marges pour le conteneur

        parent_layout.addWidget(header_container)  # Ajouter le header au parent layout

        # Main Content Section
        main_content_container = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(20, 20, 20, 20)  # Ajouter des marges uniquement au contenu principal
        self.main_layout.setSpacing(10)  # Espacement entre les composants principaux

        title = QLabel("Neural Network Evaluator")
        title_font = QFont("Arial", 18)
        title_font.setBold(True)
        title.setFont(title_font)

        # Label pour afficher la progression (à droite du titre)
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

        # Mettre le titre et le label de progression sur la même ligne
        header_layout = QHBoxLayout()
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.top_right_label)

        self.main_layout.addLayout(header_layout)

        # Mise à jour du label
        self.update_progress_label(active_step="NN Evaluator", completed_steps=["Dataset Builder", "NN Designer"])
        ''' FIN HARRY PROGRESS BAR (ALIGNE AU MM NIVEAU QUE LE TITRE) '''

        # Menu buttons
        self.add_menu_buttons()

        # Add a stacked widget for menu content
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        # Add an empty page as the default page
        self.add_empty_page()

        # Add pages to the stacked widget
        self.add_training_overview_page()
        self.add_evaluate_test_set_page()
        self.add_compare_models_page()

        # Set the empty page as the default page
        self.stacked_widget.setCurrentIndex(0)

        # Footer
        footer = QLabel("Data Monitoring Software version 1.0.13")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: gray; font-size: 12px; padding: 10px;")
        self.main_layout.addWidget(footer)

        # Ajouter le contenu principal au conteneur
        main_content_container.setLayout(self.main_layout)
        parent_layout.addWidget(main_content_container)

        # Set layout
        central_widget.setLayout(parent_layout)
        self.setCentralWidget(central_widget)

        # Add menu bar
        self.add_menu_bar()

    ''' DEBUT HARRY PROGRESS BAR '''
    def update_progress_label(self, active_step, completed_steps=None):
        """
        Met à jour le label de progression avec :
        - les étapes complètes en vert
        - l’étape active en orange
        - les autres en style normal
        """
        steps = ["Dataset Builder", "NN Designer", "NN Evaluator"]
        completed_steps = completed_steps or []

        parts = []
        for step in steps:
            if step in completed_steps:
                # Vert gras
                parts.append(f'<span style="color: green; font-weight:bold;">{step}</span>')
            elif step == active_step:
                # Orange gras
                parts.append(f'<span style="color: orange; font-weight:bold;">{step}</span>')
            else:
                # Style normal
                parts.append(step)

        text = " → ".join(parts)
        self.top_right_label.setText(f"Progress Statement : {text}")
    ''' FIN HARRY PROGRESS BAR '''

    def add_menu_buttons(self):
        """Add the main menu buttons."""
        menu_button_layout = QHBoxLayout()

        # Training Overview button
        self.training_overview_button = QPushButton("Training Overview")
        self.training_overview_button.setStyleSheet(self.get_button_style())
        self.training_overview_button.clicked.connect(lambda: [self.switch_menu(1), self.update_menu_button_styles(0)])
        menu_button_layout.addWidget(self.training_overview_button)

        self.evaluate_test_set_button = QPushButton("Evaluate on Test Set")
        self.evaluate_test_set_button.setStyleSheet(self.get_button_style())
        self.evaluate_test_set_button.clicked.connect(lambda: [self.switch_menu(2), self.update_menu_button_styles(1)])
        menu_button_layout.addWidget(self.evaluate_test_set_button)

        self.compare_models_button = QPushButton("Compare Models")
        self.compare_models_button.setStyleSheet(self.get_button_style())
        self.compare_models_button.clicked.connect(lambda: [self.switch_menu(3), self.update_menu_button_styles(2)])
        menu_button_layout.addWidget(self.compare_models_button)

        self.main_layout.addLayout(menu_button_layout)

    def add_menu_bar(self):
        """Add the menu bar with file and options menus."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        # Dataset actions
        create_dataset_action = file_menu.addAction("Create new dataset")
        create_dataset_action.triggered.connect(self.create_new_dataset)

        load_dataset_action = file_menu.addAction("Load existing dataset")
        load_dataset_action.triggered.connect(self.load_existing_dataset)

        file_menu.addSeparator()

        save_dataset_action = file_menu.addAction("Save dataset")
        save_dataset_action.setEnabled(False)  # Disabled by default
        save_dataset_action.triggered.connect(self.save_dataset)

        save_dataset_as_action = file_menu.addAction("Save dataset as...")
        save_dataset_as_action.setEnabled(False)  # Disabled by default
        save_dataset_as_action.triggered.connect(self.save_dataset_as)

        # Model actions
        load_model_action = file_menu.addAction("Load existing Model")
        load_model_action.triggered.connect(self.load_existing_model)

        file_menu.addSeparator()

        save_model_action = file_menu.addAction("Save current Model")
        save_model_action.setEnabled(False)  # Disabled by default
        save_model_action.triggered.connect(self.save_model)

        save_model_as_action = file_menu.addAction("Save current Model as...")
        save_model_as_action.setEnabled(False)  # Disabled by default
        save_model_as_action.triggered.connect(self.save_model_as)

        # Add more options if needed
        file_menu.addSeparator()

        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        edit_menu = menubar.addMenu("Edit")
        options_menu = menubar.addMenu("Options")

    def get_button_style(self):
        """Return the style for the buttons."""
        return """
            QPushButton {
                background-color: #e7f3ff;
                border: 1px solid #a6c8ff;
                border-radius: 5px;
                font-size: 16px;
                padding: 12px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #d0e7ff;
            }
            QPushButton:pressed {
                background-color: #b3d4ff;
            }
        """

    def switch_menu(self, index):
        """Switch the displayed menu in the stacked widget."""
        self.stacked_widget.setCurrentIndex(index)

    def update_menu_button_styles(self, active_index):
        """Update the styles of the menu buttons to highlight the active one."""
        buttons = [self.training_overview_button, self.evaluate_test_set_button, self.compare_models_button]
        for i, button in enumerate(buttons):
            if i == active_index:
                button.setStyleSheet("""
                    QPushButton {
                        background-color: white;
                        border: 2px solid #a6c8ff;
                        border-radius: 5px;
                        font-size: 16px;
                        padding: 12px;
                        min-width: 150px;
                    }
                    QPushButton:hover {
                        background-color: #f0f8ff;
                    }
                    QPushButton:pressed {
                        background-color: #e0f0ff;
                    }
                """)
            else:
                button.setStyleSheet(self.get_button_style())

    def add_training_overview_page(self):
        """Add the Training Overview page."""
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Add the "Load Model" button
        load_model_button = QPushButton("Load Model")
        load_model_button.setStyleSheet("""
            QPushButton {
                background-color: #e7f3ff;
                border: 1px solid #a6c8ff;
                border-radius: 5px;
                font-size: 16px;
                padding: 12px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #d0e7ff;
            }
            QPushButton:pressed {
                background-color: #b3d4ff;
            }
        """)
        load_model_button.clicked.connect(self.load_existing_model)
        layout.addWidget(load_model_button, alignment=Qt.AlignLeft)

        # Create a horizontal layout for the plots
        plots_layout = QHBoxLayout()
        plots_layout.setSpacing(20)  # Add spacing between the plots

        # Add the Ground Truth vs Prediction plot
        ground_truth_container = QWidget()
        ground_truth_layout = QVBoxLayout()
        ground_truth_container.setLayout(ground_truth_layout)
        ground_truth_container.setStyleSheet("background-color: white; padding: 10px;")  # Removed border

        ground_truth_plot = FigureCanvas(Figure(figsize=(6, 4)))  
        self.add_ground_truth_vs_prediction(ground_truth_plot.figure)
        ground_truth_layout.addWidget(ground_truth_plot)
        plots_layout.addWidget(ground_truth_container, stretch=3)  # Stretch factor for left plot

        # Add the Predicted vs Ground Truth plot
        predicted_vs_ground_truth_container = QWidget()
        predicted_vs_ground_truth_layout = QVBoxLayout()
        predicted_vs_ground_truth_container.setLayout(predicted_vs_ground_truth_layout)
        predicted_vs_ground_truth_container.setStyleSheet("background-color: white; padding: 10px;")  # Removed border

        predicted_vs_ground_truth_plot = FigureCanvas(Figure(figsize=(4, 4)))  # Smaller plot
        self.add_predicted_vs_ground_truth(predicted_vs_ground_truth_plot.figure)
        predicted_vs_ground_truth_layout.addWidget(predicted_vs_ground_truth_plot)
        plots_layout.addWidget(predicted_vs_ground_truth_container, stretch=2)  # Stretch factor for right plot

        # Add the horizontal layout with plots to the main layout
        layout.addLayout(plots_layout)

        # Set the layout for the page
        page.setLayout(layout)
        self.stacked_widget.addWidget(page)

    def add_ground_truth_vs_prediction(self, figure):
        """Add the Ground Truth vs Prediction plot."""
        ax = figure.add_subplot(111)
        # Simulated data for Ground Truth and Prediction
        x = range(0, 1600, 10)
        ground_truth = [2 * (i % 100) / 100 - 1 for i in x]  # Simulated sine-like data
        prediction = [gt + (0.1 * (-1) ** i) for i, gt in enumerate(ground_truth)]  # Slightly noisy prediction

        ax.plot(x, ground_truth, label="Ground Truth", color="blue", linewidth=1.5)
        ax.plot(x, prediction, label="Prediction", color="orange", linestyle="--", linewidth=1.5)
        ax.set_title("Ground Truth vs Prediction", fontsize=14)
        ax.set_xlabel("Frame", fontsize=12)
        ax.set_ylabel("Forward Speed (m/s)", fontsize=12)
        ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)  # Move legend closer to the plot
        ax.grid(True)

        # Adjust margins to reduce the bottom space
        figure.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)  # Reduced bottom margin

    def add_predicted_vs_ground_truth(self, figure):
        """Add the Predicted vs Ground Truth scatter plot."""
        ax = figure.add_subplot(111)
        # Simulated data for scatter plot
        import numpy as np
        ground_truth = np.linspace(-2, 3, 200)
        prediction = ground_truth + np.random.normal(0, 0.2, size=ground_truth.shape)

        ax.scatter(ground_truth, prediction, label="Data", alpha=0.7, color="blue")
        ax.plot([-2, 3], [-2, 3], linestyle="--", color="black", label="Ideal", linewidth=1.5)
        ax.set_title("Predicted vs Ground Truth", fontsize=14)
        ax.set_xlabel("Ground Truth (m/s)", fontsize=12)
        ax.set_ylabel("Prediction (m/s)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True)

    # Methods for menu actions
    def create_new_dataset(self):
        """Open the DatasetBuilderWindow."""
        self.dataset_builder = DatasetBuilderWindow(start_window_ref=self)
        self.dataset_builder.show()
        self.close()

    def load_existing_dataset(self):
        """Open a file dialog to load an existing dataset."""
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly  # Open in read-only mode
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Existing Dataset",
            "",
            "Dataset Files (*.csv *.json *.txt);;All Files (*)",
            options=options
        )
        if file_path:
            print(f"Dataset loaded from: {file_path}")
            # Add logic to process the dataset file here

    def save_dataset(self):
        print("Saving the dataset...")

    def save_dataset_as(self):
        print("Saving the dataset as...")

    def load_existing_model(self):
        """Open a file dialog to load an existing model."""
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly  # Open in read-only mode
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Existing Model",
            "",
            "Model Files (*.h5 *.pt *.onnx);;All Files (*)",
            options=options
        )
        if file_path:
            print(f"Model loaded from: {file_path}")
            # Add logic to process the model file here

    def save_model(self):
        print("Saving the current model...")

    def save_model_as(self):
        print("Saving the current model as...")

    def show_training_overview(self):
        """Display the Ground Truth vs Prediction and Predicted vs Ground Truth plots."""
        # Clear the plots container
        while self.plots_container.count():
            child = self.plots_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add the "Load Model" button
        load_model_button = QPushButton("Load model")
        load_model_button.setStyleSheet("""
            QPushButton {
                background-color: #e7f3ff;
                border: 1px solid #a6c8ff;
                border-radius: 5px;
                font-size: 16px;
                padding: 12px;
                min-width: 150px;  /* Largeur minimale */
            }
            QPushButton:hover {
                background-color: #d0e7ff;
            }
            QPushButton:pressed {
                background-color: #b3d4ff;
            }
        """)
        load_model_button.clicked.connect(self.load_existing_model)

        # Create a layout for the "Load Model" button
        load_model_layout = QVBoxLayout()
        load_model_layout.addWidget(load_model_button, alignment=Qt.AlignLeft)
        load_model_layout.setContentsMargins(10, 20, 0, 20)  # Margins: left, top, right, bottom
        self.plots_container.addLayout(load_model_layout)

        # Create a horizontal layout for the plots
        plots_layout = QHBoxLayout()
        plots_layout.setSpacing(20)  # Add spacing between the plots

        # Add the Ground Truth vs Prediction plot
        ground_truth_container = QWidget()
        ground_truth_layout = QVBoxLayout()
        ground_truth_container.setLayout(ground_truth_layout)
        ground_truth_container.setStyleSheet("background-color: white; border: 2px solid black; padding: 10px;")

        ground_truth_plot = FigureCanvas(Figure(figsize=(5, 4)))
        self.add_ground_truth_vs_prediction(ground_truth_plot.figure)
        ground_truth_layout.addWidget(ground_truth_plot)
        plots_layout.addWidget(ground_truth_container)

        # Add the Predicted vs Ground Truth plot
        predicted_vs_ground_truth_container = QWidget()
        predicted_vs_ground_truth_layout = QVBoxLayout()
        predicted_vs_ground_truth_container.setLayout(predicted_vs_ground_truth_layout)
        predicted_vs_ground_truth_container.setStyleSheet("background-color: white; border: 2px solid black; padding: 10px;")

        predicted_vs_ground_truth_plot = FigureCanvas(Figure(figsize=(5, 4)))
        self.add_predicted_vs_ground_truth(predicted_vs_ground_truth_plot.figure)
        predicted_vs_ground_truth_layout.addWidget(predicted_vs_ground_truth_plot)
        plots_layout.addWidget(predicted_vs_ground_truth_container)

        # Add the horizontal layout with plots to the main layout
        self.plots_container.addLayout(plots_layout)

    def add_training_loss_plot(self, figure):
        """Add a plot showing the training loss over epochs."""
        ax = figure.add_subplot(111)
        epochs = range(1, 21)  # Example: 20 epochs
        loss = [1 / (epoch + 1) for epoch in epochs]  # Example: decreasing loss
        ax.plot(epochs, loss, label="Training Loss", color="blue")
        ax.set_title("Training Loss Over Time")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)

    def add_training_accuracy_plot(self, figure):
        """Add a plot showing the training accuracy over epochs."""
        ax = figure.add_subplot(111)
        epochs = range(1, 21)  # Example: 20 epochs
        accuracy = [epoch / 20 for epoch in epochs]  # Example: increasing accuracy
        ax.plot(epochs, accuracy, label="Training Accuracy", color="green")
        ax.set_title("Training Accuracy Over Time")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True)

    def evaluate_on_test_set(self):
        """Placeholder for Evaluate on Test Set functionality."""
        print("Evaluate on Test Set clicked!")

    def compare_models(self):
        """Placeholder for Compare Models functionality."""
        print("Compare Models clicked!")

    def add_evaluate_test_set_page(self):
        """Add the Evaluate on Test Set page."""
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Main grid layout
        main_grid = QGridLayout()
        main_grid.setHorizontalSpacing(20)  
        main_grid.setVerticalSpacing(10)   
        main_grid.setContentsMargins(10, 10, 10, 10) 

        # Classification Section
        classification_label = QLabel("<b><i>Classification</i></b>")
        classification_label.setAlignment(Qt.AlignLeft)
        classification_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred) 
        main_grid.addWidget(classification_label, 0, 0, 1, 2)

        # Accuracy row
        accuracy_label = QLabel("<b><i>Accuracy</i></b>")
        accuracy_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  
        accuracy_value = QLabel("99.8%")
        accuracy_value.setStyleSheet("border: 1px solid gray; padding: 2px; background-color: white; font-size: 10px;")
        accuracy_value.setFixedSize(80, 30)  
        main_grid.addWidget(accuracy_label, 1, 0, alignment=Qt.AlignRight)  
        main_grid.addWidget(accuracy_value, 1, 1, alignment=Qt.AlignLeft)  

        main_grid.addItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum), 1, 2)

        # Confusion Matrix row
        confusion_matrix_label = QLabel("<b><i>Confusion matrix</i></b>")
        confusion_matrix_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  
        confusion_matrix_button = QPushButton("Visualize confusion matrix")
        confusion_matrix_button.setFixedSize(120, 120)  
        confusion_matrix_button.setStyleSheet("""
            QPushButton {
                background-color: #add8e6;
                border: 1px solid #000;
                font-size: 10px;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: #87ceeb;
            }
        """)
        main_grid.addWidget(confusion_matrix_label, 2, 0, alignment=Qt.AlignRight)  
        main_grid.addWidget(confusion_matrix_button, 2, 1, alignment=Qt.AlignLeft)  

        main_grid.addItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum), 2, 2)

        # Regression Section
        regression_label = QLabel("<b><i>Regression</b></i>")
        regression_label.setAlignment(Qt.AlignLeft)
        regression_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  
        main_grid.addWidget(regression_label, 3, 0, 1, 2)

        # MSE row
        mse_label = QLabel("<b><i>MSE</i></b>")
        mse_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  
        mse_value = QLabel("99.8%")
        mse_value.setStyleSheet("border: 1px solid gray; padding: 2px; background-color: white; font-size: 10px;")
        mse_value.setFixedSize(80, 30)  
        main_grid.addWidget(mse_label, 4, 0, alignment=Qt.AlignRight)  
        main_grid.addWidget(mse_value, 4, 1, alignment=Qt.AlignLeft) 

        # RMSE row
        rmse_label = QLabel("<b><i>RMSE</i></b>")
        rmse_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  
        rmse_value = QLabel("99.8%")
        rmse_value.setStyleSheet("border: 1px solid gray; padding: 2px; background-color: white; font-size: 10px;")
        rmse_value.setFixedSize(80, 30) 
        main_grid.addWidget(rmse_label, 5, 0, alignment=Qt.AlignRight) 
        main_grid.addWidget(rmse_value, 5, 1, alignment=Qt.AlignLeft) 

        # Regression plot
        regression_plot = FigureCanvas(Figure(figsize=(12, 10)))  
        regression_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  
        self.add_predicted_vs_ground_truth(regression_plot.figure)

        regression_plot.figure.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

        main_grid.addWidget(regression_plot, 6, 0, 1, 2)  

        main_grid.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding), 7, 0, 1, 2)

        # Add the grid layout to the main layout
        layout.addLayout(main_grid)
        layout.setContentsMargins(0, 0, 0, 0)  
        page.setLayout(layout)
        self.stacked_widget.addWidget(page)

    def add_compare_models_page(self):
        """Add the Compare Models page with maximized plot visualization."""
        page = QWidget()
        layout = QVBoxLayout()

        comparison_layout = QHBoxLayout()
        
        # Model A
        model_a_layout = QVBoxLayout()
        model_a_label = QLabel("<b><i>Model A</b></i>")
        model_a_layout.addWidget(model_a_label, alignment=Qt.AlignCenter)
        
        model_a_selector = QComboBox()
        model_a_selector.addItems(["Select the model...", "Model 1", "Model 2", "Model 3"])
        model_a_selector.setStyleSheet("""
            QComboBox {
                border: 1px solid #a6c8ff;
                border-radius: 5px;
                padding: 5px;
                background-color: #f9f9f9;
                font-size: 14px;
                color: #333;
            }
            QComboBox:hover {
                border: 1px solid #87ceeb;
                background-color: #ffffff;
            }
            QComboBox::drop-down {
                border-left: 1px solid #a6c8ff;
                background-color: #e7f3ff;
            }
            QComboBox::down-arrow {
                image: url(assets/arrow_down.png); /* Replace with your arrow icon path */
                width: 10px;
                height: 10px;
            }
        """)
        model_a_layout.addWidget(model_a_selector)
        
        model_a_plot = FigureCanvas(Figure(figsize=(10, 8)))  
        model_a_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.add_ground_truth_vs_prediction(model_a_plot.figure)
        model_a_layout.addWidget(model_a_plot)
        
        model_a_layout.addWidget(QLabel("<b><i>Accuracy</b></i>"), alignment=Qt.AlignLeft)
        model_a_layout.addWidget(QLabel("<b><i>Used data</b></i>"), alignment=Qt.AlignLeft)
        model_a_layout.addWidget(QLabel("<b><i>Summary of Hyperparameters</b></i>"), alignment=Qt.AlignLeft)
        
        comparison_layout.addLayout(model_a_layout)
        
        # Model B
        model_b_layout = QVBoxLayout()
        model_b_label = QLabel("<b><i>Model B</b></i>")
        model_b_layout.addWidget(model_b_label, alignment=Qt.AlignCenter)
        
        model_b_selector = QComboBox()
        model_b_selector.addItems(["Select the model...", "Model 1", "Model 2", "Model 3"])
        model_b_selector.setStyleSheet("""
            QComboBox {
                border: 1px solid #a6c8ff;
                border-radius: 5px;
                padding: 5px;
                background-color: #f9f9f9;
                font-size: 14px;
                color: #333;
            }
            QComboBox:hover {
                border: 1px solid #87ceeb;
                background-color: #ffffff;
            }
            QComboBox::drop-down {
                border-left: 1px solid #a6c8ff;
                background-color: #e7f3ff;
            }
            QComboBox::down-arrow {
                image: url(assets/arrow_down.png); /* Replace with your arrow icon path */
                width: 10px;
                height: 10px;
            }
        """)
        model_b_layout.addWidget(model_b_selector)
        
        model_b_plot = FigureCanvas(Figure(figsize=(10, 8))) 
        model_b_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.add_ground_truth_vs_prediction(model_b_plot.figure)
        model_b_layout.addWidget(model_b_plot)
        
        model_b_layout.addWidget(QLabel("<b><i>Accuracy</b></i>"), alignment=Qt.AlignLeft)
        model_b_layout.addWidget(QLabel("<b><i>Used data</b></i>"), alignment=Qt.AlignLeft)
        model_b_layout.addWidget(QLabel("<b><i>Summary of Hyperparameters</b></i>"), alignment=Qt.AlignLeft)
        
        comparison_layout.addLayout(model_b_layout)
        
        layout.addLayout(comparison_layout)
        page.setLayout(layout)
        self.stacked_widget.addWidget(page)

    def add_empty_page(self):
        """Add an empty page to the stacked widget."""
        empty_page = QWidget()
        self.stacked_widget.addWidget(empty_page)

    def resizeEvent(self, event):
        """Handle window resize events to adjust the plots dynamically."""
        for i in range(self.stacked_widget.count()):
            page = self.stacked_widget.widget(i)
            if page:
                page.updateGeometry()
        super().resizeEvent(event)
