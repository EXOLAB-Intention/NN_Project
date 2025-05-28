# windows/progress_state.py

# Dataset
dataset_path = None                # Chemin vers le dataset généré
dataset_files = []                # Fichiers utilisés dans le dataset
dataset_built = False             # Indicateur si le dataset a été construit

# Neural Network Designer
nn_designed = False               # Si le réseau a été configuré
training_started = False          # Si l'entraînement a commencé

# Résultats d'entraînement
training_history = {}            # Contient .history d’un objet keras.callbacks.History
trained_model = None             # Modèle entraîné (optionnel si tu veux l’utiliser ailleurs)

# Résultats de test
test_results = {                 # Contient les résultats pour affichage des courbes
    "y_true": [],
    "y_pred": []
}