import h5py
import os
import numpy as np

class ModelManager:
    def __init__(self):
        self.models = {}
        self.unnamed_count = 1

    def load_model(self, filepath):
        """Load model from H5 file"""
        try:
            with h5py.File(filepath, 'r') as f:
                model_name = os.path.splitext(os.path.basename(filepath))[0]
                
                # Extract data from H5 file
                model_info = {
                    'filepath': filepath,
                    'results': {
                        'y_true': np.array(f['test_results/y_true'][:]),
                        'y_pred': np.array(f['test_results/y_pred'][:]),
                        'time': np.array(f['test_results/time'][:])
                    },
                    'accuracy': f.attrs.get('accuracy', 0.0),
                    'inputs': f.attrs.get('inputs', []),
                    'hyperparameters': {
                        'learning_rate': f.attrs.get('learning_rate', 'N/A'),
                        'batch_size': f.attrs.get('batch_size', 'N/A'),
                        'epochs': f.attrs.get('epochs', 'N/A'),
                        'layers': f.attrs.get('architecture', 'N/A')
                    }
                }
                
                self.models[model_name] = model_info
                return model_name
                
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def add_current_model(self, test_results):
        """Add current model to manager"""
        model_name = f"Model {self.unnamed_count}"
        self.unnamed_count += 1
        
        self.models[model_name] = {
            'filepath': None,
            'results': test_results,
            'accuracy': test_results.get('accuracy', 0.0),
            'inputs': [],
            'hyperparameters': {}
        }
        return model_name

    def get_model_info(self, model_name):
        """Get model information"""
        return self.models.get(model_name)

    def get_model_names(self):
        """Get list of model names"""
        return list(self.models.keys())