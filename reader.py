import h5py
import numpy as np
import json

def read_h5_content(file_path):
    """
    Lit et affiche le contenu d'un fichier H5, y compris les datasets et les attributs.
    """
    try:
        with h5py.File(file_path, 'r') as hf:
            print(f"Contenu du fichier H5 : {file_path}\n")

            def print_dataset(name, obj):
                """Fonction interne pour afficher les datasets."""
                print(f"  Nom: {name}")
                if isinstance(obj, h5py.Dataset):
                    print(f"    Type: Dataset")
                    print(f"    Shape: {obj.shape}")
                    print(f"    Type de données: {obj.dtype}")

                    try:
                        data = obj[()]

                        if obj.dtype == np.dtype('O'):
                            print("    Contenu:")
                            if obj.shape == ():  # Gère les datasets scalaires de type object
                                try:
                                    decoded_data = json.loads(np.array(data).item().decode('utf-8'))
                                    print(f"      {decoded_data}")
                                except (UnicodeDecodeError, json.JSONDecodeError, AttributeError) as e:
                                    print(f"      (Non-decodable ou non-JSON: {e})")
                                except Exception as e:
                                    print(f"      (Erreur lors du décodage JSON: {e})")
                            else:  # Gère les datasets array de type object
                                for i in range(min(10, len(data))):
                                    try:
                                        decoded_data = json.loads(data[i].decode('utf-8'))
                                        print(f"      [{i}]: {decoded_data}")
                                    except (UnicodeDecodeError, json.JSONDecodeError) as e:
                                        print(f"      [{i}]: (Non-decodable ou non-JSON: {e})")
                                    except Exception as e:
                                        print(f"      (Erreur lors du décodage JSON: {e})")

                        elif obj.dtype.char == 'S':
                            print("    Contenu (10 premiers éléments):")
                            if isinstance(data, np.ndarray):
                                for i in range(min(10, len(data))):
                                    try:
                                        print(f"      [{i}]: {data[i].decode('utf-8')}")
                                    except UnicodeDecodeError:
                                        print(f"      [{i}]: (Non-decodable)")
                            else:
                                print("      Dataset vide ou de type non supporté.")
                        else:
                            print("    Contenu (10 premiers éléments):")
                            if isinstance(data, np.ndarray):
                                for i in range(min(10, len(data))):
                                    print(f"      [{i}]: {data[i]}")
                            else:
                                print("      Dataset vide ou de type non supporté.")

                    except Exception as e:
                        print(f"    Erreur lors de la lecture des données : {e}")

                elif isinstance(obj, h5py.Group):
                    print(f"  Type: Groupe")
                    # Appel récursif pour afficher le contenu du groupe
                    for key in obj.keys():
                        print_dataset(f"{name}/{key}", obj[key])

            hf.visititems(print_dataset)

            # Gestion spécifique pour 'selected_files' et 'checked_files'
            if 'files/selected_files' in hf:
                print("\nContenu de 'files/selected_files':")
                try:
                    selected_files = hf['files/selected_files'][()]
                    if isinstance(selected_files, np.ndarray):
                        for i, file in enumerate(selected_files):
                            try:
                                print(f"      [{i}]: {file.decode('utf-8')}")
                            except UnicodeDecodeError:
                                print(f"      [{i}]: (Non-decodable)")
                    else:
                        print("  Non affichable")
                except Exception as e:
                    print(f"  Erreur lors de la lecture : {e}")

            if 'files/checked_files' in hf:
                print("\nContenu de 'files/checked_files':")
                try:
                    checked_files = hf['files/checked_files'][()]
                    if isinstance(checked_files, np.ndarray):
                        for i, file in enumerate(checked_files):
                            try:
                                print(f"      [{i}]: {file.decode('utf-8')}")
                            except UnicodeDecodeError:
                                print(f"      [{i}]: (Non-decodable)")
                    else:
                        print("  Non affichable")
                except Exception as e:
                    print(f"  Erreur lors de la lecture : {e}")

    except Exception as e:
        print(f"Erreur lors de la lecture du fichier H5 : {e}")

if __name__ == '__main__':
    # Remplace 'path/to/your/file.h5' par le chemin réel de ton fichier H5
    file_path = r'C:\Users\harry\OneDrive - UPEC\Documents\BUT2\Stage\KAIST\NN Package\NN_Project\datasets\dataset3\filtered_cropped_20250520_214656.h5'
    read_h5_content(file_path)