import h5py
import numpy as np

file_path = r"C:\Users\harry\OneDrive - UPEC\Documents\BUT2\Stage\KAIST\NN Package\NN-Package\datasets\dataset1\filtered_sensor_20250520_214656.h5"

with h5py.File(file_path, 'r') as f:
    if "Sensor" not in f:
        print("❌ Ce fichier ne contient pas de groupe 'Sensor'")
    else:
        sensor = f["Sensor"]
        print("✅ Clés disponibles dans Sensor :", list(sensor.keys()))
        print()

        for key in sensor:
            data = np.array(sensor[key])
            print(f"📌 Clé : {key}")
            print(f"Shape : {data.shape}")
            print(f"Extrait :\n{data[..., :10]}")  # Affiche les 10 premières valeurs
            print("-" * 40)
