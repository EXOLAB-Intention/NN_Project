import torch

# Charger le modèle
checkpoint = torch.load(r"C:\Users\harry\OneDrive - UPEC\Documents\BUT2\Stage\KAIST\NN Package\NN-Package\test.pt", map_location="cpu")

# Afficher ce qu’il y a dedans
print("Contenu du modèle sauvegardé :")
for key, value in checkpoint.items():
    print(f"{key}: {type(value)}")

# Si c'est un modèle hybride
if 'model_type' in checkpoint:
    model_type = checkpoint['model_type']
    if isinstance(model_type, list):
        print("🧠 Modèle hybride utilisé avec les couches suivantes :")
        for i, layer in enumerate(model_type):
            print(f"  Couche {i+1}: {layer}")
    else:
        print(f"Modèle utilisé : {model_type}")
else:
    print("Aucune information sur le type de modèle.")
