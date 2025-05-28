from tensorflow import keras

# Charger le modèle
model = keras.models.load_model(r"C:\Users\harry\OneDrive - UPEC\Documents\BUT2\Stage\KAIST\NN Package\NN-Package\keras.h5")

# Résumé du modèle
model.summary()

# Utilisation possible (si tu as des données) :
# predictions = model.predict(X_test)
