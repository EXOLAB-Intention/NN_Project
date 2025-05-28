import json

with open(r"C:\Users\harry\OneDrive - UPEC\Documents\BUT2\Stage\KAIST\NN Package\NN-Package\keras50_results.json", "r") as f:
    results = json.load(f)

print("Accuracy:", results["accuracy"])
print("Report:", results["report"])

# Pour accéder aux prédictions et vrais labels :
preds = results["predictions"]
true = results["true_labels"]
