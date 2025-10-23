import joblib

# Nom du fichier du modèle sauvegardé
model_filename = 'lgbm_rossmann_model.joblib'

# Charger le modèle depuis le fichier
loaded_model = joblib.load(model_filename)

print("Modèle chargé avec succès.")

# Tu peux maintenant utiliser loaded_model pour faire des prédictions
# new_predictions = loaded_model.predict(new_data)