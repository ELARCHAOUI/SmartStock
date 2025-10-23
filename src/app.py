import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import  matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Rossmann Sales Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Chargement du Modèle (une seule fois pour la performance) ---
@st.cache_resource # Cache le chargement pour éviter de le recharger à chaque interaction
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../models/lgbm_rossmann_model.joblib')
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

model = load_model()

# --- Chargement d'un exemple de données pour les colonnes (IMPORTANT) ---
# Il faut connaître l'ordre et le type des colonnes que le modèle attend.
# Nous allons charger une partie de ton dataset préparé pour extraire ces infos.
@st.cache_data # Cache le chargement des données
def load_example_data():
    try:
        # Adapte ce chemin si ton CSV est directement à la racine du dossier 'app'
        # Ou si tu le places ailleurs
        data_path = os.path.join(os.path.dirname(__file__), './data/processed/data_prepared.csv')
        # Pour le déploiement, il est préférable d'avoir un petit échantillon
        # Ou juste la liste des colonnes et leurs valeurs par défaut.
        # Pour simplifier, nous allons juste extraire la liste des features.

        # Pour commencer, je te donne une structure basée sur ce que nous avons vu
        # Tu devras remplir les valeurs par défaut de ces features
        
        # IMPORTANT : La liste `feature_columns` doit être EXACTEMENT la même que celle utilisée
        # pour X_train en Python et dans le même ordre.
        
        # Tu dois re-générer un fichier CSV SAMPLE (petit échantillon) de ton rossmann_prepared_data.csv
        # Ou au moins avoir la liste exacte des colonnes.
        
        # Pour l'instant, listons les colonnes que nous avons identifiées:
        feature_columns_names = [
            'Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
            'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'IsWeekend',
            'IsStartOfMonth', 'IsEndOfMonth', 'DayOfYear', 'Quarter', 'HasCompetition',
            'DateCompetition', 'IsPromo2ActiveMonth',
            'StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d',
            'Assortment_a', 'Assortment_b', 'Assortment_c',
            'StateHoliday_0', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c',
            'SchoolHoliday_0', 'SchoolHoliday_1',
            'DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7',
            'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
            'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12',
            'PromoInterval_None', 'PromoInterval_Jan_Apr_Jul_Oct',
            'PromoInterval_Feb_May_Aug_Nov', 'PromoInterval_Mar_Jun_Sept_Dec',
            # Et n'oublie pas 'Year' si elle est dans ton X_train
            'Year' # Si tu as laissé 'Year' numérique
        ]
        return feature_columns_names
    except Exception as e:
        st.error(f"Erreur lors du chargement des noms de features : {e}")
        return []

feature_cols_names = load_example_data()

# Dans ton app.py, après avoir défini df_full et le modèle
# Et après la logique de prédiction, si tu veux afficher les résultats du passé

@st.cache_data
def load_results_for_dashboard():
    try:
        # Assurez-vous d'avoir exporté ce fichier de Colab et de le placer dans votre dossier 'app'
        results_path = os.path.join(os.path.dirname(__file__), '../PowerBi/rossmann_predictions_for_powerbi.csv')
        df_results = pd.read_csv(results_path)
        df_results['Date'] = pd.to_datetime(df_results['Date'])
        return df_results
    except Exception as e:
        st.warning(f"Impossible de charger les résultats passés pour le dashboard : {e}")
        return pd.DataFrame()

df_results = load_results_for_dashboard()

if not df_results.empty:
    st.subheader("Performance Historique des Prédictions")
    
    # Slicer pour choisir un magasin pour le graphique
    store_ids_available = sorted(df_results['Store'].unique().tolist())
    dashboard_store_id = st.selectbox("Sélectionnez un magasin pour l'historique :", options=store_ids_available)

    store_data = df_results[df_results['Store'] == dashboard_store_id].sort_values(by='Date')

    if not store_data.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(store_data['Date'], store_data['Actual_Sales'], label='Ventes Réelles', marker='o', markersize=4)
        ax.plot(store_data['Date'], store_data['Predicted_Sales'], label='Ventes Prédites', linestyle='--', marker='x', markersize=4)
        ax.set_xlabel("Date")
        ax.set_ylabel("Ventes")
        ax.set_title(f"Prédictions de Ventes vs Réalité pour le Magasin {dashboard_store_id}")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig) # Affiche le graphique Matplotlib dans Streamlit
        plt.close(fig) # Ferme la figure pour éviter les avertissements Streamlit

        # Afficher des métriques pour ce magasin spécifique
        mae_store = mean_absolute_error(store_data['Actual_Sales'], store_data['Predicted_Sales'])
        mape_store = np.mean(np.abs((store_data['Actual_Sales'] - store_data['Predicted_Sales']) / store_data['Actual_Sales'])) * 100
        st.write(f"**MAE pour le Magasin {dashboard_store_id} :** {mae_store:.2f}")
        st.write(f"**MAPE pour le Magasin {dashboard_store_id} :** {mape_store:.2f}%")
    else:
        st.info(f"Aucune donnée historique disponible pour le magasin {dashboard_store_id}.")

    # --- Tu peux ajouter ici d'autres graphiques : ---
    # Ex: importance des features (si tu l'as exportée)
    # Ex: distribution des erreurs
    # Ex: ventes moyennes par jour de la semaine pour le magasin choisi
# --- Interface utilisateur Streamlit ---
st.title("📈 Rossmann Sales Predictor")
st.write("Entrez les informations pour prédire les ventes journalières d'un magasin.")

if model is not None:
    # Créer des inputs pour les features clés
    st.sidebar.header("Paramètres de Prédiction")
    
    # Utiliser un dictionnaire pour stocker les inputs de l'utilisateur
    user_input = {}

    # Input pour la date (cela nous aidera à pré-remplir d'autres champs)
    prediction_date_raw = st.sidebar.date_input("Date de la prédiction", pd.to_datetime('2015-08-01')) # Une date après ton train set
    prediction_date = pd.Timestamp(prediction_date_raw)
    # Input pour le magasin (exemple: liste des IDs uniques de ton dataset)
    # Pour simplifier, on pourrait prendre une liste statique ou permettre une saisie
    # Pour un déploiement réel, tu aurais une liste des Store IDs uniques du dataset.
    # Pour cet exemple, on peut juste le taper
    store_id = st.sidebar.number_input("Store ID", min_value=1, max_value=1115, value=1)


    # --- Génération automatique des features basées sur la date ---
    # Ces features doivent être recréées exactement comme dans Alteryx
    year = prediction_date.year
    month = prediction_date.month
    day = prediction_date.day
    day_of_week = prediction_date.dayofweek + 1 # Python dayofweek est 0-6, Alteryx est souvent 1-7
    week_of_year = prediction_date.isocalendar().week # Semaine ISO

    is_weekend = 1 if day_of_week in [6, 7] else 0
    is_start_of_month = 1 if day <= 7 else 0 # Adapte selon ta logique Alteryx
    is_end_of_month = 1 if day >= 24 else 0 # Adapte selon ta logique Alteryx
    day_of_year = prediction_date.timetuple().tm_yday
    quarter = (month - 1) // 3 + 1


    # --- Inputs manuels pour les autres features ---
    st.sidebar.subheader("Autres Paramètres (à adapter)")
    promo = st.sidebar.selectbox("Promo", options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non", index=0)
    promo2 = st.sidebar.selectbox("Promo2", options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non", index=0)

    # Il faudrait idéalement récupérer les valeurs réelles de ces colonnes pour le magasin donné et la date
    # Pour un MVP, on peut laisser l'utilisateur les saisir ou les valoriser à des moyennes/par défauts sensés
    # C'est la partie la plus délicate pour un déploiement sans API de données.

    # Pour un MVP, on pourrait simplifier et donner des sliders ou inputs pour des valeurs typiques
    # Ou mieux : faire un mini-dataframe pour un Store ID et une Date FUTURE, et remplir les features
    # C'est pourquoi avoir un petit dataset préparé peut aider.

    # Pour l'exemple, remplissons un DataFrame avec des valeurs par défaut / saisies
    # C'est ici que tu dois mapper TOUTES tes features de ton X_train
    # La liste `feature_cols_names` doit être complète et ordonnée.

    # EXEMPLE DE REMPLISSAGE (cette partie est à ADAPTER TRÈS PRÉCISÉMENT avec TES features)
    # Tu devras remplir le dictionnaire `feature_values` pour toutes les 52 features de ton modèle
    feature_values = {
        # Features numériques
        'Store': store_id,
        'CompetitionDistance': st.sidebar.slider("Competition Distance", min_value=20, max_value=100000, value=5000),
        'CompetitionOpenSinceMonth': st.sidebar.slider("Competition Open Month", min_value=0, max_value=12, value=0),
        'CompetitionOpenSinceYear': st.sidebar.slider("Competition Open Year", min_value=0, max_value=2015, value=0),
        'Promo': promo,
        'Promo2': promo2,
        'Promo2SinceWeek': st.sidebar.slider("Promo2 Since Week", min_value=0, max_value=52, value=0),
        'Promo2SinceYear': st.sidebar.slider("Promo2 Since Year", min_value=0, max_value=2015, value=0),
        'IsWeekend': is_weekend,
        'IsStartOfMonth': is_start_of_month,
        'IsEndOfMonth': is_end_of_month,
        'DayOfYear': day_of_year,
        'Quarter': quarter,
        'HasCompetition': 1 if st.sidebar.slider("Has Competition (0=No, 1=Yes)", min_value=0, max_value=1, value=1) == 1 else 0, # Ex: si CompetitionDistance=100000, HasCompetition=0
        'DateCompetition': 0, # Calculé par la suite si nécessaire, ou valeur par défaut
        'IsPromo2ActiveMonth': 0, # Calculé par la suite
        'Year': year,

        # Features One-Hot Encoded (toutes doivent être à 0 sauf celle qui correspond)
        # Il faut s'assurer que ces noms correspondent EXACTEMENT à la sortie de ton Dummy tool
        'StoreType_a': 0, 'StoreType_b': 0, 'StoreType_c': 0, 'StoreType_d': 0,
        'Assortment_a': 0, 'Assortment_b': 0, 'Assortment_c': 0,
        'StateHoliday_0': 0, 'StateHoliday_a': 0, 'StateHoliday_b': 0, 'StateHoliday_c': 0,
        'SchoolHoliday_0': 0, 'SchoolHoliday_1': 0,
        'DayOfWeek_1': 0, 'DayOfWeek_2': 0, 'DayOfWeek_3': 0, 'DayOfWeek_4': 0, 'DayOfWeek_5': 0, 'DayOfWeek_6': 0, 'DayOfWeek_7': 0,
        'Month_1': 0, 'Month_2': 0, 'Month_3': 0, 'Month_4': 0, 'Month_5': 0, 'Month_6': 0,
        'Month_7': 0, 'Month_8': 0, 'Month_9': 0, 'Month_10': 0, 'Month_11': 0, 'Month_12': 0,
        'PromoInterval_None': 0, 'PromoInterval_Jan_Apr_Jul_Oct': 0,
        'PromoInterval_Feb_May_Aug_Nov': 0, 'PromoInterval_Mar_Jun_Sept_Dec': 0,
    }
    
    # --- Mise à jour dynamique des features One-Hot ---
    # StoreType
    store_type = st.sidebar.selectbox("Store Type", options=['a', 'b', 'c', 'd'], index=0)
    feature_values[f'StoreType_{store_type}'] = 1

    # Assortment
    assortment_type = st.sidebar.selectbox("Assortment", options=['a', 'b', 'c'], index=0)
    feature_values[f'Assortment_{assortment_type}'] = 1

    # StateHoliday
    state_holiday = st.sidebar.selectbox("State Holiday", options=['0', 'a', 'b', 'c'], index=0)
    feature_values[f'StateHoliday_{state_holiday}'] = 1

    # SchoolHoliday (attention, c'est binaire donc on peut faire un selectbox 0 ou 1)
    school_holiday = st.sidebar.selectbox("School Holiday", options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non", index=0)
    feature_values[f'SchoolHoliday_{school_holiday}'] = 1

    # DayOfWeek
    feature_values[f'DayOfWeek_{day_of_week}'] = 1

    # Month
    feature_values[f'Month_{month}'] = 1
    
    # PromoInterval (si Promo2 = 1, il faut trouver le bon intervalle)
    if promo2 == 0:
        feature_values['PromoInterval_None'] = 1
    else:
        # C'est la partie la plus complexe: il faudrait déduire l'intervalle à partir de la date et Promo2Since...
        # Pour un MVP, on peut laisser l'utilisateur choisir ou avoir une logique simplifiée
        # Ou ne pas l'utiliser si les inputs ne sont pas faciles à déduire
        # Ici, je vais simplifier pour le MVP en supposant qu'on a juste les valeurs du training
        # Si tu as une logique pour déduire l'intervalle pour une date future, implémente-la ici
        promo_interval = st.sidebar.selectbox("Promo 2 Interval (si Promo2 = Oui)", 
                                                options=['None', 'Jan,Apr,Jul,Oct', 'Feb,May,Aug,Nov', 'Mar,Jun,Sept,Dec'], index=0)
        if promo_interval != 'None':
            feature_values[f'PromoInterval_{promo_interval}'] = 1
        else: # Si l'utilisateur a choisi 'None' alors que Promo2 est 'Oui', il y a une incohérence, à gérer
            feature_values['PromoInterval_None'] = 1


    # Re-calcul de certaines features si elles dépendent d'autres inputs
    # CompetitionOpenDurationDays
    if feature_values['CompetitionOpenSinceMonth'] == 0:
        feature_values['DateCompetition'] = 0
    else:
        comp_open_date = pd.Timestamp(feature_values['CompetitionOpenSinceYear'], feature_values['CompetitionOpenSinceMonth'], 1)
        feature_values['DateCompetition'] = (pd.Timestamp(prediction_date) - comp_open_date).days

    # HasCompetition
    feature_values['HasCompetition'] = 0 if feature_values['CompetitionDistance'] == 100000 else 1

    # IsPromo2ActiveMonth (re-calcul car dépend de Promo2 et PromoInterval)
    if promo2 == 0:
        feature_values['IsPromo2ActiveMonth'] = 0
    else:
        current_month_abbr = prediction_date.strftime("%b") # Ex: 'Aug'
        # Il faudrait la PromoInterval réelle pour ce store et cette date
        # Pour le MVP, si l'utilisateur a choisi un PromoInterval
        if promo_interval != 'None' and current_month_abbr in promo_interval.split(','):
            feature_values['IsPromo2ActiveMonth'] = 1
        else:
            feature_values['IsPromo2ActiveMonth'] = 0


    # --- Création du DataFrame d'input pour la prédiction ---
    # S'assurer que l'ordre des colonnes est le même que lors de l'entraînement
    input_df = pd.DataFrame([feature_values], columns=feature_cols_names)

    st.subheader("Prédiction")
    if st.button("Prédire les Ventes"):
        if model:
            try:
                prediction = model.predict(input_df)
                st.success(f"Ventes Prédites : {prediction[0]:,.2f} unités monétaires")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
                st.write("Veuillez vérifier les inputs et l'ordre des features.")
                st.write(input_df) # Affiche le DataFrame d'entrée pour le débogage
                st.write(input_df.dtypes)
        else:
            st.warning("Le modèle n'a pas pu être chargé. Veuillez contacter l'administrateur.")

else:
    st.warning("Application en cours de chargement du modèle ou modèle non trouvé.")

# --- Footer ---
st.markdown("---")
st.markdown("Développé par EL ARCHAOUI pour le projet SmartStock - Optimisation Prédictive Logistique.")