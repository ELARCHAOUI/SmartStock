# 📈 Projet SmartStock : Optimisation Prédictive Logistique pour Rossmann Store Sales

## Table des Matières
1.  [Introduction](#1-introduction)
2.  [Objectif du Projet](#2-objectif-du-projet)
3.  [Méthodologie et Stack Technologique](#3-méthodologie-et-stack-technologique)
    *   [3.1. Phase 1 : Compréhension des Données et du Problème](#31-phase-1--compréhension-des-données-et-du-problème)
    *   [3.2. Phase 2 : Préparation et Feature Engineering (Alteryx)](#32-phase-2--préparation-et-feature-engineering-alteryx)
    *   [3.3. Phase 3 : Modélisation Prédictive (Python - LightGBM)](#33-phase-3--modélisation-prédictive-python---lightgbm)
    *   [3.4. Phase 4 : Reporting et Analyse (Power BI)](#34-phase-4--reporting-et-analyse-power-bi)
    *   [3.5. Phase 5 : Déploiement de l'Application (Streamlit sur Render)](#35-phase-5--déploiement-de-lapplication-streamlit-sur-render)
4.  [Résultats et Performance du Modèle](#4-résultats-et-performance-du-modèle)
5.  [Structure du Dépôt](#5-structure-du-dépôt)
6.  [Comment Exécuter l'Application (Déployée)](#6-comment-exécuter-lapplication-déployée)
7.  [Prochaines Étapes et Améliorations](#7-prochaines-étapes-et-améliorations)
8.  [Auteur](#8-auteur)

---

## 1. Introduction

Ce projet, nommé **"SmartStock"**, est une solution complète de Data Science et d'Intelligence Artificielle appliquée à l'optimisation de la chaîne logistique pour la chaîne de pharmacies et drogueries **Rossmann**. Il s'appuie sur le dataset "Rossmann Store Sales" de Kaggle pour développer un système de **prédiction de la demande** robuste.

L'objectif est de transformer les données historiques en insights prédictifs pour permettre une gestion des stocks plus efficace et proactive.

## 2. Objectif du Projet

L'objectif principal du projet SmartStock est de :
*   **Prédire les ventes journalières** pour chaque magasin Rossmann avec la plus grande précision possible.
*   Fournir des prévisions fiables pour aider à **optimiser les niveaux de stock**, permettant ainsi de **réduire significativement les ruptures de stock** (perte de ventes) et les **coûts de surstockage** (coûts de stockage, produits périmés/invendus).
*   Développer une solution **bout-en-bout**, de l'ingestion des données brutes au déploiement d'une application interactive et à la visualisation des résultats.

## 3. Méthodologie et Stack Technologique

Le projet a suivi une approche structurée, intégrant plusieurs outils et techniques de Data Science :

### 3.1. Phase 1 : Compréhension des Données et du Problème
*   **Outils :** Explorateur de fichiers, Tableur, Alteryx Designer (`Browse` Tool).
*   **Description :** Analyse du contexte métier de Rossmann et exploration initiale des datasets `train.csv` et `store.csv` pour identifier la structure, les types de données, les valeurs manquantes et les premières tendances.

### 3.2. Phase 2 : Préparation et Feature Engineering (Alteryx)
*   **Outils :** Alteryx Designer (`Input Data`, `Join`, `Select`, `Formula`, `Multi-Field Formula`, `Filter`, `DateTime`, `Dummy`, `Output Data`).
*   **Description :**
    *   **Intégration et Nettoyage :** Jointure des données de ventes et des magasins, correction des types de données (passant de `V_String` à `Date`, `Int`, `Bool`, `Double`). Filtrage des données non pertinentes (ventes à zéro pour magasins ouverts) et gestion intelligente des valeurs manquantes.
        *   `CompetitionDistance`: Imputation des `Nulls` par `100000` (absence de concurrent).
        *   `CompetitionOpenSinceMonth/Year`, `Promo2SinceWeek/Year`: Imputation des `Nulls` par `0` (absence logique d'information).
        *   `PromoInterval`: Imputation des `Nulls` par `"None"`.
    *   **Feature Engineering Avancé :** Création de plus de **50 caractéristiques** (features) pertinentes pour le modèle, incluant :
        *   **Caractéristiques temporelles :** `Year`, `Month`, `Day`, `DayOfWeek`, `WeekOfYear`, `DayOfYear`, `Quarter`, `IsWeekend`, `IsStartOfMonth`, `IsEndOfMonth`.
        *   **Caractéristiques de concurrence :** `HasCompetition`, `CompetitionOpenDurationDays` (durée d'ouverture du concurrent en jours).
        *   **Caractéristiques promotionnelles :** `IsPromo2ActiveMonth` (indiquant si le mois est un mois de promo2 active).
        *   **Encodage catégoriel :** Transformation des variables catégorielles (`StoreType`, `Assortment`, `StateHoliday`, `SchoolHoliday`, `DayOfWeek`, `Month`, `PromoInterval`) en variables numériques binaires (One-Hot Encoding) pour la modélisation.
*   **Résultat :** Un dataset propre, enrichi et prêt à être consommé par un modèle de Machine Learning, exporté au format CSV.

### 3.3. Phase 3 : Modélisation Prédictive (Python - LightGBM)
*   **Outils :** Python, Jupyter Notebook (Google Colab), librairies `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `matplotlib`, `seaborn`, `joblib`.
*   **Description :**
    *   **Chargement et Préparation :** Importation du dataset nettoyé, séparation des features (`X`) de la cible (`y` - Sales), et division chronologique en ensembles d'entraînement et de test.
    *   **Modélisation :** Entraînement d'un modèle de régression **LightGBM (`LGBMRegressor`)**, un algorithme de boosting très performant. Utilisation de l'arrêt précoce (`early_stopping`) pour optimiser la performance et prévenir le surapprentissage.
    *   **Sauvegarde du modèle :** Persistance du modèle entraîné au format `.joblib` pour un déploiement futur.

### 3.4. Phase 4 : Reporting et Analyse (Power BI)
*   **Outils :** Power BI Desktop.
*   **Description :** Création d'un dashboard interactif pour visualiser les résultats du modèle et les insights métiers. Le dashboard permet de :
    *   Suivre la performance globale du modèle (MAPE, R², MAE).
    *   Comparer les ventes réelles et prédites pour l'ensemble de la chaîne et pour des magasins spécifiques via des filtres dynamiques.
    *   Identifier les périodes ou les magasins où le modèle est le plus/moins précis.
    *   (Ajoute ici d'autres points si tu as pu développer plus de pages, ex: analyser l'impact de certaines features).
*   **Lien vers le dashboard Power BI (fichier .pbix) :** [Lien vers ton fichier Power BI si tu le pousses, ou le lien vers Power BI Service si tu le publies]

### 3.5. Phase 5 : Déploiement de l'Application (Streamlit sur Render)
*   **Outils :** Python, Streamlit, Render.com, `joblib`.
*   **Description :** Développement et déploiement d'une application web simple pour rendre les prédictions du modèle accessibles aux utilisateurs.
    *   L'application permet de saisir une date et un ID de magasin, ainsi que d'autres paramètres clés.
    *   Elle renvoie une prédiction de ventes en temps réel en chargeant le modèle entraîné.
    *   Elle affiche également les ventes réelles passées (si disponibles) pour faciliter la contextualisation.
*   **Lien vers l'application déployée :** [https://rossmann-sales-predictor.onrender.com](https://rossmann-sales-predictor.onrender.com)

## 4. Résultats et Performance du Modèle

Le modèle LightGBM a démontré une performance robuste sur l'ensemble de test :
*   **Mean Absolute Error (MAE):** `[ta valeur, ex: 932.18]` unités monétaires.
*   **Root Mean Squared Error (RMSE):** `[ta valeur, ex: 1390.04]` unités monétaires.
*   **R-squared (R²):** `[ta valeur, ex: 0.7901]` (Le modèle explique environ 79% de la variabilité des ventes).
*   **Mean Absolute Percentage Error (MAPE):** `[ta valeur, ex: 13.42]%` (Une erreur moyenne d'environ 13% par rapport aux ventes réelles).

Ces métriques indiquent une grande précision du modèle, le rendant exploitable pour des décisions opérationnelles d'optimisation des stocks.

**Importance des Caractéristiques (Top 5) :**
Les facteurs les plus influents sur les ventes identifiés par le modèle sont :
1.  `CompetitionDistance` (Distance à la concurrence)
2.  `Store` (Identifiant unique du magasin)
3.  `DayOfYear` (Jour de l'année)
4.  `CompetitionOpenDurationDays` (Ancienneté du concurrent)
5.  `Promo` (Indicateur de promotion en cours)

Ces insights confirment le rôle prépondérant des facteurs externes (concurrence) et internes (promotions, saisonnalité) dans la dynamique des ventes.

## 5. Structure du Dépôt

*   `Alteryx/`: Contient les workflows Alteryx (`.yxmd`) utilisés pour le Data Wrangling et le Feature Engineering.
*   `Plan/`: Documentation, notes de projet.
*   `PowerBi/`: Contient le fichier Power BI Desktop (`.pbix`) du dashboard de reporting.
*   `data/`: Contient les datasets originaux et les données préparées (`.csv`) (gérés par Git LFS).
*   `models/`: Contient le modèle LightGBM entraîné (`.joblib`) (géré par Git LFS).
*   `src/`: Contient l'application Streamlit (`app.py`) et ses dépendances (`requirements.txt`).
*   `.gitattributes`: Fichier de configuration pour Git LFS.
*   `requirements.txt`: Liste des dépendances Python pour le déploiement.
*   `README.md`: Ce fichier.

## 6. Comment Exécuter l'Application (Déployée)

L'application est déployée et accessible via Render.com. Vous pouvez interagir avec elle directement :
*   **Lien de l'application :** [https://rossmann-sales-predictor.onrender.com](https://rossmann-sales-predictor.onrender.com)
    *   Changez la date, l'ID du magasin et d'autres paramètres dans le panneau latéral pour obtenir une prédiction de ventes en temps réel.

*(Instructions pour l'exécution locale si souhaité :)*
Pour exécuter l'application localement, assurez-vous d'avoir Python et les librairies listées dans `src/requirements.txt` installées.
1.  Clonez ce dépôt : `git clone https://github.com/ELARCHAOUI/SmartStock.git`
2.  Accédez au dossier de l'application : `cd SmartStock/src`
3.  Installez les dépendances : `pip install -r requirements.txt`
4.  Lancez l'application : `streamlit run app.py`

## 7. Prochaines Étapes et Améliorations

*   **Optimisation du Modèle :** Effectuer un hyperparamétrage plus poussé du modèle LightGBM (ex: via `GridSearchCV` ou `RandomizedSearchCV`) pour affiner encore la précision, potentiellement en augmentant `n_estimators`.
*   **Features Supplémentaires :** Intégrer des données externes (événements locaux, données économiques macro, informations démographiques) pour enrichir le modèle.
*   **Amélioration du Dashboard Power BI :** Ajouter plus d'interactivité et des analyses plus granulaires basées sur les importances de features.
*   **Monitoring :** Mettre en place un système de monitoring de la performance du modèle en production pour détecter la dérive et assurer une maintenance continue.
*   **Robustesse du déploiement :** Améliorer la gestion des erreurs et l'expérience utilisateur de l'application.

## 8. Auteur

Mohamed ELARCHAOUI - [[Lien vers votre LinkedIn](https://www.linkedin.com/in/mohamed-el-archaoui/)]
