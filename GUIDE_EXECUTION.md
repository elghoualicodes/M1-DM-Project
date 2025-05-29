# 🎯 GUIDE D'EXÉCUTION RAPIDE

## ⚡ Démarrage Immédiat

### 🚀 Exécution Complète (Recommandée)
```powershell
# Aller dans le répertoire du projet
cd "c:\Users\MSI\Data-maining\projet-DM"

# Exécuter l'analyse complète
python crypto_correlation_analysis.py
```

### 📓 Mode Interactif (Notebook)
```powershell
# Lancer Jupyter
jupyter notebook Explore-Kmeans.ipynb
```

## 📋 Vérification des Prérequis

### ✅ Modules Installés
Vérifier que tous les modules sont disponibles :
```powershell
python -c "import pandas, numpy, matplotlib, seaborn, yfinance, sklearn, scipy, networkx; print('✅ Tous les modules OK')"
```

### 📦 Installation si Nécessaire
```powershell
pip install -r requirements.txt
```

## 📊 Résultats Attendus

Après exécution, vous devriez voir :

### 📁 Fichiers Générés
- `crypto_analysis_correlation_matrix.csv`
- `crypto_analysis_log_returns.csv` 
- `crypto_analysis_clustering_results.csv`
- `correlation_heatmap.png`
- `temporal_clusters.png`
- `correlation_network.png`

### 📈 Sortie Console
```
🚀 ANALYSE DE CORRÉLATION DES CRYPTOMONNAIES
==================================================
🚀 Analyseur de Corrélation Crypto initialisé
   Cryptomonnaies: 9
   Période: 730 jours
   Fenêtre glissante: 30 jours

📈 Collecte des données en cours...
Période d'analyse : 2023-05-29 à 2025-05-29
  [1/9] Téléchargement ETH-USD...
  [2/9] Téléchargement BNB-USD...
  ...
Données collectées: 730 → 729 jours (après nettoyage)

🔢 Calcul des rendements logarithmiques...
Rendements calculés pour 728 jours
Rendement moyen : 0.0012
Volatilité moyenne : 0.0456

🔄 Calcul des corrélations glissantes (fenêtre: 30 jours)...
  Progression: 7.1%
  Progression: 14.3%
  ...
✅ 699 matrices calculées
Période: 2023-06-27 à 2025-05-29

🌳 Clustering hiérarchique...
📊 Groupes hiérarchiques:
  Groupe 1: ETH-USD, BNB-USD, SOL-USD
  Groupe 2: XRP-USD, ADA-USD, DOT-USD
  Groupe 3: SHIB-USD, LTC-USD, AVAX-USD

🌟 Clustering spectral avec optimisation...
  2 clusters → Score: 0.234
  3 clusters → Score: 0.456
  4 clusters → Score: 0.523
  5 clusters → Score: 0.487
  6 clusters → Score: 0.421

🎯 Configuration optimale: 4 clusters
  Groupe 1: ETH-USD, SOL-USD
  Groupe 2: BNB-USD, AVAX-USD
  Groupe 3: XRP-USD, ADA-USD, DOT-USD
  Groupe 4: SHIB-USD, LTC-USD

⏰ Clustering temporel K-Means (3 clusters)...
📊 Répartition temporelle:
  Régime 1: 245 jours (35.1%)
  Régime 2: 223 jours (31.9%)
  Régime 3: 231 jours (33.0%)

🔍 Détection des changements de régime...
📊 Changements détectés: 4
📅 Dates des changements:
    2023-08-15
    2023-11-22
    2024-03-18
    2024-09-07

💾 Sauvegarde des résultats...
✅ Sauvegarde terminée avec succès

🎯 ANALYSE TERMINÉE AVEC SUCCÈS!
==================================================

📊 Génération de visualisations...
✅ Visualisations générées avec succès!

📊 Fichiers générés:
  • crypto_analysis_correlation_matrix.csv
  • crypto_analysis_log_returns.csv
  • crypto_analysis_clustering_results.csv

✨ Analyse terminée avec succès!
```

## 🔧 Dépannage

### ❌ Erreur de Connexion Internet
```
Solution : Vérifier la connexion pour télécharger les données yfinance
```

### ❌ Module Manquant
```powershell
pip install [nom_du_module]
```

### ❌ Erreur de Mémoire
```
Solution : Réduire num_days ou window dans les paramètres
```

## 📞 Support

En cas de problème :
1. Vérifier les prérequis
2. Consulter `README.md` pour plus de détails
3. Consulter `PROJET_FINALISE.md` pour le statut complet

---
*Guide d'exécution - Projet Data Mining - Mai 2025*
