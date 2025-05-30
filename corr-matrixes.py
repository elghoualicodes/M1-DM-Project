#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de génération des matrices de corrélation pour cryptomonnaies
====================================================================

Ce script permet de générer des matrices de corrélation pour des données 
de cryptomonnaies à partir de fichiers historiques. Il calcule les rendements 
logarithmiques et ensuite les corrélations glissantes.

Auteur: Data Mining Project
Date: Mai 2025
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import yfinance as yf

# Configuration
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (14, 8),
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white'
})

def collect_crypto_data(symbols, num_days=730):
    """
    Collecte les données historiques des cryptomonnaies
    
    Args:
        symbols: Liste des symboles de cryptomonnaies
        num_days: Nombre de jours d'historique (défaut: 730 = 2 ans)
    
    Returns:
        DataFrame avec les prix de clôture quotidiens
    """
    print("📈 Collecte des données en cours...")
    
    # Configuration des dates
    start = datetime.date.today() - datetime.timedelta(days=num_days)
    end = datetime.date.today()
    
    print(f"📅 Période : {start} → {end} ({num_days} jours)")
    
    # Téléchargement des données
    data_close = pd.DataFrame()
    success_count = 0
    
    for i, symbol in enumerate(symbols, 1):
        try:
            print(f"   [{i:2d}/{len(symbols)}] {symbol:<10} ... ", end="")
            data = yf.download(symbol, start=start, end=end, interval="1d", progress=False)
            
            if not data.empty:
                data_close[symbol] = data["Close"]
                success_count += 1
                print("✅")
            else:
                print("❌ (Données vides)")
                
        except Exception as e:
            print(f"❌ ({str(e)[:30]}...)")
    
    # Nettoyage des données
    initial_length = len(data_close)
    data_close.dropna(inplace=True)
    final_length = len(data_close)
    
    print(f"\n📊 Résumé :")
    print(f"   • Cryptos collectées : {success_count}/{len(symbols)}")
    print(f"   • Données brutes : {initial_length:,} jours")
    print(f"   • Données nettoyées : {final_length:,} jours")
    print(f"   • Données supprimées : {initial_length - final_length:,} jours")
    
    return data_close

def calculate_log_returns(price_data):
    """
    Calcule les rendements logarithmiques à partir des prix
    
    Args:
        price_data: DataFrame avec les prix de clôture
    
    Returns:
        DataFrame avec les rendements logarithmiques
    """
    print("🔢 Calcul des rendements logarithmiques...")
    
    # Calcul des rendements logarithmiques
    log_returns = np.log(price_data / price_data.shift(1))
    log_returns.dropna(inplace=True)
    
    print(f"Rendements calculés pour {len(log_returns)} jours")

    # Normalisation (centrage-réduction)
    scaler = StandardScaler()
    log_returns_normalized = pd.DataFrame(
        scaler.fit_transform(log_returns),
        index=log_returns.index,
        columns=log_returns.columns
    )
    
    # Statistiques descriptives
    print("\n📈 Statistiques des rendements normalisés:")
    print(f"Rendement moyen : {log_returns_normalized.mean().mean():.4f}")
    print(f"Volatilité moyenne : {log_returns_normalized.std().mean():.4f}")
    
    return log_returns

def validate_data(returns_data):
    """
    Valide et nettoie les données de rendements
    
    Args:
        returns_data: DataFrame des rendements
    
    Returns:
        DataFrame nettoyé ou None si problème critique
    """
    print("🔍 Validation des données...")
    
    # Vérifications de base
    if returns_data.empty:
        print("❌ Erreur: DataFrame vide")
        return None
    
    if returns_data.isnull().all().all():
        print("❌ Erreur: Toutes les valeurs sont NaN")
        return None
    
    # Statistiques sur les données manquantes
    missing_pct = (returns_data.isnull().sum() / len(returns_data)) * 100
    print(f"📊 Données manquantes par colonne:")
    for col, pct in missing_pct.items():
        if pct > 0:
            print(f"  {col}: {pct:.1f}%")
        else:
            print(f"  {col}: ✅ Aucune")
    
    # Statistiques descriptives
    print(f"\n📈 Statistiques des rendements:")
    print(f"  Période: {returns_data.index[0].date()} à {returns_data.index[-1].date()}")
    print(f"  Observations: {len(returns_data)}")
    print(f"  Cryptomonnaies: {list(returns_data.columns)}")
    
    # Vérifier la variance (éviter les séries constantes)
    zero_variance = returns_data.var() == 0
    if zero_variance.any():
        print(f"⚠️  Attention: Variance nulle détectée pour {zero_variance[zero_variance].index.tolist()}")
    
    # Nettoyer les données
    clean_data = returns_data.dropna()
    if len(clean_data) < len(returns_data) * 0.8:
        print(f"⚠️  Attention: {len(returns_data) - len(clean_data)} lignes supprimées (beaucoup de NaN)")
    
    print(f"✅ Données validées: {len(clean_data)} observations utilisables")
    return clean_data

def calculate_rolling_correlations(returns_data, window=30):
    """
    Calcule les corrélations glissantes pour analyser l'évolution temporelle
    
    Args:
        returns_data: DataFrame des rendements
        window: Taille de la fenêtre glissante (jours)
    
    Returns:
        Liste des matrices de corrélation, dates correspondantes
    """
    print(f"🔄 Calcul des corrélations glissantes (fenêtre: {window} jours)...")
    
    # Validation des données
    clean_data = validate_data(returns_data)
    if clean_data is None:
        return None, None
    
    if window > len(clean_data):
        print(f"❌ Erreur: Fenêtre ({window}) > données disponibles ({len(clean_data)})")
        return None, None
    
    rolling_corrs = []
    dates = []
    
    for i in range(window, len(clean_data) + 1):
        if i % 50 == 0 or i == window:
            progress = (i - window + 1) / (len(clean_data) - window + 1) * 100
            print(f"  Progression: {progress:.1f}%")
        
        # Calculer la corrélation pour la fenêtre actuelle
        window_data = clean_data.iloc[i-window:i]
        corr_matrix = window_data.corr()
        
        # Vérifier que la matrice n'est pas vide
        if not corr_matrix.isnull().all().all():
            rolling_corrs.append(corr_matrix)
            dates.append(clean_data.index[i-1])
    
    if not rolling_corrs:
        print("❌ Erreur: Aucune matrice de corrélation valide calculée")
        return None, None
    
    print(f"✅ {len(rolling_corrs)} matrices de corrélation calculées")
    print(f"Période couverte: {dates[0].date()} à {dates[-1].date()}")
    
    return rolling_corrs, dates

def save_correlation_matrices(rolling_correlations, dates, output_dir='output_corr'):
    """
    Sauvegarde les matrices de corrélation dans des fichiers CSV
    
    Args:
        rolling_correlations: Liste des matrices de corrélation
        dates: Dates correspondantes
        output_dir: Répertoire de sortie
    """
    print(f"💾 Sauvegarde des matrices de corrélation...")
    
    # Créer le répertoire s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  Répertoire créé: {output_dir}")
    
    # Sauvegarder chaque matrice dans un fichier distinct
    for i, (corr_matrix, date) in enumerate(zip(rolling_correlations, dates)):
        if i % 50 == 0:
            progress = (i / len(rolling_correlations)) * 100
            print(f"  Progression: {progress:.1f}%")
        
        # Format du nom de fichier: YYYY-MM-DD_correlation_matrix.csv
        filename = f"{date.strftime('%Y-%m-%d')}_correlation_matrix.csv"
        file_path = os.path.join(output_dir, filename)
        
        # Sauvegarde
        corr_matrix.to_csv(file_path)
    
    # Sauvegarder la dernière matrice dans un fichier spécial
    latest_file_path = os.path.join(output_dir, "latest_correlation_matrix.csv")
    rolling_correlations[-1].to_csv(latest_file_path)
    
    # Créer un fichier avec les dates
    dates_df = pd.DataFrame({'date': dates})
    dates_file_path = os.path.join(output_dir, "correlation_dates.csv")
    dates_df.to_csv(dates_file_path, index=False)
    
    print(f"✅ {len(rolling_correlations)} matrices sauvegardées dans {output_dir}")
    print(f"  Dernière matrice sauvegardée: {latest_file_path}")
    print(f"  Fichier des dates: {dates_file_path}")

def plot_sample_correlations(rolling_correlations, dates, n_samples=3):
    """
    Affiche des exemples de matrices de corrélation
    
    Args:
        rolling_correlations: Liste des matrices de corrélation
        dates: Dates correspondantes
        n_samples: Nombre d'exemples à afficher
    """
    if not rolling_correlations or len(rolling_correlations) < n_samples:
        print("❌ Pas assez de matrices pour afficher des exemples")
        return
    
    # Sélection d'indices uniformément répartis
    indices = [0]
    if n_samples > 2:
        step = len(rolling_correlations) // (n_samples - 1)
        indices.extend([i * step for i in range(1, n_samples - 1)])
    indices.append(-1)
    
    # Création de la figure
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 8, 7))
    
    # Colormap personnalisée
    colors = ['#d73027', '#ffffff', '#1a9850']
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', colors, N=100)
    
    for i, idx in enumerate(indices):
        corr_matrix = rolling_correlations[idx]
        date = dates[idx]
        
        # Masque pour la diagonale supérieure
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Affichage de la heatmap
        sns.heatmap(corr_matrix,
                    mask=mask,
                    annot=True,
                    cmap=cmap,
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                    fmt=".2f",
                    cbar_kws={'label': 'Corrélation'},
                    ax=axes[i],
                    linewidths=0.5)
        
        axes[i].set_title(f"Corrélation - {date.date()}", 
                         fontsize=14, fontweight='bold')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), 
                               rotation=45, ha='right')
        axes[i].set_yticklabels(axes[i].get_yticklabels(), 
                               rotation=0)
    
    plt.suptitle("Exemples de Matrices de Corrélation", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("sample_correlation_matrices.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Point d'entrée principal du script"""
    print("=" * 80)
    print("🚀 GÉNÉRATION DE MATRICES DE CORRÉLATION POUR CRYPTOMONNAIES")
    print("=" * 80)
    
    # Liste des cryptomonnaies à analyser
    crypto_symbols = [
        "BTC-USD",   # Bitcoin
        "ETH-USD",   # Ethereum
        "BNB-USD",   # Binance Coin
        "XRP-USD",   # Ripple
        "SOL-USD",   # Solana
        "ADA-USD",   # Cardano
        "DOT-USD",   # Polkadot
        "SHIB-USD",  # Shiba Inu
        "LTC-USD",   # Litecoin
        "AVAX-USD"   # Avalanche
    ]
    
    # Paramètres
    window_size = 30  # Taille de la fenêtre glissante en jours
    num_days = 730    # 2 ans de données
    output_dir = "correlation_matrices"  # Répertoire de sortie
    
    # 1. Collecte des données
    data_close = collect_crypto_data(crypto_symbols, num_days)
    
    # Sauvegarder les prix de clôture
    data_close.to_csv("crypto_prices.csv")
    print("💾 Prix de clôture sauvegardés dans crypto_prices.csv")
    
    # 2. Calcul des rendements logarithmiques
    log_returns = calculate_log_returns(data_close)
    
    # Sauvegarder les rendements logarithmiques
    log_returns.to_csv("log_returns.csv")
    print("💾 Rendements logarithmiques sauvegardés dans log_returns.csv")
    
    # 3. Calcul des corrélations glissantes
    rolling_correlations, correlation_dates = calculate_rolling_correlations(log_returns, window_size)
    
    if rolling_correlations is not None:
        # 4. Sauvegarde des matrices
        save_correlation_matrices(rolling_correlations, correlation_dates, output_dir)
        
        # 5. Visualisation d'exemples
        plot_sample_correlations(rolling_correlations, correlation_dates, n_samples=3)
    
    print("\n✅ TRAITEMENT TERMINÉ")
    print("=" * 80)

if __name__ == "__main__":
    main()
