#!/usr/bin/env python3
"""
Analyse de Corrélation des Cryptomonnaies - Script Autonome
==========================================================

Ce script effectue une analyse complète de corrélation entre cryptomonnaies
avec détection de groupes dynamiques et changements de régime.

Fonctionnalités :
- Collecte automatique de données financières
- Calcul de corrélations glissantes
- Clustering hiérarchique et spectral
- Détection automatique de changements de régime
- Visualisations interactives
- Sauvegarde automatique des résultats

Auteur: Projet Data Mining
Version: 2.0
Date: Mai 2025

Usage:
    python crypto_correlation_analysis.py
"""

import warnings
warnings.filterwarnings('ignore')

# Imports principaux
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Imports pour analyse et clustering
import networkx as nx
from itertools import combinations
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks

# Import pour données financières
import yfinance as yf

# Configuration des graphiques
plt.style.use('default')
sns.set_palette("husl")

def setup_plotting():
    """Configure les paramètres de visualisation"""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.facecolor': 'white'
    })

class CryptoCorrelationAnalyzer:
    """
    Classe principale pour l'analyse de corrélation des cryptomonnaies
    """
    
    def __init__(self, symbols, num_days=730, window=30, verbose=True):
        """
        Initialise l'analyseur
        
        Args:
            symbols: Liste des symboles de cryptomonnaies
            num_days: Nombre de jours d'historique
            window: Taille de la fenêtre glissante
            verbose: Affichage détaillé
        """
        self.symbols = symbols
        self.num_days = num_days
        self.window = window
        self.verbose = verbose
        
        # Résultats
        self.data_close = None
        self.log_returns = None
        self.rolling_correlations = None
        self.dates = None
        self.latest_correlation = None
        
        # Résultats de clustering
        self.hierarchical_clusters = None
        self.spectral_clusters = None
        self.temporal_clusters = None
        
        if self.verbose:
            print("🚀 Analyseur de Corrélation Crypto initialisé")
            print(f"   Cryptomonnaies: {len(symbols)}")
            print(f"   Période: {num_days} jours")
            print(f"   Fenêtre glissante: {window} jours")
    
    def collect_data(self):
        """Collecte les données historiques des cryptomonnaies"""
        if self.verbose:
            print("\n📈 Collecte des données en cours...")
        
        start = datetime.date.today() - datetime.timedelta(days=self.num_days)
        end = datetime.datetime.today()
        
        if self.verbose:
            print(f"Période d'analyse : {start} à {end.date()}")
        
        self.data_close = pd.DataFrame()
        
        for i, symbol in enumerate(self.symbols, 1):
            try:
                if self.verbose:
                    print(f"  [{i}/{len(self.symbols)}] Téléchargement {symbol}...")
                
                data = yf.download(symbol, start=start, end=end, 
                                 interval="1d", progress=False)
                self.data_close[symbol] = data["Close"]
                
            except Exception as e:
                if self.verbose:
                    print(f"  ❌ Erreur pour {symbol}: {e}")
        
        # Nettoyage des données
        initial_length = len(self.data_close)
        self.data_close.dropna(inplace=True)
        
        if self.verbose:
            print(f"Données collectées: {initial_length} → {len(self.data_close)} jours (après nettoyage)")
        
        return self.data_close
    
    def calculate_returns(self):
        """Calcule les rendements logarithmiques"""
        if self.verbose:
            print("\n🔢 Calcul des rendements logarithmiques...")
        
        self.log_returns = np.log(self.data_close / self.data_close.shift(1))
        self.log_returns.dropna(inplace=True)
        
        if self.verbose:
            print(f"Rendements calculés pour {len(self.log_returns)} jours")
            print(f"Rendement moyen : {self.log_returns.mean().mean():.4f}")
            print(f"Volatilité moyenne : {self.log_returns.std().mean():.4f}")
        
        return self.log_returns
    
    def calculate_rolling_correlations(self):
        """Calcule les corrélations glissantes"""
        if self.verbose:
            print(f"\n🔄 Calcul des corrélations glissantes (fenêtre: {self.window} jours)...")
        
        self.rolling_correlations = []
        
        for i in range(self.window, len(self.log_returns) + 1):
            if self.verbose and i % 50 == 0:
                progress = (i - self.window + 1) / (len(self.log_returns) - self.window + 1) * 100
                print(f"  Progression: {progress:.1f}%")
            
            corr_matrix = self.log_returns.iloc[i-self.window:i].corr()
            self.rolling_correlations.append(corr_matrix)
        
        self.dates = self.log_returns.index[self.window-1:]
        self.latest_correlation = self.rolling_correlations[-1]
        
        if self.verbose:
            print(f"✅ {len(self.rolling_correlations)} matrices calculées")
            print(f"Période: {self.dates[0].date()} à {self.dates[-1].date()}")
        
        return self.rolling_correlations, self.dates
    
    def perform_hierarchical_clustering(self, threshold=1.5):
        """Effectue un clustering hiérarchique"""
        if self.verbose:
            print("\n🌳 Clustering hiérarchique...")
        
        # Matrice de distance
        distance_matrix = np.sqrt(0.5 * (1 - self.latest_correlation))
        condensed_distances = squareform(distance_matrix)
        
        # Clustering
        linkage_matrix = linkage(condensed_distances, method='ward')
        self.hierarchical_clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')
        
        if self.verbose:
            cluster_dict = {}
            for crypto, cluster in zip(self.latest_correlation.columns, self.hierarchical_clusters):
                if cluster not in cluster_dict:
                    cluster_dict[cluster] = []
                cluster_dict[cluster].append(crypto)
            
            print("📊 Groupes hiérarchiques:")
            for cluster_id, cryptos in cluster_dict.items():
                print(f"  Groupe {cluster_id}: {', '.join(cryptos)}")
        
        return self.hierarchical_clusters, linkage_matrix
    
    def perform_spectral_clustering(self, max_clusters=6):
        """Effectue un clustering spectral optimisé"""
        if self.verbose:
            print("\n🌟 Clustering spectral avec optimisation...")
        
        similarity_matrix = (self.latest_correlation + 1) / 2
        
        cluster_range = range(2, max_clusters + 1)
        silhouette_scores = []
        all_labels = []
        
        for n_clusters in cluster_range:
            spectral_model = SpectralClustering(
                n_clusters=n_clusters, 
                affinity='precomputed', 
                random_state=42
            )
            labels = spectral_model.fit_predict(similarity_matrix)
            score = silhouette_score(similarity_matrix, labels, metric='precomputed')
            
            silhouette_scores.append(score)
            all_labels.append(labels)
            
            if self.verbose:
                print(f"  {n_clusters} clusters → Score: {score:.3f}")
        
        # Sélection optimal
        optimal_index = np.argmax(silhouette_scores)
        optimal_clusters = cluster_range[optimal_index]
        self.spectral_clusters = all_labels[optimal_index]
        
        if self.verbose:
            print(f"\n🎯 Configuration optimale: {optimal_clusters} clusters")
            
            spectral_groups = {}
            for i, crypto in enumerate(self.latest_correlation.columns):
                cluster_id = self.spectral_clusters[i]
                if cluster_id not in spectral_groups:
                    spectral_groups[cluster_id] = []
                spectral_groups[cluster_id].append(crypto)
            
            for cluster_id, cryptos in spectral_groups.items():
                print(f"  Groupe {cluster_id + 1}: {', '.join(cryptos)}")
        
        return self.spectral_clusters, optimal_clusters, silhouette_scores
    
    def perform_temporal_clustering(self, n_clusters=3):
        """Effectue un clustering temporel K-Means"""
        if self.verbose:
            print(f"\n⏰ Clustering temporel K-Means ({n_clusters} clusters)...")
        
        # Aplatissement des matrices
        def flatten_correlation_matrix(correlation_matrix):
            mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
            return correlation_matrix.values[mask]
        
        flattened_data = np.array([
            flatten_correlation_matrix(corr_matrix) 
            for corr_matrix in self.rolling_correlations
        ])
        
        # K-Means
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.temporal_clusters = kmeans_model.fit_predict(flattened_data)
        
        if self.verbose:
            unique_labels, counts = np.unique(self.temporal_clusters, return_counts=True)
            print("📊 Répartition temporelle:")
            for label, count in zip(unique_labels, counts):
                percentage = count / len(self.temporal_clusters) * 100
                print(f"  Régime {label + 1}: {count} jours ({percentage:.1f}%)")
        
        return self.temporal_clusters, kmeans_model
    
    def detect_regime_changes(self):
        """Détecte les changements de régime automatiquement"""
        if self.verbose:
            print("\n🔍 Détection des changements de régime...")
        
        # Calcul de la variance temporelle
        variance_timeline = []
        for correlation_matrix in self.rolling_correlations:
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            variance = np.nanvar(upper_triangle.values.flatten())
            variance_timeline.append(variance)
        
        variance_timeline = np.array(variance_timeline)
        
        # Détection des pics
        threshold_height = np.mean(variance_timeline) + 1.5 * np.std(variance_timeline)
        regime_change_indices, _ = find_peaks(
            variance_timeline, 
            height=threshold_height,
            distance=10
        )
        
        if self.verbose:
            print(f"📊 Changements détectés: {len(regime_change_indices)}")
            if len(regime_change_indices) > 0:
                print("📅 Dates des changements:")
                for idx in regime_change_indices:
                    date_str = self.dates[idx].strftime("%Y-%m-%d")
                    print(f"    {date_str}")
        
        return regime_change_indices, variance_timeline
    
    def save_results(self, prefix="crypto_analysis"):
        """Sauvegarde tous les résultats"""
        if self.verbose:
            print(f"\n💾 Sauvegarde des résultats...")
        
        try:
            # Matrice de corrélation finale
            self.latest_correlation.to_csv(f'{prefix}_correlation_matrix.csv')
            
            # Rendements
            self.log_returns.to_csv(f'{prefix}_log_returns.csv')
            
            # Résultats de clustering
            if all([self.hierarchical_clusters is not None, 
                   self.spectral_clusters is not None,
                   self.temporal_clusters is not None]):
                
                clustering_results = pd.DataFrame({
                    'cryptocurrency': self.latest_correlation.columns,
                    'hierarchical_cluster': self.hierarchical_clusters,
                    'spectral_cluster': self.spectral_clusters + 1,
                    'temporal_cluster_final': self.temporal_clusters[-1] + 1
                })
                clustering_results.to_csv(f'{prefix}_clustering_results.csv', index=False)
            
            if self.verbose:
                print("✅ Sauvegarde terminée avec succès")
                
        except Exception as e:
            if self.verbose:
                print(f"❌ Erreur lors de la sauvegarde: {e}")
    
    def run_complete_analysis(self):
        """Exécute l'analyse complète"""
        try:
            # Étapes principales
            self.collect_data()
            self.calculate_returns()
            self.calculate_rolling_correlations()
            
            # Analyses
            self.perform_hierarchical_clustering()
            self.perform_spectral_clustering()
            self.perform_temporal_clustering()
            self.detect_regime_changes()
            
            # Sauvegarde
            self.save_results()
            
            if self.verbose:
                print("\n🎯 ANALYSE TERMINÉE AVEC SUCCÈS!")
                print("=" * 50)
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"\n❌ ERREUR LORS DE L'ANALYSE: {e}")
            return False

def main():
    """Fonction principale"""
    print("🚀 ANALYSE DE CORRÉLATION DES CRYPTOMONNAIES")
    print("=" * 50)
    
    # Configuration
    crypto_symbols = [
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
    
    # Configuration des graphiques
    setup_plotting()
    
    # Création et exécution de l'analyseur
    analyzer = CryptoCorrelationAnalyzer(
        symbols=crypto_symbols,
        num_days=730,
        window=30,
        verbose=True
    )
    
    success = analyzer.run_complete_analysis()
    
    if success:
        print(f"\n📊 Fichiers générés:")
        print(f"  • crypto_analysis_correlation_matrix.csv")
        print(f"  • crypto_analysis_log_returns.csv")
        print(f"  • crypto_analysis_clustering_results.csv")
        print(f"\n✨ Analyse terminée avec succès!")
        
        # Génération de visualisations supplémentaires
        generate_additional_visualizations(analyzer)
        
    else:
        print(f"\n💥 L'analyse a échoué. Vérifiez les logs d'erreur.")
    
    return analyzer if success else None

def generate_additional_visualizations(analyzer):
    """Génère des visualisations supplémentaires"""
    try:
        print("\n📊 Génération de visualisations...")
        
        # 1. Heatmap de corrélation
        plt.figure(figsize=(12, 10))
        sns.heatmap(analyzer.latest_correlation, annot=True, cmap='coolwarm', 
                   center=0, fmt='.2f', square=True)
        plt.title('Matrice de Corrélation Finale des Cryptomonnaies')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Évolution temporelle des clusters
        if analyzer.temporal_clusters is not None:
            plt.figure(figsize=(14, 6))
            plt.plot(analyzer.dates, analyzer.temporal_clusters, 
                    marker='o', alpha=0.7, linewidth=2)
            plt.ylabel('Cluster Temporel')
            plt.xlabel('Date')
            plt.title('Évolution des Clusters Temporels')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('temporal_clusters.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. Réseau de corrélations fortes
        threshold = 0.6
        G = nx.Graph()
        
        for i, crypto1 in enumerate(analyzer.latest_correlation.columns):
            for j, crypto2 in enumerate(analyzer.latest_correlation.columns):
                if i < j:
                    corr_val = analyzer.latest_correlation.iloc[i, j]
                    if corr_val > threshold:
                        G.add_edge(crypto1, crypto2, weight=corr_val)
        
        if G.number_of_edges() > 0:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)
            weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
            
            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                   edge_color='gray', node_size=2000, width=weights, 
                   font_size=10, font_weight='bold')
            plt.title(f'Réseau des Corrélations Fortes (> {threshold})')
            plt.tight_layout()
            plt.savefig('correlation_network.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("✅ Visualisations générées avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération des visualisations: {e}")

if __name__ == "__main__":
    main()
