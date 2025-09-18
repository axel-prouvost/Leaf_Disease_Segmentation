import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def calculate_metrics_table(csv_file_path):
    """
    Calcule les métriques de performance pour la segmentation des maladies des feuilles
    et génère un tableau avec les colonnes rust et powdery mildew.
    """
    
    # Lire le fichier CSV
    df = pd.read_csv(csv_file_path)
    
    print("Analyse du fichier CSV:")
    print(f"Nombre total de feuilles: {len(df)}")
    print(f"Colonnes disponibles: {list(df.columns)}")
    
    # Vérifier s'il y a des données de powdery mildew (PM)
    has_powdery_mildew = df['surfGT_PM'].sum() > 0
    
    print(f"\nStatistiques des données:")
    print(f"Rust (surfGT_rust) - Min: {df['surfGT_rust'].min()}, Max: {df['surfGT_rust'].max()}, Mean: {df['surfGT_rust'].mean():.2f}")
    print(f"Powdery Mildew (surfGT_PM) - Min: {df['surfGT_PM'].min()}, Max: {df['surfGT_PM'].max()}, Mean: {df['surfGT_PM'].mean():.2f}")
    
    # Créer le tableau de métriques
    metrics_data = {
        'Metric': ['Precision', 'Accuracy', 'F1-Score', 'Surface Error (%)'],
        'Rust': [],
        'PM': []
    }
    
    # Calculer les métriques pour Rust
    rust_metrics = calculate_rust_metrics(df)
    metrics_data['Rust'] = rust_metrics
    
    # Calculer les métriques pour Powdery Mildew
    if has_powdery_mildew:
        pm_metrics = calculate_pm_metrics(df)
        metrics_data['PM'] = pm_metrics
    else:
        # Si pas de données de powdery mildew, afficher des valeurs par défaut
        metrics_data['PM'] = ['N/A', 'N/A', 'N/A', 'N/A']
    
    # Créer le DataFrame final
    metrics_df = pd.DataFrame(metrics_data)
    
    return metrics_df, has_powdery_mildew

def calculate_rust_metrics(df):
    """
    Calcule les métriques pour la détection de la rouille (rust)
    """
    # Calculer les métriques basées sur les surfaces prédites vs ground truth
    total_pred_rust = df['surfPred_rust'].sum()
    total_gt_rust = df['surfGT_rust'].sum()
    
    # Calculer l'intersection pour chaque échantillon
    intersection = np.minimum(df['surfPred_rust'], df['surfGT_rust'])
    total_intersection = intersection.sum()
    
    # Precision: intersection / surface prédite totale
    precision = total_intersection / total_pred_rust if total_pred_rust > 0 else 0
    
    # Accuracy: (Vrais Positifs + Vrais Négatifs) / Total
    # Vrais Positifs = intersection (surface correctement prédite comme malade)
    # Vrais Négatifs = surface saine correctement prédite
    vrais_positifs = intersection
    vrais_negatifs = df['surfGT_leaf'] - df['surfGT_rust'] - (df['surfPred_rust'] - intersection)
    vrais_negatifs = np.maximum(vrais_negatifs, 0)  # Éviter les valeurs négatives
    
    # Accuracy globale
    total_pixels = df['surfGT_leaf'].sum()
    total_correct = vrais_positifs.sum() + vrais_negatifs.sum()
    accuracy = total_correct / total_pixels if total_pixels > 0 else 0
    
    # F1-Score
    recall = total_intersection / total_gt_rust if total_gt_rust > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Surface Error (%): (surface prédite - surface GT) / surface GT * 100
    surface_error = ((total_pred_rust - total_gt_rust) / total_gt_rust) * 100 if total_gt_rust > 0 else 0
    
    return [f"{precision:.3f}", f"{accuracy:.3f}", f"{f1:.3f}", f"{surface_error:.2f}"]

def calculate_pm_metrics(df):
    """
    Calcule les métriques pour la détection de l'oïdium (powdery mildew)
    """
    # Calculer les métriques basées sur les surfaces prédites vs ground truth
    total_pred_pm = df['surfPred_PM'].sum()
    total_gt_pm = df['surfGT_PM'].sum()
    
    # Calculer l'intersection pour chaque échantillon
    intersection = np.minimum(df['surfPred_PM'], df['surfGT_PM'])
    total_intersection = intersection.sum()
    
    # Precision: intersection / surface prédite totale
    precision = total_intersection / total_pred_pm if total_pred_pm > 0 else 0
    
    # Accuracy: (Vrais Positifs + Vrais Négatifs) / Total
    # Vrais Positifs = intersection (surface correctement prédite comme malade)
    # Vrais Négatifs = surface saine correctement prédite
    vrais_positifs = intersection
    vrais_negatifs = df['surfGT_leaf'] - df['surfGT_PM'] - (df['surfPred_PM'] - intersection)
    vrais_negatifs = np.maximum(vrais_negatifs, 0)  # Éviter les valeurs négatives
    
    # Accuracy globale
    total_pixels = df['surfGT_leaf'].sum()
    total_correct = vrais_positifs.sum() + vrais_negatifs.sum()
    accuracy = total_correct / total_pixels if total_pixels > 0 else 0
    
    # F1-Score
    recall = total_intersection / total_gt_pm if total_gt_pm > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Surface Error (%): (surface prédite - surface GT) / surface GT * 100
    surface_error = ((total_pred_pm - total_gt_pm) / total_gt_pm) * 100 if total_gt_pm > 0 else 0
    
    return [f"{precision:.3f}", f"{accuracy:.3f}", f"{f1:.3f}", f"{surface_error:.2f}"]

def perform_linear_regression(df, disease_type):
    """
    Effectue une régression linéaire pour une maladie donnée
    """
    if disease_type == 'rust':
        gt_col = 'surfGT_rust'
        pred_col = 'surfPred_rust'
        title = 'Régression Linéaire: Rust'
        filename = 'rust_regression.png'
    elif disease_type == 'pm':
        gt_col = 'surfGT_PM'
        pred_col = 'surfPred_PM'
        title = 'Régression Linéaire: Powdery Mildew'
        filename = 'pm_regression.png'
    else:
        return None, None
    
    # Filtrer les données où il y a effectivement de la maladie (GT > 0)
    mask = df[gt_col] > 0
    if mask.sum() == 0:
        print(f"Aucune donnée de {disease_type} trouvée pour la régression.")
        return None, None
    
    filtered_df = df[mask]
    
    # Préparer les données
    X = filtered_df[gt_col].values.reshape(-1, 1)  # Ground truth
    y = filtered_df[pred_col].values  # Prédictions
    
    # Effectuer la régression linéaire
    model = LinearRegression()
    model.fit(X, y)
    
    # Prédictions du modèle
    y_pred = model.predict(X)
    
    # Calculer R²
    r2 = r2_score(y, y_pred)
    
    # Créer le graphique
    plt.figure(figsize=(10, 8))
    
    # Nuage de points
    plt.scatter(X, y, alpha=0.6, color='blue', label='Données')
    
    # Ligne de régression
    plt.plot(X, y_pred, color='red', linewidth=2, label=f'Régression (R² = {r2:.3f})')
    
    # Ligne parfaite (y=x)
    min_val = min(X.min(), y.min())
    max_val = max(X.max(), y.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='green', alpha=0.7, label='Ligne parfaite (y=x)')
    
    plt.xlabel(f'Surface Ground Truth {disease_type.upper()} (mm²)')
    plt.ylabel(f'Surface Prédite {disease_type.upper()} (mm²)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ajouter les coefficients
    plt.text(0.05, 0.95, f'Équation: y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Graphique de régression sauvegardé dans '{filename}'")
    print(f"R² pour {disease_type}: {r2:.3f}")
    print(f"Équation: y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}")
    
    return r2, model

def display_table(metrics_df, has_powdery_mildew):
    """
    Affiche le tableau de métriques de manière formatée
    """
    print("\n" + "="*80)
    print("TABLEAU DE MÉTRIQUES DE SEGMENTATION DES MALADIES DES FEUILLES")
    print("="*80)
    
    if not has_powdery_mildew:
        print("⚠️  ATTENTION: Aucune donnée de powdery mildew (PM) trouvée dans le fichier CSV.")
        print("   Toutes les valeurs surfGT_PM sont égales à 0.0")
        print()
    
    # Afficher le tableau
    print(metrics_df.to_string(index=False))
    
    # Afficher des statistiques supplémentaires
    print("\n" + "="*80)
    print("STATISTIQUES SUPPLÉMENTAIRES")
    print("="*80)
    
    df = pd.read_csv("results_normal.csv")
    
    # Statistiques pour Rust
    rust_samples = df[df['surfGT_rust'] > 0]
    print(f"Échantillons avec Rust: {len(rust_samples)}")
    if len(rust_samples) > 0:
        print(f"IoU Rust moyen (échantillons avec Rust): {rust_samples['IoU_rust'].mean():.3f}")
    
    # Statistiques pour Powdery Mildew
    pm_samples = df[df['surfGT_PM'] > 0]
    print(f"Échantillons avec Powdery Mildew: {len(pm_samples)}")
    if len(pm_samples) > 0:
        print(f"IoU PM moyen (échantillons avec PM): {pm_samples['IoU_PM'].mean():.3f}")
    
    # Statistiques pour les échantillons mixtes
    mixed_samples = df[(df['surfGT_rust'] > 0) & (df['surfGT_PM'] > 0)]
    print(f"Échantillons avec Rust ET Powdery Mildew: {len(mixed_samples)}")

def main():
    """
    Fonction principale
    """
    # Chemin vers le fichier CSV
    csv_file_path = "results_normal.csv"
    
    try:
        # Calculer les métriques
        metrics_df, has_powdery_mildew = calculate_metrics_table(csv_file_path)
        
        # Afficher le tableau
        display_table(metrics_df, has_powdery_mildew)
        
        # Sauvegarder le tableau en CSV
        output_file = "metrics_table.csv"
        metrics_df.to_csv(output_file, index=False)
        print(f"\n💾 Tableau sauvegardé dans '{output_file}'")
        
        # Effectuer les régressions linéaires
        print("\n" + "="*80)
        print("ANALYSE DE RÉGRESSION LINÉAIRE")
        print("="*80)
        
        df = pd.read_csv(csv_file_path)
        
        # Régression pour Rust
        print("\n🔍 RÉGRESSION POUR RUST:")
        r2_rust, model_rust = perform_linear_regression(df, 'rust')
        
        # Régression pour Powdery Mildew
        print("\n🔍 RÉGRESSION POUR POWDERY MILDEW:")
        r2_pm, model_pm = perform_linear_regression(df, 'pm')
        
        # Résumé des R²
        print("\n" + "="*80)
        print("RÉSUMÉ DES COEFFICIENTS DE DÉTERMINATION (R²)")
        print("="*80)
        print(f"Rust: R² = {r2_rust:.3f}" if r2_rust is not None else "Rust: Pas de données")
        print(f"Powdery Mildew: R² = {r2_pm:.3f}" if r2_pm is not None else "Powdery Mildew: Pas de données")
        
    except FileNotFoundError:
        print(f"❌ Erreur: Le fichier {csv_file_path} n'a pas été trouvé.")
    except Exception as e:
        print(f"❌ Erreur lors du traitement: {str(e)}")

if __name__ == "__main__":
    main()