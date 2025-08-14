import json
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import difflib
from sklearn.metrics import auc
from scipy.optimize import linear_sum_assignment
from itertools import permutations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
from datetime import datetime
from scipy import stats

logging.basicConfig(level=logging.INFO)

# =============================================================================
# ENUMS ET DATACLASSES
# =============================================================================

class PlotType(Enum):
    LINE = "line"
    SCATTER = "scatter" 
    BAR = "bar"
    PLOT = "plot"
    HORIZONTAL_BAR = "horizontal_bar"
    VERTICAL_BOX = "vertical_box"
    HORIZONTAL_BOX = "horizontal_box"

class MatchMethod(Enum):
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    SIMILARITY_MATCH = "similarity_match"
    SIMILARITY_MATCH_GENERIC = "similarity_match_generic"
    POSITION_MATCH = "position_match"
    POSITION_MATCH_GENERIC = "position_match_generic"
    ERROR = "error"

@dataclass
class SeriesFeatures:
    """Caractéristiques d'une série de données."""
    x_min: float
    x_max: float
    x_range: float
    y_min: float
    y_max: float
    y_range: float
    n_points: int
    x_mean: float
    y_mean: float

@dataclass
class MatchResult:
    """Résultat d'un appariement entre séries GT et extraites."""
    gt_name: str
    extracted_name: str
    method: MatchMethod
    score: float

@dataclass
class ComparisonMetrics:
    """Métriques de comparaison entre deux séries."""
    mae: float
    mae_rel: float
    left_miss_rel: float
    right_miss_rel: float
    x_cutoff: Optional[float] = None
    
    # NOUVEAU: Statistiques descriptives des données GT
    gt_num_points: int = None
    gt_x_min: float = None
    gt_x_max: float = None
    gt_x_std: float = None
    gt_y_min: float = None
    gt_y_max: float = None
    gt_y_std: float = None
    gt_x_range: float = None
    gt_y_range: float = None
    
    # NOUVEAU: Statistiques des données extraites
    extracted_num_points: int = None

    # Métriques de distribution
    skewness_diff: float = None          # Différence d'asymétrie
    kurtosis_diff: float = None          # Différence de kurtosis
    percentile_90_diff: float = None     # Différence au 90e percentile
    percentile_10_diff: float = None     # Différence au 10e percentile
    iqr_diff: float = None               # Différence d'écart interquartile

    monotonicity_preserved: float = None     # Conservation de la monotonie 
    trend_correlation: float = None          # Corrélation des tendances
    turning_points_diff: int = None          # Différence de points de retournement
    # Métriques de forme
    shape_similarity: float = None           # Similarité de forme DTW
    frechet_distance: float = None           # Distance de Fréchet
    hausdorff_distance: float = None         # Distance de Hausdorff

    # Métriques de couverture
    coverage_x: float = None                 # Couverture de l'intervalle X
    coverage_y: float = None                 # Couverture de l'intervalle Y
    data_density_ratio: float = None         # Ratio de densité des données
    missing_regions_count: int = None        # Nombre de régions manquantes

    # Métriques de qualité
    noise_level_diff: float = None           # Différence de niveau de bruit
    smoothness_diff: float = None            # Différence de lissage
    outlier_ratio_diff: float = None         # Différence de ratio d'outliers




# =============================================================================
# CLASSES UTILITAIRES
# =============================================================================
def diagnose_json_keys(json_path: str):
    """Fonction utilitaire pour diagnostiquer les clés présentes dans un fichier JSON."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            File = json.load(f)
            
        if 'task' in list(File.keys())[0]:
            data = File['task6']['output']
        else:       
            data = File
            
        Series = data[list(data.keys())[0]]
        
        for i, serie in enumerate(Series):
            if serie["data"]:
                first_point = serie["data"][0]
                available_keys = list(first_point.keys())
                
                # Identifier les clés X/Y qui seraient choisies
                x_key, y_key = DataLoader._identify_xy_keys(available_keys)
                
                # Vérifier les types de données
                sample_points = serie["data"][:3]  # Prendre 3 premiers points
                for j, point in enumerate(sample_points):
                    logging.debug(f"Serie {i}, Point {j}: {point}")
                    
        logging.info(f"Diagnostic terminé pour {json_path}")
             
    except Exception as e:
        logging.error(f"Erreur lors du diagnostic: {e}")
        pass

def sanitize_filename(filename: str) -> str:
    """Nettoyer un nom de fichier en supprimant les caractères invalides."""
    # Caractères invalides pour les noms de fichiers Windows
    invalid_chars = '<>:"/ \\|?*'
    
    # Remplacer les caractères invalides par des underscores
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Supprimer les espaces multiples et les underscores multiples
    import re
    filename = re.sub(r'[_\s]+', '_', filename)
    
    # Supprimer les underscores en début et fin
    filename = filename.strip('_')
    
    # Limiter la longueur (Windows a une limite de 260 caractères pour le chemin complet)
    if len(filename) > 100:
        filename = filename[:100]
    
    return filename

def calculate_gt_statistics(x_data, y_data, x_type='numeric') -> dict:
    """Calculer les statistiques descriptives des données GT."""
    
    # Conversion en numpy pour les calculs
    y_array = np.array(y_data)
    
    stats = {
        'gt_num_points': len(y_data),
        'gt_y_min': float(y_array.min()),
        'gt_y_max': float(y_array.max()),
        'gt_y_std': float(y_array.std()) if len(y_array) > 1 else 0.0,
        'gt_y_range': float(y_array.max() - y_array.min())
    }
    
    # Statistiques X seulement si numérique
    if x_type == 'numeric':
        x_array = np.array(x_data)
        stats.update({
            'gt_x_min': float(x_array.min()),
            'gt_x_max': float(x_array.max()),
            'gt_x_std': float(x_array.std()) if len(x_array) > 1 else 0.0,
            'gt_x_range': float(x_array.max() - x_array.min())
        })
    else:
        # Pour X catégoriel, on peut compter les catégories uniques
        unique_x = len(set(x_data))
        stats.update({
            'gt_x_min': np.nan,  # Non applicable pour catégoriel
            'gt_x_max': np.nan,
            'gt_x_std': np.nan,
            'gt_x_range': float(unique_x)  # Nombre de catégories uniques
        })
    
    return stats

def calculate_extracted_statistics(x_data, y_data) -> dict:
    """Calculer les statistiques des données extraites (nombre de points principalement)."""
    
    stats = {
        'extracted_num_points': len(y_data)
    }
    
    return stats

def calculate_distribution_metrics(gt_data, extracted_data):
    """Calculer les métriques de distribution."""
    from scipy import stats
    
    gt_array = np.array(gt_data)
    ext_array = np.array(extracted_data)
    
    return {
        'skewness_diff': abs(stats.skew(ext_array) - stats.skew(gt_array)),
        'kurtosis_diff': abs(stats.kurtosis(ext_array) - stats.kurtosis(gt_array)),
        'percentile_90_diff': abs(np.percentile(ext_array, 90) - np.percentile(gt_array, 90)),
        'percentile_10_diff': abs(np.percentile(ext_array, 10) - np.percentile(gt_array, 10)),
        'iqr_diff': abs(np.percentile(ext_array, 75) - np.percentile(gt_array, 25) - 
                       (np.percentile(gt_array, 75) - np.percentile(gt_array, 25)))
    }

def calculate_trend_metrics(x_gt, y_gt, x_ext, y_ext):
    """Calculer les métriques de tendance."""
    
    # 1. Monotonie
    def is_monotonic(y):
        return np.all(np.diff(y) >= 0) or np.all(np.diff(y) <= 0)
    
    mono_gt = is_monotonic(y_gt)
    mono_ext = is_monotonic(y_ext)
    monotonicity_preserved = 1.0 if mono_gt == mono_ext else 0.0
    
    # 2. Corrélation des gradients
    if len(y_gt) > 1 and len(y_ext) > 1:
        grad_gt = np.gradient(y_gt)
        grad_ext = np.gradient(y_ext)
        
        # Interpoler pour même longueur
        if len(grad_gt) != len(grad_ext):
            min_len = min(len(grad_gt), len(grad_ext))
            grad_gt = np.interp(np.linspace(0, 1, min_len), 
                               np.linspace(0, 1, len(grad_gt)), grad_gt)
            grad_ext = np.interp(np.linspace(0, 1, min_len), 
                                np.linspace(0, 1, len(grad_ext)), grad_ext)
        
        trend_correlation = np.corrcoef(grad_gt, grad_ext)[0, 1]
    else:
        trend_correlation = np.nan
    
    # 3. Points de retournement
    def count_turning_points(y):
        if len(y) < 3:
            return 0
        grad = np.gradient(y)
        return np.sum(np.diff(np.sign(grad)) != 0)
    
    turning_points_diff = abs(count_turning_points(y_ext) - count_turning_points(y_gt))
    
    return {
        'monotonicity_preserved': monotonicity_preserved,
        'trend_correlation': trend_correlation if not np.isnan(trend_correlation) else 0.0,
        'turning_points_diff': turning_points_diff
    }
def calculate_shape_metrics(x_gt, y_gt, x_ext, y_ext):
    """Calculer les métriques de forme."""
    try:
        from scipy.spatial.distance import directed_hausdorff
        
        # 1. Distance de Hausdorff
        gt_points = np.column_stack([x_gt, y_gt])
        ext_points = np.column_stack([x_ext, y_ext])
        hausdorff_dist = max(directed_hausdorff(gt_points, ext_points)[0],
                            directed_hausdorff(ext_points, gt_points)[0])
        
        # 2. Similarité cosinus des séries normalisées
        y_gt_norm = (y_gt - np.mean(y_gt)) / np.std(y_gt) if np.std(y_gt) > 0 else y_gt
        y_ext_norm = (y_ext - np.mean(y_ext)) / np.std(y_ext) if np.std(y_ext) > 0 else y_ext
        
        # Interpoler pour même longueur
        if len(y_gt_norm) != len(y_ext_norm):
            min_len = min(len(y_gt_norm), len(y_ext_norm))
            y_gt_norm = np.interp(np.linspace(0, 1, min_len), 
                                 np.linspace(0, 1, len(y_gt_norm)), y_gt_norm)
            y_ext_norm = np.interp(np.linspace(0, 1, min_len), 
                                  np.linspace(0, 1, len(y_ext_norm)), y_ext_norm)
        
        cosine_sim = np.dot(y_gt_norm, y_ext_norm) / (np.linalg.norm(y_gt_norm) * np.linalg.norm(y_ext_norm))
        
        return {
            'hausdorff_distance': hausdorff_dist,
            'shape_similarity': cosine_sim if not np.isnan(cosine_sim) else 0.0
        }
    except Exception as e:
        return {
            'hausdorff_distance': np.nan,
            'shape_similarity': np.nan
        }
def calculate_coverage_metrics(x_gt, y_gt, x_ext, y_ext):
    """Calculer les métriques de couverture."""
    
    # 1. Couverture X
    x_gt_range = max(x_gt) - min(x_gt) if len(x_gt) > 1 else 1
    x_ext_range = max(x_ext) - min(x_ext) if len(x_ext) > 1 else 1
    
    overlap_start = max(min(x_gt), min(x_ext))
    overlap_end = min(max(x_gt), max(x_ext))
    overlap_range = max(0, overlap_end - overlap_start)
    
    coverage_x = overlap_range / x_gt_range if x_gt_range > 0 else 0
    
    # 2. Couverture Y
    y_gt_range = max(y_gt) - min(y_gt) if len(y_gt) > 1 else 1
    y_ext_range = max(y_ext) - min(y_ext) if len(y_ext) > 1 else 1
    
    y_overlap_start = max(min(y_gt), min(y_ext))
    y_overlap_end = min(max(y_gt), max(y_ext))
    y_overlap_range = max(0, y_overlap_end - y_overlap_start)
    
    coverage_y = y_overlap_range / y_gt_range if y_gt_range > 0 else 0
    
    # 3. Ratio de densité
    gt_density = len(x_gt) / x_gt_range if x_gt_range > 0 else 0
    ext_density = len(x_ext) / x_ext_range if x_ext_range > 0 else 0
    data_density_ratio = ext_density / gt_density if gt_density > 0 else np.inf
    
    # 4. Régions manquantes (approximation)
    x_gt_sorted = np.sort(x_gt)
    gaps_gt = np.diff(x_gt_sorted)
    threshold = np.percentile(gaps_gt, 75) * 2  # Seuil adaptatif
    missing_regions = np.sum(gaps_gt > threshold)
    
    return {
        'coverage_x': coverage_x,
        'coverage_y': coverage_y,
        'data_density_ratio': data_density_ratio,
        'missing_regions_count': missing_regions
    }
def calculate_quality_metrics(y_gt, y_ext):
    """Calculer les métriques de qualité."""
    
    # 1. Niveau de bruit (écart-type des différences secondes)
    def noise_level(y):
        if len(y) < 3:
            return 0
        second_diff = np.diff(y, n=2)
        return np.std(second_diff)
    
    noise_gt = noise_level(y_gt)
    noise_ext = noise_level(y_ext)
    noise_level_diff = abs(noise_ext - noise_gt)
    
    # 2. Lissage (variance des gradients)
    def smoothness(y):
        if len(y) < 2:
            return 0
        gradients = np.gradient(y)
        return np.var(gradients)
    
    smooth_gt = smoothness(y_gt)
    smooth_ext = smoothness(y_ext)
    smoothness_diff = abs(smooth_ext - smooth_gt)
    
    # 3. Outliers (méthode IQR)
    def outlier_ratio(y):
        if len(y) < 4:
            return 0
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = np.sum((y < lower_bound) | (y > upper_bound))
        return outliers / len(y)
    
    outlier_gt = outlier_ratio(y_gt)
    outlier_ext = outlier_ratio(y_ext)
    outlier_ratio_diff = abs(outlier_ext - outlier_gt)
    
    return {
        'noise_level_diff': noise_level_diff,
        'smoothness_diff': smoothness_diff,
        'outlier_ratio_diff': outlier_ratio_diff
    }
def calculate_composite_similarity_score(metrics: ComparisonMetrics) -> float:
    """Calculer un score composite de similarité (0-1, 1 = parfait)."""
    
    scores = []
    weights = []
    
    # MAE relative (inversé)
    if not np.isnan(metrics.mae_rel):
        mae_score = max(0, 1 - metrics.mae_rel)  # Plus MAE est faible, mieux c'est
        scores.append(mae_score)
        weights.append(0.3)
    
    # Corrélation des tendances
    if hasattr(metrics, 'trend_correlation') and not np.isnan(metrics.trend_correlation):
        scores.append((metrics.trend_correlation + 1) / 2)  # Normaliser -1,1 vers 0,1
        weights.append(0.2)
    
    # Similarité de forme
    if hasattr(metrics, 'shape_similarity') and not np.isnan(metrics.shape_similarity):
        scores.append((metrics.shape_similarity + 1) / 2)
        weights.append(0.2)
    
    # Couverture
    if hasattr(metrics, 'coverage_x') and not np.isnan(metrics.coverage_x):
        scores.append(metrics.coverage_x)
        weights.append(0.15)
    
    # Conservation de la monotonie
    if hasattr(metrics, 'monotonicity_preserved'):
        scores.append(metrics.monotonicity_preserved)
        weights.append(0.15)
    
    if scores:
        return np.average(scores, weights=weights)
    else:
        return 0.0
class DataLoader:
    """Classe responsable du chargement des données JSON."""
    
    @staticmethod
    def load_series(json_path: str) -> pd.DataFrame:
        """Charger les séries de données depuis un fichier JSON."""
        DF = [['name', 'x', 'y', 'x_type', 'x_key', 'y_key']]  # Ajouter les clés utilisées
        
        with open(json_path, "r", encoding="utf-8") as f:
            File = json.load(f)
            
            if 'task' in list(File.keys())[0]:
                data = File['task6']['output']
            else:       
                data = File
                
            name = data.keys()
            if len(name) > 2:
                raise ValueError(f"Expected one item {json_path}, found {len(name)}")
                
            Series = data[list(name)[0]]
            
            for i, serie in enumerate(Series):
                DF.append([])
                
                # NOUVEAU: Détecter automatiquement les clés X et Y disponibles
                if not serie["data"]:
                    continue
                
                # Examiner le premier point pour déterminer les clés disponibles
                first_point = serie["data"][0]
                available_keys = list(first_point.keys())
                
                # Identifier les clés X et Y
                x_key, y_key = DataLoader._identify_xy_keys(available_keys)
                
                if x_key is None or y_key is None:
                    continue
                
                
                # Extraire les données avec les bonnes clés
                x_values = []
                y_values = []
                x_type = 'numeric'  # Par défaut
                
                for point in serie["data"]:
                    if x_key not in point or y_key not in point:
                        continue
                    
                    # Essayer de convertir X en float
                    try:
                        x_val = float(str(point[x_key]).replace(',', '.').strip())
                        x_values.append(x_val)
                    except (ValueError, TypeError):
                        # Si échec, c'est catégoriel
                        x_values.append(str(point[x_key]))
                        x_type = 'categorical'
                    
                    # Y doit toujours être numérique
                    try:
                        y_val = float(str(point[y_key]).replace(',', '.').strip())
                        y_values.append(y_val)
                    except (ValueError, TypeError):
                        continue
                
                # Si on a détecté du catégoriel, garder toutes les X en string
                if x_type == 'categorical':
                    x_values = [str(point[x_key]) for point in serie["data"] 
                               if x_key in point and len(str(point[x_key]).strip()) > 0]
                
                # Vérifier qu'on a des données
                if not x_values or not y_values or len(x_values) != len(y_values):
                    continue
                
                DF[i+1].append(serie["name"])
                DF[i+1].append(x_values)
                DF[i+1].append(y_values)
                DF[i+1].append(x_type)
                DF[i+1].append(x_key)  # Sauvegarder la clé X utilisée
                DF[i+1].append(y_key)  # Sauvegarder la clé Y utilisée
                
        # Filtrer les lignes vides
        DF_filtered = [DF[0]]  # Headers
        for row in DF[1:]:
            if len(row) == 6:  # Ligne complète
                DF_filtered.append(row)
        
        DF = pd.DataFrame(DF_filtered[1:], columns=DF_filtered[0])
        return DF
    
    @staticmethod
    def _identify_xy_keys(available_keys: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """Identifier les clés X et Y parmi les clés disponibles."""
        
        # Priorités pour les clés X
        x_priorities = ['x', 'x1', 'x2', 'X', 'time', 'date', 'category']
        
        # Priorités pour les clés Y  
        y_priorities = ['y', 'y1', 'y2', 'y3', 'Y', 'value', 'val', 'data']
        
        # Trouver la meilleure clé X
        x_key = None
        for priority in x_priorities:
            if priority in available_keys:
                x_key = priority
                break
        
        # Si aucune priorité trouvée, prendre la première clé qui n'est pas Y
        if x_key is None:
            for key in available_keys:
                if key.lower() not in ['y', 'y1', 'y2', 'y3', 'value', 'val', 'data']:
                    x_key = key
                    break
        
        # Trouver la meilleure clé Y
        y_key = None
        for priority in y_priorities:
            if priority in available_keys:
                y_key = priority
                break
        
        # Si aucune priorité trouvée, prendre la première clé qui n'est pas X
        if y_key is None:
            for key in available_keys:
                if key != x_key and key.lower() not in ['x', 'x1', 'x2', 'time', 'date', 'category']:
                    y_key = key
                    break
        
        return x_key, y_key
    
    @staticmethod
    def extract_gt_plot_type(gt_json_path: str) -> str:
        """Extraire le type de plot depuis le fichier GT JSON."""
        try:
            with open(gt_json_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
            
            # Extraire le chart_type depuis task1
            if "task1" in gt_data and "output" in gt_data["task1"]:
                chart_type = gt_data["task1"]["output"].get("chart_type", "unknown")
                return chart_type
            else:
                return "unknown"
                
        except Exception as e:
            return "unknown"

class BoxPlotDataLoader:

    """Loader spécialisé pour les données de box plot."""
    
    @staticmethod
    def load_boxplot_series(json_path: str) -> pd.DataFrame:
        """Charger les séries de box plot avec gestion flexible des structures."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                File = json.load(f)
            
            
            # Navigation flexible dans la structure JSON
            if 'task6' in File:
                data = File['task6']['output']
                Series = data[list(data.keys())[0]]
                # NOUVEAU: Analyser la structure pour déterminer le format
                return BoxPlotDataLoader._load_from_task6_structure(Series)
                
            elif 'data_series' in File:
                # Structure directe - peut être incorrectement interprétée par l'extracteur
                series_list = File['data_series']
                
                # NOUVEAU: Détecter si c'est vraiment des séries multiples ou des boxes d'une série
                return BoxPlotDataLoader._load_from_data_series_structure(series_list)
                
            else:
                first_key = list(File.keys())[0]
                data = File[first_key]
                
                if isinstance(data, list):
                    return BoxPlotDataLoader._load_from_data_series_structure(data)
                else:
                    # Structure avec clés
                    Series = data[list(data.keys())[0]]
                    return BoxPlotDataLoader._load_from_task6_structure(Series)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return pd.DataFrame(columns=['name', 'x', 'y', 'x_type', 'x_key', 'y_key'])
    
    @staticmethod
    def _load_from_task6_structure(series_list):
        """Charger depuis la structure task6 (GT format)."""
        results = []
        
        for i, serie in enumerate(series_list):
            try:
                serie_name = serie.get("name", f"BoxSerie_{i}")
                serie_data = serie.get("data", [])
                
                
                if not serie_data:
                    continue
                
                x_values = []
                box_data = []
                
                for j, box in enumerate(serie_data):
                    # Extraire la catégorie
                    category = box.get('x', f'Box_{j+1}')
                    x_values.append(str(category))
                    
                    # Extraire les statistiques de box
                    box_stats = BoxPlotDataLoader._extract_box_stats(box)
                    if box_stats:
                        box_data.append(box_stats)
                
                if x_values and box_data:
                    results.append({
                        'name': serie_name,
                        'x': x_values,
                        'y': box_data,
                        'x_type': 'categorical',
                        'x_key': 'x',
                        'y_key': 'box_data'
                    })
                
            except Exception as e:
                continue
        
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(columns=['name', 'x', 'y', 'x_type', 'x_key', 'y_key'])
    
    @staticmethod
    def _load_from_data_series_structure(series_list):
        """Charger depuis la structure data_series (Extracted format)."""
        
        # NOUVEAU: Détecter si ce sont vraiment des séries ou des boxes d'une série
        
        # Cas 1: Chaque élément est une série avec plusieurs boxes
        has_multiple_boxes = any(
            len(serie.get('data', [])) > 1 
            for serie in series_list 
            if isinstance(serie, dict) and 'data' in serie
        )
        
        # Cas 2: Chaque élément est en fait une box d'une seule série
        all_single_boxes = all(
            len(serie.get('data', [])) == 1 
            for serie in series_list 
            if isinstance(serie, dict) and 'data' in serie
        )
        
        if all_single_boxes and len(series_list) > 1:
            return BoxPlotDataLoader._reconstruct_single_series(series_list)
        else:
            return BoxPlotDataLoader._load_multiple_series(series_list)
    
    @staticmethod
    def _reconstruct_single_series(series_list):
        """Reconstituer une série unique à partir de fausses séries multiples."""
        x_values = []
        box_data = []
        
        for serie in series_list:
            try:
                # Le nom de la "série" est en fait la catégorie de la box
                category = serie.get('name', 'Unknown')
                data = serie.get('data', [])
                
                if data and len(data) == 1:
                    box = data[0]
                    x_values.append(str(category))
                    
                    box_stats = BoxPlotDataLoader._extract_box_stats(box)
                    if box_stats:
                        box_data.append(box_stats)
                
            except Exception as e:
                continue
        
        if x_values and box_data:
            results = [{
                'name': 'Reconstructed BoxPlot Series',
                'x': x_values,
                'y': box_data,
                'x_type': 'categorical',
                'x_key': 'x',
                'y_key': 'box_data'
            }]
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(columns=['name', 'x', 'y', 'x_type', 'x_key', 'y_key'])
    
    @staticmethod
    def _load_multiple_series(series_list):
        """Charger plusieurs séries de box plots."""
        results = []
        
        for i, serie in enumerate(series_list):
            try:
                serie_name = serie.get("name", f"BoxSerie_{i}")
                serie_data = serie.get("data", [])
                
                
                if not serie_data:
                    continue
                
                x_values = []
                box_data = []
                
                for j, box in enumerate(serie_data):
                    category = box.get('x', f'Box_{j+1}')
                    x_values.append(str(category))
                    
                    box_stats = BoxPlotDataLoader._extract_box_stats(box)
                    if box_stats:
                        box_data.append(box_stats)
                
                if x_values and box_data:
                    results.append({
                        'name': serie_name,
                        'x': x_values,
                        'y': box_data,
                        'x_type': 'categorical',
                        'x_key': 'x',
                        'y_key': 'box_data'
                    })
                
            except Exception as e:
                print(f"ERROR BoxPlot MULTI: Erreur série {i}: {e}")
                continue
        
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(columns=['name', 'x', 'y', 'x_type', 'x_key', 'y_key'])
    
    @staticmethod
    def _extract_box_stats(box):
        """Extraire les statistiques d'une box de manière flexible."""
        key_mapping = {
            'min': ['min', 'minimum', 'q0'],
            'max': ['max', 'maximum', 'q4'],
            'median': ['median', 'q2', 'med'],
            'first_quartile': ['first_quartile', 'q1', 'lower_quartile'],
            'third_quartile': ['third_quartile', 'q3', 'upper_quartile']
        }
        
        box_stats = {}
        
        for standard_key, possible_keys in key_mapping.items():
            value = None
            for key in possible_keys:
                if key in box:
                    value = box[key]
                    break
            box_stats[standard_key] = value
        
        # Vérifier qu'on a au moins 2 statistiques valides
        valid_stats = sum(1 for v in box_stats.values() if v is not None)
        if valid_stats >= 2:
            return box_stats
        else:
            return None

class NameAnalyzer:
    """Classe pour analyser les noms de séries."""
    
    GENERIC_PATTERNS = [
        'unnamed data series',
        'series',
        'data series', 
        'curve',
        'line',
        'dataset'
    ]
    
    @classmethod
    def is_generic_name(cls, name: str) -> bool:
        """Déterminer si un nom de série est générique/automatique."""
        name_lower = name.lower()
        
        for pattern in cls.GENERIC_PATTERNS:
            if pattern in name_lower:
                remaining = name_lower.replace(pattern, '').strip()
                if remaining.startswith('#') or remaining.isdigit() or len(remaining) <= 2:
                    return True
        return False
    
    @classmethod
    def has_mostly_generic_names(cls, names: List[str], threshold: float = 0.7) -> bool:
        """Vérifier si la majorité des noms sont génériques."""
        if not names:
            return False
        
        generic_count = sum(1 for name in names if cls.is_generic_name(name))
        ratio = generic_count / len(names)
        return ratio >= threshold

class FeatureExtractor:
    """Classe pour extraire les caractéristiques des séries de données."""
    
    @staticmethod
    def extract_features(x: List[float], y: List[float]) -> SeriesFeatures:
        """Extraire les caractéristiques numériques d'une série."""
        x = np.array(x)
        y = np.array(y)
        
        return SeriesFeatures(
            x_min=x.min(),
            x_max=x.max(),
            x_range=x.max() - x.min(),
            y_min=y.min(),
            y_max=y.max(),
            y_range=y.max() - y.min(),
            n_points=len(x),
            x_mean=x.mean(),
            y_mean=y.mean()
        )
    
    @staticmethod
    def calculate_similarity(features1: SeriesFeatures, features2: SeriesFeatures) -> float:
        """Calculer le score de similarité entre deux ensembles de caractéristiques."""
        weights = {
            'x_range': 0.3,
            'y_range': 0.3,
            'n_points': 0.2,
            'x_mean': 0.1,
            'y_mean': 0.1
        }
        
        similarity = 0.0
        
        for feature, weight in weights.items():
            val1 = getattr(features1, feature)
            val2 = getattr(features2, feature)
            
            if val1 == 0 and val2 == 0:
                feature_sim = 1.0
            elif val1 == 0 or val2 == 0:
                feature_sim = 0.0
            else:
                diff = abs(val1 - val2)
                max_val = max(abs(val1), abs(val2))
                feature_sim = 1.0 - (diff / max_val)
            
            similarity += weight * feature_sim
        
        return similarity

# =============================================================================
# CLASSES DE MATCHING
# =============================================================================

class SeriesMatcher:
    """Classe responsable de l'appariement des séries."""
    
    def __init__(self, gt_df: pd.DataFrame, extracted_df: pd.DataFrame):
        self.gt_df = gt_df
        self.extracted_df = extracted_df
        self.gt_names = gt_df['name'].tolist()
        self.extracted_names = extracted_df['name'].tolist()
        self.feature_extractor = FeatureExtractor()
        
    def match_series(self) -> List[MatchResult]:
        """Trouver l'appariement optimal entre les séries."""
        # Détecter si les noms sont génériques
        if self._should_skip_name_matching():
            return self._similarity_matching_only()
        
        # Procédure normale : exact -> fuzzy -> similarity
        matched_pairs = []
        unmatched_gt = self.gt_names.copy()
        unmatched_extracted = self.extracted_names.copy()
        
        # 1. Exact matching
        exact_matches = self._exact_matching(unmatched_gt, unmatched_extracted)
        matched_pairs.extend(exact_matches)
        
        # 2. Fuzzy matching
        fuzzy_matches = self._fuzzy_matching(unmatched_gt, unmatched_extracted)
        matched_pairs.extend(fuzzy_matches)
        
        # 3. Similarity matching
        similarity_matches = self._similarity_matching(unmatched_gt, unmatched_extracted)
        matched_pairs.extend(similarity_matches)
        
        return matched_pairs
    
    def _should_skip_name_matching(self) -> bool:
        """Déterminer si on doit passer directement au similarity matching."""
        gt_generic = NameAnalyzer.has_mostly_generic_names(self.gt_names)
        extracted_generic = NameAnalyzer.has_mostly_generic_names(self.extracted_names)
        return gt_generic or extracted_generic
    
    def _exact_matching(self, unmatched_gt: List[str], unmatched_extracted: List[str]) -> List[MatchResult]:
        """Appariement exact des noms."""
        matches = []
        extracted_set = set(unmatched_extracted)
        
        for gt_name in unmatched_gt.copy():
            if gt_name in extracted_set:
                matches.append(MatchResult(gt_name, gt_name, MatchMethod.EXACT_MATCH, 1.0))
                unmatched_gt.remove(gt_name)
                unmatched_extracted.remove(gt_name)
                extracted_set.remove(gt_name)
        
        return matches

    def _extract_boxplot_features(self, x_data, box_data):
        """Extraire les features pour les box plots."""
        # Collecter toutes les valeurs numériques des boxes
        all_values = []
        for box in box_data:
            values = [box.get('min'), box.get('max'), box.get('median'),
                     box.get('first_quartile'), box.get('third_quartile')]
            all_values.extend([v for v in values if v is not None])
        
        if not all_values:
            # Valeurs par défaut si pas de données
            return SeriesFeatures(0, 1, 1, 0, 1, 1, len(box_data), 0.5, 0.5)
        
        y_array = np.array(all_values)
        x_array = np.arange(len(box_data))  # Position des boxes
        
        return SeriesFeatures(
            x_min=float(x_array.min()),
            x_max=float(x_array.max()),
            x_range=float(x_array.max() - x_array.min()) if len(x_array) > 1 else 1.0,
            y_min=float(y_array.min()),
            y_max=float(y_array.max()),
            y_range=float(y_array.max() - y_array.min()),
            n_points=len(box_data),  # Nombre de boxes
            x_mean=float(x_array.mean()),
            y_mean=float(y_array.mean())
        )
    
    def _fuzzy_matching(self, unmatched_gt: List[str], unmatched_extracted: List[str]) -> List[MatchResult]:
        """Appariement flou des noms."""
        matches = []
        
        for gt_name in unmatched_gt.copy():
            best_match, best_score = self._find_best_fuzzy_match(gt_name, unmatched_extracted)
            
            if best_match and best_score >= 0.3:
                matches.append(MatchResult(gt_name, best_match, MatchMethod.FUZZY_MATCH, best_score))
                unmatched_gt.remove(gt_name)
                unmatched_extracted.remove(best_match)
        
        return matches
    
    def _find_best_fuzzy_match(self, gt_name: str, candidates: List[str]) -> Tuple[Optional[str], float]:
        """Trouver le meilleur match flou pour un nom donné."""
        for cutoff in [i / 100 for i in range(95, 25, -5)]:
            matches = difflib.get_close_matches(gt_name, candidates, n=1, cutoff=cutoff)
            if matches:
                return matches[0], cutoff
        return None, 0.0
    
    def _similarity_matching(self, unmatched_gt: List[str], unmatched_extracted: List[str]) -> List[MatchResult]:
        """Appariement basé sur la similarité des données."""
        if not unmatched_gt or not unmatched_extracted:
            return []
        
        try:
            similarity_matrix = self._compute_similarity_matrix(unmatched_gt, unmatched_extracted)
            row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
            
            matches = []
            for row_idx, col_idx in zip(row_indices, col_indices):
                if row_idx < len(unmatched_gt) and col_idx < len(unmatched_extracted):
                    similarity_score = similarity_matrix[row_idx, col_idx]
                    if similarity_score > 0.2:
                        gt_name = unmatched_gt[row_idx]
                        extracted_name = unmatched_extracted[col_idx]
                        matches.append(MatchResult(gt_name, extracted_name, MatchMethod.SIMILARITY_MATCH, similarity_score))
            
            return matches
            
        except Exception as e:
            print(f"Erreur dans similarity matching: {e}")
            return self._position_fallback(unmatched_gt, unmatched_extracted)
    
    def _similarity_matching_only(self) -> List[MatchResult]:
        """Appariement uniquement basé sur la similarité (pour noms génériques)."""
        try:
            similarity_matrix = self._compute_similarity_matrix(self.gt_names, self.extracted_names)
            row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
            
            matches = []
            for row_idx, col_idx in zip(row_indices, col_indices):
                if row_idx < len(self.gt_names) and col_idx < len(self.extracted_names):
                    similarity_score = similarity_matrix[row_idx, col_idx]
                    if similarity_score > 0.1:
                        gt_name = self.gt_names[row_idx]
                        extracted_name = self.extracted_names[col_idx]
                        matches.append(MatchResult(gt_name, extracted_name, MatchMethod.SIMILARITY_MATCH_GENERIC, similarity_score))
            
            return matches
            
        except Exception as e:
            print(f"Erreur dans similarity matching générique: {e}")
            return self._position_fallback(self.gt_names, self.extracted_names)
    
    def _compute_similarity_matrix(self, gt_names: List[str], extracted_names: List[str]) -> np.ndarray:
        """Calculer la matrice de similarité avec gestion des box plots."""
        similarity_matrix = np.zeros((len(gt_names), len(extracted_names)))
        
        for i, gt_name in enumerate(gt_names):
            gt_row = self.gt_df[self.gt_df['name'] == gt_name].iloc[0]
            
            # NOUVEAU: Gestion spéciale pour les box plots
            if isinstance(gt_row['y'], list) and len(gt_row['y']) > 0 and isinstance(gt_row['y'][0], dict):
                # Box plot data
                gt_features = self._extract_boxplot_features(gt_row['x'], gt_row['y'])
            else:
                # Data normale
                gt_features = self.feature_extractor.extract_features(gt_row['x'], gt_row['y'])
            
            for j, extracted_name in enumerate(extracted_names):
                extracted_row = self.extracted_df[self.extracted_df['name'] == extracted_name].iloc[0]
                
                # NOUVEAU: Gestion spéciale pour les box plots
                if isinstance(extracted_row['y'], list) and len(extracted_row['y']) > 0 and isinstance(extracted_row['y'][0], dict):
                    # Box plot data
                    extracted_features = self._extract_boxplot_features(extracted_row['x'], extracted_row['y'])
                else:
                    # Data normale
                    extracted_features = self.feature_extractor.extract_features(extracted_row['x'], extracted_row['y'])
                
                similarity_matrix[i, j] = self.feature_extractor.calculate_similarity(gt_features, extracted_features)
        
        return similarity_matrix
    
    def _position_fallback(self, gt_names: List[str], extracted_names: List[str]) -> List[MatchResult]:
        """Appariement de fallback par position."""
        matches = []
        min_len = min(len(gt_names), len(extracted_names))
        
        for i in range(min_len):
            method = MatchMethod.POSITION_MATCH_GENERIC if self._should_skip_name_matching() else MatchMethod.POSITION_MATCH
            matches.append(MatchResult(gt_names[i], extracted_names[i], method, 0.1))
        
        return matches

# =============================================================================
# CLASSES DE COMPARAISON
# =============================================================================

class SeriesComparator(ABC):
    """Classe abstraite pour les comparateurs de séries."""
    
    @abstractmethod
    def compare(self, gt_data: Tuple[List[float], List[float]], 
                extracted_data: Tuple[List[float], List[float]], 
                global_bounds: Dict[str, float] = None,
                **kwargs) -> ComparisonMetrics:
        """Comparer deux séries de données avec bornes globales optionnelles."""
        pass

class LineScatterComparator(SeriesComparator):
    """Comparateur pour les graphiques en ligne et scatter."""
    
    def compare(self, gt_data: Tuple[List[float], List[float]], 
                extracted_data: Tuple[List[float], List[float]], 
                save: bool = False, image_name: str = None, 
                series_name: str = None, save_dir: str = None, 
                plot: bool = False, global_bounds: Dict[str, float] = None,
                x_type: str = 'numeric', **kwargs) -> ComparisonMetrics:
        """Comparer deux séries line/scatter avec stats GT et extraites."""
        
        x_gt, y_gt = gt_data
        x_pred, y_pred = extracted_data
        
        # Calculer les statistiques GT
        gt_stats = calculate_gt_statistics(x_gt, y_gt, x_type)
        
        # NOUVEAU: Calculer les statistiques des données extraites
        extracted_stats = calculate_extracted_statistics(x_pred, y_pred)
        
        # Conversion en numpy arrays et tri
        x_gt, y_gt = np.array(x_gt), np.array(y_gt)
        x_pred, y_pred = np.array(x_pred), np.array(y_pred)
        
        idx_gt = np.argsort(x_gt)
        idx_pred = np.argsort(x_pred)
        x_gt, y_gt = x_gt[idx_gt], y_gt[idx_gt]
        x_pred, y_pred = x_pred[idx_pred], y_pred[idx_pred]
        
        # Calcul des métriques avec bornes globales
        metrics = self._calculate_metrics(x_gt, y_gt, x_pred, y_pred, global_bounds)
        
        # Ajouter les stats GT aux métriques
        metrics.gt_num_points = gt_stats['gt_num_points']
        metrics.gt_x_min = gt_stats['gt_x_min']
        metrics.gt_x_max = gt_stats['gt_x_max']
        metrics.gt_x_std = gt_stats['gt_x_std']
        metrics.gt_y_min = gt_stats['gt_y_min']
        metrics.gt_y_max = gt_stats['gt_y_max']
        metrics.gt_y_std = gt_stats['gt_y_std']
        metrics.gt_x_range = gt_stats['gt_x_range']
        metrics.gt_y_range = gt_stats['gt_y_range']
        
        # NOUVEAU: Ajouter les stats des données extraites
        metrics.extracted_num_points = extracted_stats['extracted_num_points']
            # Ajouter les nouvelles métriques
        dist_metrics = calculate_distribution_metrics(y_gt, y_pred)
        trend_metrics = calculate_trend_metrics(x_gt, y_gt, x_pred, y_pred)
        shape_metrics = calculate_shape_metrics(x_gt, y_gt, x_pred, y_pred)
        coverage_metrics = calculate_coverage_metrics(x_gt, y_gt, x_pred, y_pred)
        quality_metrics = calculate_quality_metrics(y_gt, y_pred)
        
        # Ajouter aux métriques existantes
        for key, value in {**dist_metrics, **trend_metrics, **shape_metrics, 
                        **coverage_metrics, **quality_metrics}.items():
            setattr(metrics, key, value)
        
        # Score composite
        metrics.composite_similarity_score = calculate_composite_similarity_score(metrics)
    
        # Sauvegarde des stats si demandée
        if save:
            self._save_stats(metrics, save_dir, image_name, series_name)
            self._save_plot_comparison(x_gt, y_gt, x_pred, y_pred, metrics, save_dir, image_name, series_name)
        
        # Affichage du plot si demandé
        if plot:
            self._plot_comparison(x_gt, y_gt, x_pred, y_pred, metrics)
        
        return metrics

    def _calculate_metrics(self, x_gt: np.ndarray, y_gt: np.ndarray, 
                          x_pred: np.ndarray, y_pred: np.ndarray,
                          global_bounds: Dict[str, float] = None) -> ComparisonMetrics:
        """Calculer les métriques de comparaison avec bornes globales."""
        
        # Interpolation sur l'intervalle commun
        overlap_x_min = max(x_pred.min(), x_gt.min())
        overlap_x_max = min(x_pred.max(), x_gt.max())
        
        # Calcul des zones manquantes
        left_miss = x_pred.min() - x_gt.min()
        right_miss = x_gt.max() - x_pred.max()
        
        # NOUVEAU: Utiliser les bornes globales si disponibles
        if global_bounds is not None:
            xrange_global = global_bounds['x_range']
            yrange_global = global_bounds['y_range']
        else:
            # Fallback: bornes locales
            xrange_global = x_gt.max() - x_gt.min()
            yrange_global = y_gt.max() - y_gt.min()
        
        # Interpolation et calcul MAE
        if overlap_x_max > overlap_x_min:
            common_x = np.linspace(overlap_x_min, overlap_x_max, num=1000)
            interp_pred = np.interp(common_x, x_pred, y_pred)
            interp_gt = np.interp(common_x, x_gt, y_gt)
            diff = interp_pred - interp_gt
            mae = np.mean(np.abs(diff))
        else:
            mae = np.nan
        
        # Normalisation par les bornes globales
        mae_rel = mae / yrange_global if yrange_global != 0 else np.nan
        left_miss_rel = left_miss / xrange_global if xrange_global != 0 else np.nan
        right_miss_rel = right_miss / xrange_global if xrange_global != 0 else np.nan
        
        return ComparisonMetrics(mae, mae_rel, left_miss_rel, right_miss_rel)
    
    def _save_stats(self, metrics: ComparisonMetrics, save_dir: str, 
                   image_name: str, series_name: str):
        """Sauvegarder les statistiques."""
        if save_dir and image_name and series_name:
            # NOUVEAU: Nettoyer les noms pour éviter les erreurs de fichier
            clean_image_name = sanitize_filename(image_name)
            clean_series_name = sanitize_filename(series_name)
            
            os.makedirs(save_dir, exist_ok=True)
            stats_path = os.path.join(save_dir, f"GT_VS_Extr_{clean_image_name}_{clean_series_name}.txt")
            
            try:
                with open(stats_path, "w", encoding='utf-8') as f:
                    f.write(f"MAE {metrics.mae_rel}\tLeftMissed {metrics.left_miss_rel}\tRightMissed {metrics.right_miss_rel}\n")
            except Exception as e:
                print(f"Erreur sauvegarde stats: {e}")
    
    def _save_plot_comparison(self, x_gt: np.ndarray, y_gt: np.ndarray, 
                             x_pred: np.ndarray, y_pred: np.ndarray, 
                             metrics: ComparisonMetrics, save_dir: str,
                             image_name: str, series_name: str):
        """Sauvegarder le plot de comparaison."""
        if not save_dir or not image_name or not series_name:
            return
        
        # NOUVEAU: Nettoyer les noms pour éviter les erreurs de fichier
        clean_image_name = sanitize_filename(image_name)
        clean_series_name = sanitize_filename(series_name)
        
        # Créer le répertoire s'il n'existe pas
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Créer le plot
            plt.figure(figsize=(10, 6))
            plt.plot(x_gt, y_gt, label="GT", linestyle='-', linewidth=2)
            plt.plot(x_pred, y_pred, label="Extracted", linestyle='--', linewidth=2)
            
            # Ajouter les informations de différence
            overlap_x_min = max(x_pred.min(), x_gt.min())
            overlap_x_max = min(x_pred.max(), x_gt.max())
            
            if overlap_x_max > overlap_x_min:  # Vérifier qu'il y a un overlap
                common_x = np.linspace(overlap_x_min, overlap_x_max, num=1000)
                interp_pred = np.interp(common_x, x_pred, y_pred)
                interp_gt = np.interp(common_x, x_gt, y_gt)
                diff = interp_pred - interp_gt
                bottom_y = min(y_gt.min(), y_pred.min())
                
                plt.fill_between(common_x, diff + bottom_y, bottom_y, color='gray', alpha=0.3, 
                                label=f"Difference (MAE rel={metrics.mae_rel:.3f})")
            
            # Améliorer l'apparence avec titre nettoyé
            plt.title(f"Comparison: {clean_series_name}")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Sauvegarder avec nom nettoyé
            plot_path = os.path.join(save_dir, f"comparison_{clean_image_name}_{clean_series_name}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()  # Fermer la figure pour libérer la mémoire
            
            
        except Exception as e:
            plt.close()  # S'assurer que la figure est fermée même en cas d'erreur
        

    
    def _plot_comparison(self, x_gt: np.ndarray, y_gt: np.ndarray, 
                        x_pred: np.ndarray, y_pred: np.ndarray, 
                        metrics: ComparisonMetrics):
        """Afficher le plot de comparaison (pour visualisation interactive)."""
        plt.figure(figsize=(10, 6))
        plt.plot(x_gt, y_gt, label="GT", linestyle='-', linewidth=2)
        plt.plot(x_pred, y_pred, label="Extracted", linestyle='--', linewidth=2)
        
        # Ajouter les informations de différence
        overlap_x_min = max(x_pred.min(), x_gt.min())
        overlap_x_max = min(x_pred.max(), x_gt.max())
        
        if overlap_x_max > overlap_x_min:
            common_x = np.linspace(overlap_x_min, overlap_x_max, num=1000)
            interp_pred = np.interp(common_x, x_pred, y_pred)
            interp_gt = np.interp(common_x, x_gt, y_gt)
            diff = interp_pred - interp_gt
            bottom_y = min(y_gt.min(), y_pred.min())
            
            plt.fill_between(common_x, diff + bottom_y, bottom_y, color='gray', alpha=0.5, 
                            label=f"relative MAE={metrics.mae_rel:.3f}")
        
        plt.title("Comparison of Ground Truth vs Extracted Data")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class CategoricalLineComparator(SeriesComparator):
    """Comparateur pour les graphiques en ligne avec X catégoriel (comme les dates)."""
    
    def compare(self, gt_data: Tuple[List[str], List[float]], 
                extracted_data: Tuple[List[str], List[float]], 
                save: bool = False, image_name: str = None, 
                series_name: str = None, save_dir: str = None, 
                plot: bool = False, global_bounds: Dict[str, float] = None,
                **kwargs) -> ComparisonMetrics:
        """Comparer deux séries avec X catégoriel et stats GT + extraites."""
        
        x_gt, y_gt = gt_data
        x_pred, y_pred = extracted_data
        
        # Calculer les statistiques GT
        gt_stats = calculate_gt_statistics(x_gt, y_gt, 'categorical')
        
        # NOUVEAU: Calculer les statistiques des données extraites
        extracted_stats = calculate_extracted_statistics(x_pred, y_pred)
        
        # Utiliser la même logique que BarComparator pour le matching
        gt_dict = dict(zip([str(x) for x in x_gt], y_gt))
        pred_dict = dict(zip([str(x) for x in x_pred], y_pred))
        
        # Matching des catégories avec difflib
        matched_pred = {}
        cutoffs = []
        
        for gt_x in gt_dict.keys():
            best_match, best_cutoff = self._find_best_match(gt_x, list(pred_dict.keys()))
            if best_match:
                matched_pred[gt_x] = pred_dict[best_match]
                cutoffs.append(best_cutoff)
        
        if not matched_pred:
            raise ValueError("No matched x values between GT and EX.")
        
        # Calcul des métriques avec bornes globales
        abs_diffs = [abs(gt_dict[gt_x] - matched_pred[gt_x]) for gt_x in matched_pred]
        mae = np.mean(abs_diffs)
        
        # Utiliser les bornes globales pour la normalisation
        if global_bounds is not None:
            yrange_global = global_bounds['y_range']
        else:
            yrange_global = max(y_gt) - min(y_gt) if len(y_gt) > 1 else 1
        
        mae_rel = mae / yrange_global if yrange_global != 0 else np.nan
        x_cutoff = np.mean(cutoffs) if cutoffs else 0.0
        
        metrics = ComparisonMetrics(mae, mae_rel, np.nan, np.nan, x_cutoff)
        
        # Ajouter les stats GT
        metrics.gt_num_points = gt_stats['gt_num_points']
        metrics.gt_x_min = gt_stats['gt_x_min']
        metrics.gt_x_max = gt_stats['gt_x_max']
        metrics.gt_x_std = gt_stats['gt_x_std']
        metrics.gt_y_min = gt_stats['gt_y_min']
        metrics.gt_y_max = gt_stats['gt_y_max']
        metrics.gt_y_std = gt_stats['gt_y_std']
        metrics.gt_x_range = gt_stats['gt_x_range']
        metrics.gt_y_range = gt_stats['gt_y_range']
        
        # NOUVEAU: Ajouter les stats des données extraites
        metrics.extracted_num_points = extracted_stats['extracted_num_points']
        
        # Sauvegarde si demandée
        if save:
            self._save_stats(metrics, save_dir, image_name, series_name)
            self._save_categorical_plot_comparison(gt_dict, matched_pred, metrics, save_dir, image_name, series_name)
        
        # Affichage si demandé
        if plot:
            self._plot_categorical_comparison(gt_dict, matched_pred, metrics)
        
        return metrics
    
    def _find_best_match(self, gt_x: str, candidates: List[str]) -> Tuple[Optional[str], float]:
        """Trouver le meilleur match pour une catégorie."""
        import difflib
        for cutoff in [i / 100 for i in range(100, 0, -1)]:
            matches = difflib.get_close_matches(gt_x, candidates, n=1, cutoff=cutoff)
            if matches:
                return matches[0], cutoff
        return None, 0.0
    
    def _save_stats(self, metrics: ComparisonMetrics, save_dir: str, 
                   image_name: str, series_name: str):
        """Sauvegarder les statistiques."""
        if save_dir and image_name and series_name:
            # NOUVEAU: Nettoyer les noms
            clean_image_name = sanitize_filename(image_name)
            clean_series_name = sanitize_filename(series_name)
            
            os.makedirs(save_dir, exist_ok=True)
            stats_path = os.path.join(save_dir, f"GT_VS_Extr_{clean_image_name}_{clean_series_name}.txt")
            
            try:
                with open(stats_path, "w", encoding='utf-8') as f:
                    f.write(f"MAE {metrics.mae_rel}\tx_cutoff {metrics.x_cutoff}\n")
            except Exception as e:
                print(f"Erreur sauvegarde categorical stats: {e}")    
    def _save_categorical_plot_comparison(self, gt_dict: dict, matched_pred: dict, 
                                        metrics: ComparisonMetrics, save_dir: str,
                                        image_name: str, series_name: str):
        """Sauvegarder le plot de comparaison pour les lignes catégorielles."""
        if not save_dir or not image_name or not series_name:
            return
        
        # NOUVEAU: Nettoyer les noms
        clean_image_name = sanitize_filename(image_name)
        clean_series_name = sanitize_filename(series_name)
        
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Préparer les données
            categories = list(matched_pred.keys())
            gt_values = [gt_dict[cat] for cat in categories]
            pred_values = [matched_pred[cat] for cat in categories]
            
            x_pos = np.arange(len(categories))
            
            fig, ax = plt.subplots(figsize=(max(12, len(categories) * 0.8), 8))
            
            # Créer des lignes connectées
            ax.plot(x_pos, gt_values, 'o-', label='Ground Truth', linewidth=2, markersize=6, color='blue')
            ax.plot(x_pos, pred_values, 's--', label='Extracted', linewidth=2, markersize=6, color='orange')
            
            # Ajouter les valeurs sur les points
            for i, (gt_val, pred_val) in enumerate(zip(gt_values, pred_values)):
                ax.annotate(f'{gt_val:.1f}', (i, gt_val), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8, color='blue')
                ax.annotate(f'{pred_val:.1f}', (i, pred_val), textcoords="offset points", 
                           xytext=(0,-15), ha='center', fontsize=8, color='orange')
            
            # Configuration avec titre nettoyé
            ax.set_xlabel('Categories', fontsize=12, fontweight='bold')
            ax.set_ylabel('Values', fontsize=12, fontweight='bold')
            ax.set_title(f'Categorical Line Comparison: {clean_series_name}\nMAE rel = {metrics.mae_rel:.4f}', 
                        fontsize=14, fontweight='bold')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Sauvegarder avec nom nettoyé
            plot_path = os.path.join(save_dir, f"categorical_line_comparison_{clean_image_name}_{clean_series_name}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            
        except Exception as e:
            print(f"Erreur sauvegarde categorical plot: {e}")
            plt.close()
        
    
    def _plot_categorical_comparison(self, gt_dict: dict, matched_pred: dict, metrics: ComparisonMetrics):
        """Afficher le plot de comparaison pour les lignes catégorielles."""
        categories = list(matched_pred.keys())
        gt_values = [gt_dict[cat] for cat in categories]
        pred_values = [matched_pred[cat] for cat in categories]
        
        x_pos = np.arange(len(categories))
        
        fig, ax = plt.subplots(figsize=(max(12, len(categories) * 0.8), 8))
        
        ax.plot(x_pos, gt_values, 'o-', label='Ground Truth', linewidth=2, markersize=6, color='blue')
        ax.plot(x_pos, pred_values, 's--', label='Extracted', linewidth=2, markersize=6, color='orange')
        
        ax.set_xlabel('Categories', fontsize=12, fontweight='bold')
        ax.set_ylabel('Values', fontsize=12, fontweight='bold')
        ax.set_title(f'Categorical Line Comparison (MAE rel = {metrics.mae_rel:.4f})', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class BarComparator(SeriesComparator):
    """Comparateur pour les graphiques en barres."""
    
    def compare(self, gt_data: Tuple[List[float], List[float]], 
                extracted_data: Tuple[List[float], List[float]], 
                save: bool = False, image_name: str = None, 
                series_name: str = None, save_dir: str = None, 
                plot: bool = False, global_bounds: Dict[str, float] = None,
                **kwargs) -> ComparisonMetrics:
        """Comparer deux séries de barres avec stats GT + extraites."""
        x_gt, y_gt = gt_data
        x_pred, y_pred = extracted_data
        
        # Calculer les statistiques GT
        gt_stats = calculate_gt_statistics(x_gt, y_gt, 'categorical')  # X généralement catégoriel pour les barres
        
        # NOUVEAU: Calculer les statistiques des données extraites
        extracted_stats = calculate_extracted_statistics(x_pred, y_pred)
        
        # Conversion en string pour le matching
        x_gt_str = [str(x) for x in x_gt]
        x_pred_str = [str(x) for x in x_pred]
        
        # Dictionnaires pour lookup rapide
        gt_dict = dict(zip(x_gt_str, y_gt))
        pred_dict = dict(zip(x_pred_str, y_pred))
        
        # Matching des valeurs x avec difflib
        matched_pred = {}
        cutoffs = []
        
        for gt_x in x_gt_str:
            best_match, best_cutoff = self._find_best_match(gt_x, x_pred_str)
            if best_match:
                matched_pred[gt_x] = pred_dict[best_match]
                cutoffs.append(best_cutoff)
        
        if not matched_pred:
            raise ValueError("No matched x values between GT and EX.")
        
        # Calcul des métriques avec bornes globales
        abs_diffs = [abs(gt_dict[gt_x] - matched_pred[gt_x]) for gt_x in matched_pred]
        mae = np.mean(abs_diffs)
        
        # Utiliser les bornes globales pour la normalisation
        if global_bounds is not None:
            yrange_global = global_bounds['y_range']
        else:
            yrange_global = max(y_gt) - min(y_gt) if len(y_gt) > 1 else 1
        
        mae_rel = mae / yrange_global if yrange_global != 0 else np.nan
        x_cutoff = np.mean(cutoffs) if cutoffs else 0.0
        
        metrics = ComparisonMetrics(mae, mae_rel, np.nan, np.nan, x_cutoff)
        
        # Ajouter les stats GT
        metrics.gt_num_points = gt_stats['gt_num_points']
        metrics.gt_x_min = gt_stats['gt_x_min']
        metrics.gt_x_max = gt_stats['gt_x_max']
        metrics.gt_x_std = gt_stats['gt_x_std']
        metrics.gt_y_min = gt_stats['gt_y_min']
        metrics.gt_y_max = gt_stats['gt_y_max']
        metrics.gt_y_std = gt_stats['gt_y_std']
        metrics.gt_x_range = gt_stats['gt_x_range']
        metrics.gt_y_range = gt_stats['gt_y_range']
        
        # NOUVEAU: Ajouter les stats des données extraites
        metrics.extracted_num_points = extracted_stats['extracted_num_points']
        
        # Sauvegarde si demandée
        if save:
            self._save_stats(metrics, save_dir, image_name, series_name)
            self._save_bar_plot_comparison(gt_dict, matched_pred, metrics, save_dir, image_name, series_name)
        
        # Affichage si demandé
        if plot:
            self._plot_bar_comparison(gt_dict, matched_pred, metrics)
        
        return metrics
    
    def _save_stats(self, metrics: ComparisonMetrics, save_dir: str, 
                   image_name: str, series_name: str):
        """Sauvegarder les statistiques pour les barres."""
        if save_dir and image_name and series_name:
            # NOUVEAU: Nettoyer les noms
            clean_image_name = sanitize_filename(image_name)
            clean_series_name = sanitize_filename(series_name)
            
            os.makedirs(save_dir, exist_ok=True)
            stats_path = os.path.join(save_dir, f"GT_VS_Extr_{clean_image_name}_{clean_series_name}.txt")
            
            try:
                with open(stats_path, "w", encoding='utf-8') as f:
                    f.write(f"MAE {metrics.mae_rel}\tx_cutoff {metrics.x_cutoff}\n")
            except Exception as e:
                print(f"Erreur sauvegarde bar stats: {e}")
    
    def _save_bar_plot_comparison(self, gt_dict: dict, matched_pred: dict, 
                                 metrics: ComparisonMetrics, save_dir: str,
                                 image_name: str, series_name: str):
        """Sauvegarder le plot de comparaison pour les barres avec barres côte à côte."""
        if not save_dir or not image_name or not series_name:
            return
        
        # NOUVEAU: Nettoyer les noms
        clean_image_name = sanitize_filename(image_name)
        clean_series_name = sanitize_filename(series_name)
        
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Préparer les données pour le plot
            categories = list(matched_pred.keys())
            gt_values = [gt_dict[cat] for cat in categories]
            pred_values = [matched_pred[cat] for cat in categories]
            
            # Configuration pour barres côte à côte
            x_pos = np.arange(len(categories))
            width = 0.35  # Largeur des barres
            
            fig, ax = plt.subplots(figsize=(max(12, len(categories) * 1.5), 8))
            
            # Créer les barres côte à côte pour chaque catégorie
            bars_gt = ax.bar(x_pos - width/2, gt_values, width, 
                            label='Ground Truth', alpha=0.8, color='blue')
            bars_pred = ax.bar(x_pos + width/2, pred_values, width, 
                              label='Extracted', alpha=0.8, color='orange')
            
            # Ajouter les valeurs numériques sur chaque barre
            for i, (gt_val, pred_val) in enumerate(zip(gt_values, pred_values)):
                # Valeur sur la barre GT
                ax.text(i - width/2, gt_val + max(max(gt_values), max(pred_values)) * 0.01, 
                       f'{gt_val:.2f}', ha='center', va='bottom', fontsize=9, 
                       fontweight='bold', color='blue')
                
                # Valeur sur la barre Extracted
                ax.text(i + width/2, pred_val + max(max(gt_values), max(pred_values)) * 0.01, 
                       f'{pred_val:.2f}', ha='center', va='bottom', fontsize=9, 
                       fontweight='bold', color='orange')
            
            # Configuration des axes et labels avec titre nettoyé
            ax.set_xlabel('Categories', fontsize=12, fontweight='bold')
            ax.set_ylabel('Values', fontsize=12, fontweight='bold')
            ax.set_title(f'Bar Comparison: {clean_series_name}\nMAE rel = {metrics.mae_rel:.4f}', 
                        fontsize=14, fontweight='bold')
            
            # Configurer les ticks de l'axe X
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
            
            # Légende et grille
            ax.legend(fontsize=11, loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Ajuster les marges pour éviter la coupure des labels
            plt.tight_layout()
            
            # Sauvegarder avec un nom plus descriptif et nettoyé
            plot_path = os.path.join(save_dir, f"bar_comparison_{clean_image_name}_{clean_series_name}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()  # Libérer la mémoire
            
            
        except Exception as e:
            print(f"Erreur sauvegarde bar plot: {e}")
            plt.close()
        

    
    def _plot_bar_comparison(self, gt_dict: dict, matched_pred: dict, metrics: ComparisonMetrics):
        """Afficher le plot de comparaison pour les barres (version interactive)."""
        categories = list(matched_pred.keys())
        gt_values = [gt_dict[cat] for cat in categories]
        pred_values = [matched_pred[cat] for cat in categories]
        
        x_pos = np.arange(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(max(12, len(categories) * 1.5), 8))
        
        bars_gt = ax.bar(x_pos - width/2, gt_values, width, 
                        label='Ground Truth', alpha=0.8, color='blue')
        bars_pred = ax.bar(x_pos + width/2, pred_values, width, 
                          label='Extracted', alpha=0.8, color='orange')
        
        # Ajouter les valeurs sur les barres
        for i, (gt_val, pred_val) in enumerate(zip(gt_values, pred_values)):
            ax.text(i - width/2, gt_val + max(max(gt_values), max(pred_values)) * 0.01, 
                   f'{gt_val:.2f}', ha='center', va='bottom', fontsize=9, 
                   fontweight='bold', color='blue')
            ax.text(i + width/2, pred_val + max(max(gt_values), max(pred_values)) * 0.01, 
                   f'{pred_val:.2f}', ha='center', va='bottom', fontsize=9, 
                   fontweight='bold', color='orange')
        
        ax.set_xlabel('Categories', fontsize=12, fontweight='bold')
        ax.set_ylabel('Values', fontsize=12, fontweight='bold')
        ax.set_title(f'Bar Comparison (MAE rel = {metrics.mae_rel:.4f})', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def _find_best_match(self, gt_x: str, candidates: List[str]) -> Tuple[Optional[str], float]:
        """Trouver le meilleur match pour une valeur x."""
        for cutoff in [i / 100 for i in range(100, 0, -1)]:
            matches = difflib.get_close_matches(gt_x, candidates, n=1, cutoff=cutoff)
            if matches:
                return matches[0], cutoff
        return None, 0.0

class BoxPlotComparator(SeriesComparator):
    """Comparateur spécialisé pour les box plots."""
    
    def compare(self, gt_data: Tuple[List[str], List[dict]], 
                extracted_data: Tuple[List[str], List[dict]], 
                save: bool = False, image_name: str = None, 
                series_name: str = None, save_dir: str = None, 
                plot: bool = False, global_bounds: Dict[str, float] = None,
                **kwargs) -> ComparisonMetrics:
        """Comparer deux séries de box plots."""
        
        x_gt, boxdata_gt = gt_data
        x_pred, boxdata_pred = extracted_data
        
        
        # Calculer les statistiques GT et extraites
        gt_stats = self._calculate_boxplot_gt_statistics(x_gt, boxdata_gt)
        extracted_stats = self._calculate_boxplot_extracted_statistics(x_pred, boxdata_pred)
        
        # Matching des catégories (noms des boxes)
        gt_dict = dict(zip([str(x) for x in x_gt], boxdata_gt))
        pred_dict = dict(zip([str(x) for x in x_pred], boxdata_pred))
        
        # Trouver les correspondances avec fuzzy matching
        matched_pred = {}
        cutoffs = []
        
        for gt_x in gt_dict.keys():
            best_match, best_cutoff = self._find_best_match(gt_x, list(pred_dict.keys()))
            if best_match:
                matched_pred[gt_x] = pred_dict[best_match]
                cutoffs.append(best_cutoff)
        
        if not matched_pred:
            raise ValueError("No matched box categories between GT and Extracted.")
        
        # Calculer les erreurs sur les métriques de box plot
        median_errors = []
        q1_errors = []
        q3_errors = []
        min_errors = []
        max_errors = []
        
        for gt_x in matched_pred.keys():
            gt_box = gt_dict[gt_x]
            pred_box = matched_pred[gt_x]
            
            # Erreurs sur chaque métrique
            if 'median' in gt_box and 'median' in pred_box:
                median_errors.append(abs(gt_box['median'] - pred_box['median']))
            
            if 'first_quartile' in gt_box and 'first_quartile' in pred_box:
                q1_errors.append(abs(gt_box['first_quartile'] - pred_box['first_quartile']))
            
            if 'third_quartile' in gt_box and 'third_quartile' in pred_box:
                q3_errors.append(abs(gt_box['third_quartile'] - pred_box['third_quartile']))
            
            if 'min' in gt_box and 'min' in pred_box:
                min_errors.append(abs(gt_box['min'] - pred_box['min']))
            
            if 'max' in gt_box and 'max' in pred_box:
                max_errors.append(abs(gt_box['max'] - pred_box['max']))
        
        # MAE global (moyenne de toutes les métriques)
        all_errors = median_errors + q1_errors + q3_errors + min_errors + max_errors
        mae = np.mean(all_errors) if all_errors else np.nan
        
        # Normalisation avec bornes globales
        if global_bounds is not None:
            yrange_global = global_bounds['y_range']
        else:
            # Calculer le range local de toutes les valeurs
            all_gt_values = []
            all_pred_values = []
            for box in gt_dict.values():
                all_gt_values.extend([box.get('min', 0), box.get('max', 0), 
                                    box.get('median', 0), box.get('first_quartile', 0), 
                                    box.get('third_quartile', 0)])
            for box in pred_dict.values():
                all_pred_values.extend([box.get('min', 0), box.get('max', 0), 
                                      box.get('median', 0), box.get('first_quartile', 0), 
                                      box.get('third_quartile', 0)])
            
            all_values = [v for v in all_gt_values + all_pred_values if v is not None]
            yrange_global = max(all_values) - min(all_values) if all_values else 1
        mae_rel = mae / yrange_global if yrange_global != 0 else np.nan
        x_cutoff = np.mean(cutoffs) if cutoffs else 0.0
        
        # Créer les métriques
        metrics = ComparisonMetrics(mae, mae_rel, np.nan, np.nan, x_cutoff)
        
        # Ajouter les statistiques GT
        metrics.gt_num_points = gt_stats['gt_num_points']
        metrics.gt_x_min = gt_stats['gt_x_min']
        metrics.gt_x_max = gt_stats['gt_x_max']
        metrics.gt_x_std = gt_stats['gt_x_std']
        metrics.gt_x_range = gt_stats['gt_x_range']
        metrics.gt_y_min = gt_stats['gt_y_min']
        metrics.gt_y_max = gt_stats['gt_y_max']
        metrics.gt_y_std = gt_stats['gt_y_std']
        metrics.gt_y_range = gt_stats['gt_y_range']
        
        # Ajouter les statistiques extraites
        metrics.extracted_num_points = extracted_stats['extracted_num_points']
        
        # Sauvegarde si demandée
        if save:
            self._save_boxplot_stats(metrics, save_dir, image_name, series_name)
            self._save_boxplot_comparison(gt_dict, matched_pred, metrics, save_dir, image_name, series_name)
        
        # Affichage si demandé
        if plot:
            self._plot_boxplot_comparison(gt_dict, matched_pred, metrics)
        
        return metrics
    
    def _calculate_boxplot_gt_statistics(self, x_data, boxdata) -> dict:
        """Calculer les statistiques GT pour box plot."""
        all_values = []
        for box in boxdata:
            all_values.extend([box.get('min', 0), box.get('max', 0), 
                             box.get('median', 0), box.get('first_quartile', 0), 
                             box.get('third_quartile', 0)])
        
        all_values = [v for v in all_values if v is not None]
        y_array = np.array(all_values)
        
        return {
            'gt_num_points': len(x_data),  # Nombre de boxes
            'gt_x_min': np.nan,  # X catégoriel
            'gt_x_max': np.nan,
            'gt_x_std': np.nan,
            'gt_x_range': float(len(set(x_data))),  # Nombre de catégories uniques
            'gt_y_min': float(y_array.min()) if len(y_array) > 0 else np.nan,
            'gt_y_max': float(y_array.max()) if len(y_array) > 0 else np.nan,
            'gt_y_std': float(y_array.std()) if len(y_array) > 1 else 0.0,
            'gt_y_range': float(y_array.max() - y_array.min()) if len(y_array) > 0 else 0.0
        }
    
    def _calculate_boxplot_extracted_statistics(self, x_data, boxdata) -> dict:
        """Calculer les statistiques des données extraites pour box plot."""
        return {
            'extracted_num_points': len(x_data)  # Nombre de boxes extraites
        }
    
    def _save_boxplot_stats(self, metrics: ComparisonMetrics, save_dir: str, image_name: str, series_name: str):
        """Sauvegarder les statistiques de box plot."""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            safe_series_name = sanitize_filename(series_name)
            stats_file = os.path.join(save_dir, f"boxplot_stats_{safe_series_name}.txt")
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(f"Box Plot Statistics - {series_name}\n")
                f.write(f"{'='*50}\n")
                f.write(f"MAE: {metrics.mae:.6f}\n")
                f.write(f"MAE rel: {metrics.mae_rel:.6f}\n")
                f.write(f"X cutoff: {metrics.x_cutoff:.6f}\n")
                f.write(f"GT Boxes: {metrics.gt_num_points}\n")
                f.write(f"Extracted Boxes: {metrics.extracted_num_points}\n")
                f.write(f"GT Y Range: {metrics.gt_y_range:.6f}\n")
    def _find_best_match(self, gt_x: str, candidates: List[str]) -> Tuple[Optional[str], float]:
        """Trouver le meilleur match pour une catégorie de box plot."""
        import difflib
        for cutoff in [i / 100 for i in range(100, 30, -5)]:  # De 100% à 30%
            matches = difflib.get_close_matches(gt_x, candidates, n=1, cutoff=cutoff)
            if matches:
                return matches[0], cutoff
        return None, 0.0
    
    def _save_boxplot_comparison(self, gt_dict: dict, matched_pred: dict, 
                                metrics: ComparisonMetrics, save_dir: str,
                                image_name: str, series_name: str):
        """Sauvegarder le plot de comparaison pour box plots."""
        if not save_dir or not image_name or not series_name:
            return
        
        clean_image_name = sanitize_filename(image_name)
        clean_series_name = sanitize_filename(series_name)
        
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            categories = list(matched_pred.keys())
            
            fig, ax = plt.subplots(figsize=(max(10, len(categories) * 2), 8))
            
            # Créer les box plots côte à côte
            positions = np.arange(len(categories))
            width = 0.35
            
            # Données GT
            gt_data = []
            pred_data = []
            
            for cat in categories:
                gt_box = gt_dict[cat]
                pred_box = matched_pred[cat]
                
                # Créer des listes pour matplotlib boxplot
                gt_values = [gt_box.get('min', 0), gt_box.get('first_quartile', 0),
                            gt_box.get('median', 0), gt_box.get('third_quartile', 0),
                            gt_box.get('max', 0)]
                pred_values = [pred_box.get('min', 0), pred_box.get('first_quartile', 0),
                              pred_box.get('median', 0), pred_box.get('third_quartile', 0),
                              pred_box.get('max', 0)]
                
                gt_data.append([v for v in gt_values if v is not None])
                pred_data.append([v for v in pred_values if v is not None])
            
            # Créer les box plots
            bp1 = ax.boxplot(gt_data, positions=positions-width/2, widths=width*0.8,
                            patch_artist=True, boxprops=dict(facecolor='lightblue'),
                            medianprops=dict(color='blue', linewidth=2))
            
            bp2 = ax.boxplot(pred_data, positions=positions+width/2, widths=width*0.8,
                            patch_artist=True, boxprops=dict(facecolor='orange'),
                            medianprops=dict(color='red', linewidth=2))
            
            # Configuration
            ax.set_xlabel('Categories', fontsize=12, fontweight='bold')
            ax.set_ylabel('Values', fontsize=12, fontweight='bold') 
            ax.set_title(f'Box Plot Comparison: {clean_series_name}\nMAE rel = {metrics.mae_rel:.4f}',
                        fontsize=14, fontweight='bold')
            
            ax.set_xticks(positions)
            ax.set_xticklabels(categories, rotation=45, ha='right')
            
            # Légende
            ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Ground Truth', 'Extracted'])
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = os.path.join(save_dir, f"boxplot_comparison_{clean_image_name}_{clean_series_name}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            
        except Exception as e:
            print(f"Erreur sauvegarde box plot: {e}")
            plt.close()
    
    def _plot_boxplot_comparison(self, gt_dict: dict, matched_pred: dict, metrics: ComparisonMetrics):
        """Afficher le plot de comparaison pour box plots."""
        categories = list(matched_pred.keys())
        
        fig, ax = plt.subplots(figsize=(max(10, len(categories) * 2), 8))
        
        positions = np.arange(len(categories))
        width = 0.35
        
        gt_data = []
        pred_data = []
        
        for cat in categories:
            gt_box = gt_dict[cat]
            pred_box = matched_pred[cat]
            
            gt_values = [gt_box.get('min', 0), gt_box.get('first_quartile', 0),
                        gt_box.get('median', 0), gt_box.get('third_quartile', 0),
                        gt_box.get('max', 0)]
            pred_values = [pred_box.get('min', 0), pred_box.get('first_quartile', 0),
                          pred_box.get('median', 0), pred_box.get('third_quartile', 0),
                          pred_box.get('max', 0)]
            
            gt_data.append([v for v in gt_values if v is not None])
            pred_data.append([v for v in pred_values if v is not None])
        
        bp1 = ax.boxplot(gt_data, positions=positions-width/2, widths=width*0.8,
                        patch_artist=True, boxprops=dict(facecolor='lightblue'))
        bp2 = ax.boxplot(pred_data, positions=positions+width/2, widths=width*0.8,
                        patch_artist=True, boxprops=dict(facecolor='orange'))
        
        ax.set_xlabel('Categories')
        ax.set_ylabel('Values')
        ax.set_title(f'Box Plot Comparison (MAE rel = {metrics.mae_rel:.4f})')
        ax.set_xticks(positions)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Ground Truth', 'Extracted'])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================


class MAEValidator:
    def __init__(self):
        self.comparators = {
            PlotType.LINE: LineScatterComparator(),
            PlotType.SCATTER: LineScatterComparator(),
            PlotType.BAR: BarComparator(),
            PlotType.PLOT: LineScatterComparator(),  # Ajout du type PLOT
            PlotType.HORIZONTAL_BAR: BarComparator(),
            PlotType.VERTICAL_BOX: BoxPlotComparator(),
            PlotType.HORIZONTAL_BOX: BoxPlotComparator()  
        }
        self.categorical_comparator = CategoricalLineComparator()
        self.series_matcher = None  # Sera initialisé dynamiquement
    
    def compare_single_image(self, gt_json_path: str, extracted_json_path: str, 
                        image_name: str, plot_type: str, 
                        save: bool = False, save_dir: str = None, 
                        plot: bool = False) -> pd.DataFrame:
        """Comparer une image avec détection automatique du type et du bon loader."""
        
        logger = ValidationLogger(save_dir, image_name) if save else None
        
        try:
            if logger:
                logger.log_step("data_loading", "Chargement des données GT et extraites")
            
            # NOUVEAU: Déterminer le type de plot avec normalisation
            plot_type_enum, gt_plot_type = self._determine_plot_type(plot_type)
            
                        
            if logger:
                logger.log_plot_type_detection(plot_type, plot_type_enum.value, gt_plot_type)
            
            # MODIFICATION PRINCIPALE: Chargement conditionnel selon le type détecté
            if plot_type_enum in [PlotType.VERTICAL_BOX, PlotType.HORIZONTAL_BOX]:
                gt_df = BoxPlotDataLoader.load_boxplot_series(gt_json_path)
                extracted_df = BoxPlotDataLoader.load_boxplot_series(extracted_json_path)
                use_categorical_comparator = False
                final_plot_type_enum = PlotType.VERTICAL_BOX  # Normaliser vers box plot
            else:
                gt_df = DataLoader.load_series(gt_json_path)
                extracted_df = DataLoader.load_series(extracted_json_path)
                
                # Déterminer si on doit utiliser le comparateur catégoriel
                categorical_x = gt_df['x_type'].eq('categorical').any() if not gt_df.empty else False
                use_categorical_comparator = categorical_x and plot_type_enum in [PlotType.LINE, PlotType.SCATTER, PlotType.PLOT]
                final_plot_type_enum = plot_type_enum
            
            if logger:
                logger.log_data_loading(gt_json_path, gt_df, "GT")
                logger.log_data_loading(extracted_json_path, extracted_df, "Extracted")
            
            if gt_df.empty or extracted_df.empty:
                if logger:
                    logger.log_error("data_loading", "DataFrames vides", "GT ou Extracted vide")
                return self._empty_result_dataframe()
            
            # Calcul des bornes globales selon le type
            if final_plot_type_enum in [PlotType.VERTICAL_BOX, PlotType.HORIZONTAL_BOX]:
                global_bounds = self._calculate_boxplot_global_bounds(gt_df, extracted_df)
            else:
                categorical_x = gt_df['x_type'].eq('categorical').any() if not gt_df.empty else False
                global_bounds = self._calculate_global_bounds(gt_df, extracted_df, categorical_x)
            
            if logger:
                logger.log_bounds_calculation(global_bounds)
            
            # Initialiser le series matcher avec les données chargées
            self.series_matcher = SeriesMatcher(gt_df, extracted_df)
            
            # Matching des séries
            if logger:
                logger.log_step("series_matching", "Appariement des séries GT et extraites")
            
            matched_pairs = self.series_matcher.match_series()
            
            if logger:
                logger.log_matching_results(matched_pairs)
            
            if not matched_pairs:
                if logger:
                    logger.log_error("series_matching", "Aucun appariement trouvé", "Pas de correspondance entre GT et Extracted")
                return self._empty_result_dataframe()
            
            # Comparaison avec le bon comparateur
            results = []
            successful_comparisons = 0
            comparison_errors = 0
            
            for match in matched_pairs:
                try:
                    # Choisir le comparateur approprié
                    if final_plot_type_enum in [PlotType.VERTICAL_BOX, PlotType.HORIZONTAL_BOX]:
                        comparator = self.comparators[final_plot_type_enum]
                        comparator_name = "BoxPlotComparator"
                    elif use_categorical_comparator:
                        comparator = self.categorical_comparator
                        comparator_name = "CategoricalLineComparator"
                    else:
                        comparator = self.comparators[final_plot_type_enum]
                        comparator_name = f"{type(comparator).__name__}"
                    
                    metrics = self._compare_matched_pair(
                        match, gt_df, extracted_df, comparator, 
                        save, save_dir, image_name, plot, global_bounds
                    )
                    
                    if logger:
                        logger.log_comparison(match, metrics, comparator_name)
                    
                    result_row = self._create_result_row(gt_plot_type, image_name, match, metrics)
                    results.append(result_row)
                    successful_comparisons += 1
                    
                except Exception as e:
                    if logger:
                        logger.log_error("series_comparison", e, f"Comparaison {match.gt_name} vs {match.extracted_name}")
                    
                    error_row = self._create_error_row(gt_plot_type, image_name, match)
                    results.append(error_row)
                    comparison_errors += 1
            
            # Finaliser le log
            if logger:
                logger.finalize_log(len(matched_pairs), successful_comparisons, comparison_errors)
            
            return pd.DataFrame(results)
            
        except Exception as e:
            if logger:
                logger.log_error("validation_process", e, f"Erreur globale lors du traitement de {image_name}")
                logger.finalize_log(0, 0, 1)
            
            print(f"Erreur lors du traitement de l'image {image_name}: {e}")
            return self._empty_result_dataframe()
    
    def _determine_plot_type(self, plot_type: str) -> Tuple[PlotType, str]:
        """Déterminer le type de plot enum et la version string pour les résultats."""
        plot_type_lower = plot_type.lower().strip()
        
        # Mapping avec variations possibles
        type_mapping = {
            'line': PlotType.LINE,
            'line plot': PlotType.LINE,
            'plot': PlotType.PLOT,
            'scatter': PlotType.SCATTER,
            'scatter plot': PlotType.SCATTER,
            'bar': PlotType.BAR,
            'bar chart': PlotType.BAR,
            'horizontal_bar': PlotType.HORIZONTAL_BAR,
            'horizontal bar': PlotType.HORIZONTAL_BAR,
            'vertical_box': PlotType.VERTICAL_BOX,
            'vertical box': PlotType.VERTICAL_BOX,
            'box': PlotType.VERTICAL_BOX,
            'boxplot': PlotType.VERTICAL_BOX,
            'box plot': PlotType.VERTICAL_BOX,
            'horizontal_box': PlotType.HORIZONTAL_BOX,
            'horizontal box': PlotType.HORIZONTAL_BOX
        }
        
        # Recherche directe
        if plot_type_lower in type_mapping:
            enum_type = type_mapping[plot_type_lower]
            return enum_type, plot_type
        
        # Recherche par contenu
        if 'box' in plot_type_lower:
            return PlotType.VERTICAL_BOX, plot_type
        elif 'bar' in plot_type_lower:
            return PlotType.BAR, plot_type
        elif 'scatter' in plot_type_lower:
            return PlotType.SCATTER, plot_type
        elif any(word in plot_type_lower for word in ['line', 'plot']):
            return PlotType.LINE, plot_type
        
        # Fallback
        print(f"Warning: Type de plot non reconnu '{plot_type}', utilisation de 'line'")
        return PlotType.LINE, plot_type
    
    
    def log_bounds_calculation(self, bounds: Dict[str, float]):
        """Logger les bornes calculées (méthode manquante)."""
        self.log_step("bounds_calculation", "Calcul des bornes globales", {
            "Y Range": f"{bounds['y_range']:.4f}",
            "Y Min-Max": f"[{bounds['y_min']:.4f}, {bounds['y_max']:.4f}]",
            "X Range": f"{bounds.get('x_range', 'N/A')}" 
        })
    
    def log_matching_results(self, matched_pairs: List[MatchResult]):
        """Logger les résultats du matching (méthode manquante)."""
        details = f"{len(matched_pairs)} appariements trouvés"
        data = {
            "Matches": [f"{m.gt_name} <-> {m.extracted_name} ({m.method.value}, score={m.score:.3f})" 
                       for m in matched_pairs]
        }
        self.log_step("matching_results", details, data)
    

    def _calculate_boxplot_global_bounds(self, gt_df: pd.DataFrame, extracted_df: pd.DataFrame) -> Dict[str, float]:
        """Calculer les bornes globales pour box plots."""
        all_values = []
        
        # Collecter toutes les valeurs des box plots
        for _, row in gt_df.iterrows():
            for box in row['y']:  # row['y'] contient les données de box
                all_values.extend([v for v in [box.get('min'), box.get('max'), 
                                             box.get('median'), box.get('first_quartile'), 
                                             box.get('third_quartile')] if v is not None])
        
        for _, row in extracted_df.iterrows():
            for box in row['y']:
                all_values.extend([v for v in [box.get('min'), box.get('max'), 
                                             box.get('median'), box.get('first_quartile'), 
                                             box.get('third_quartile')] if v is not None])
        
        if all_values:
            return {
                'y_min': min(all_values),
                'y_max': max(all_values),
                'y_range': max(all_values) - min(all_values),
                'x_min': np.nan,  # X catégoriel pour box plots
                'x_max': np.nan,
                'x_range': np.nan
            }
        else:
            return {'y_min': 0, 'y_max': 1, 'y_range': 1, 'x_min': np.nan, 'x_max': np.nan, 'x_range': np.nan}
    
    def _calculate_global_bounds(self, gt_df: pd.DataFrame, extracted_df: pd.DataFrame, 
                               categorical_x: bool = False) -> Dict[str, float]:
        """Calculer les bornes globales."""
        all_x = []
        all_y = []
        
        # Collecter toutes les valeurs Y
        for _, row in gt_df.iterrows():
            all_y.extend(row['y'])
            # Pour X, seulement si numérique
            if not categorical_x and row['x_type'] == 'numeric':
                all_x.extend(row['x'])
        
        for _, row in extracted_df.iterrows():
            all_y.extend(row['y'])
            if not categorical_x and row['x_type'] == 'numeric':
                all_x.extend(row['x'])
        
        # Convertir en numpy
        all_y = np.array(all_y)
        
        bounds = {
            'y_min': float(all_y.min()),
            'y_max': float(all_y.max()),
            'y_range': float(all_y.max() - all_y.min())
        }
        
        # Ajouter les bornes X seulement si numérique
        if not categorical_x and len(all_x) > 0:
            all_x = np.array(all_x)
            bounds.update({
                'x_min': float(all_x.min()),
                'x_max': float(all_x.max()),
                'x_range': float(all_x.max() - all_x.min())
            })
        
        return bounds
    
    def _compare_matched_pair(self, match: MatchResult, gt_df: pd.DataFrame, 
                            extracted_df: pd.DataFrame, comparator: SeriesComparator,
                            save: bool, save_dir: str, image_name: str, 
                            plot: bool, global_bounds: Dict[str, float]) -> ComparisonMetrics:
        """Comparer une paire de séries appariées avec bornes globales."""
        gt_row = gt_df[gt_df['name'] == match.gt_name].iloc[0]
        extracted_row = extracted_df[extracted_df['name'] == match.extracted_name].iloc[0]
        
        gt_data = (gt_row['x'], gt_row['y'])
        extracted_data = (extracted_row['x'], extracted_row['y'])
        
        # NOUVEAU: Passer le type X au comparateur
        x_type = gt_row.get('x_type', 'numeric')
        
        return comparator.compare(
            gt_data, extracted_data,
            save=save, save_dir=save_dir, image_name=image_name,
            series_name=f"{match.gt_name}_vs_{match.extracted_name}",
            plot=plot, global_bounds=global_bounds, x_type=x_type
        )
    
    
    def _normalize_plot_type(self, plot_type: str) -> PlotType:
        """Normaliser le type de plot."""
        plot_type_lower = plot_type.lower()
        
        if plot_type_lower in ["scatter"]:
            return PlotType.SCATTER
        elif plot_type_lower in ["plot", "line", 'line plot']:
            return PlotType.LINE
        elif plot_type_lower in ["bar"]:
            return PlotType.BAR
        elif plot_type_lower in ["boxplot"]:
            return PlotType.BOXPLOT
        else:
            print(f"Warning: Type de plot non standard: {plot_type}, utilisation de 'line'")
            return PlotType.LINE
    
    def _create_result_row(self, gt_plot_type: str, image_name: str, 
                        match: MatchResult, metrics: ComparisonMetrics) -> dict:
        """Créer une ligne de résultat avec TOUTES les stats GT et extraites."""
        return {
            'Plot Type': gt_plot_type,
            'Image Name': image_name,
            'Series Name': match.gt_name,
            'Extracted Name': match.extracted_name,
            'MAE': metrics.mae,
            'MAE rel': metrics.mae_rel,
            'Left_miss_rel': metrics.left_miss_rel,
            'Right_miss_rel': metrics.right_miss_rel,
            'x_cutoff': metrics.x_cutoff,
            'Match Method': match.method.value,
            'Match Score': match.score,
            
            # Colonnes de statistiques GT
            'GT Nb Points': metrics.gt_num_points,
            'GT X Min': metrics.gt_x_min,
            'GT X Max': metrics.gt_x_max,
            'GT X Std': metrics.gt_x_std,
            'GT X Range': metrics.gt_x_range,
            'GT Y Min': metrics.gt_y_min,
            'GT Y Max': metrics.gt_y_max,
            'GT Y Std': metrics.gt_y_std,
            'GT Y Range': metrics.gt_y_range,
            
            # Colonnes de statistiques des données extraites
            'Extracted Nb Points': metrics.extracted_num_points,
            
            # === NOUVELLES MÉTRIQUES DE DISTRIBUTION ===
            'Skewness Diff': getattr(metrics, 'skewness_diff', np.nan),
            'Kurtosis Diff': getattr(metrics, 'kurtosis_diff', np.nan),
            'Percentile 90 Diff': getattr(metrics, 'percentile_90_diff', np.nan),
            'Percentile 10 Diff': getattr(metrics, 'percentile_10_diff', np.nan),
            'IQR Diff': getattr(metrics, 'iqr_diff', np.nan),
            
            # === NOUVELLES MÉTRIQUES DE TENDANCE ===
            'Monotonicity Preserved': getattr(metrics, 'monotonicity_preserved', np.nan),
            'Trend Correlation': getattr(metrics, 'trend_correlation', np.nan),
            'Turning Points Diff': getattr(metrics, 'turning_points_diff', np.nan),
            
            # === NOUVELLES MÉTRIQUES DE FORME ===
            'Shape Similarity': getattr(metrics, 'shape_similarity', np.nan),
            'Frechet Distance': getattr(metrics, 'frechet_distance', np.nan),
            'Hausdorff Distance': getattr(metrics, 'hausdorff_distance', np.nan),
            
            # === NOUVELLES MÉTRIQUES DE COUVERTURE ===
            'Coverage X': getattr(metrics, 'coverage_x', np.nan),
            'Coverage Y': getattr(metrics, 'coverage_y', np.nan),
            'Data Density Ratio': getattr(metrics, 'data_density_ratio', np.nan),
            'Missing Regions Count': getattr(metrics, 'missing_regions_count', np.nan),
            
            # === NOUVELLES MÉTRIQUES DE QUALITÉ ===
            'Noise Level Diff': getattr(metrics, 'noise_level_diff', np.nan),
            'Smoothness Diff': getattr(metrics, 'smoothness_diff', np.nan),
            'Outlier Ratio Diff': getattr(metrics, 'outlier_ratio_diff', np.nan),
            
            # === SCORE COMPOSITE ===
            'Composite Similarity Score': getattr(metrics, 'composite_similarity_score', np.nan)
        }


    def _create_error_row(self, gt_plot_type: str, image_name: str, match: MatchResult) -> dict:
        """Créer une ligne d'erreur avec TOUTES les colonnes GT et extraites vides."""
        return {
            'Plot Type': gt_plot_type,
            'Image Name': image_name,
            'Series Name': match.gt_name,
            'Extracted Name': match.extracted_name,
            'MAE': np.nan,
            'MAE rel': np.nan,
            'Left_miss_rel': np.nan,
            'Right_miss_rel': np.nan,
            'x_cutoff': np.nan,
            'Match Method': 'error',
            'Match Score': 0.0,
            
            # Colonnes GT vides pour erreurs
            'GT Nb Points': np.nan,
            'GT X Min': np.nan,
            'GT X Max': np.nan,
            'GT X Std': np.nan,
            'GT X Range': np.nan,
            'GT Y Min': np.nan,
            'GT Y Max': np.nan,
            'GT Y Std': np.nan,
            'GT Y Range': np.nan,
            
            # Colonnes extraites vides pour erreurs
            'Extracted Nb Points': np.nan,
            
            # === NOUVELLES MÉTRIQUES DE DISTRIBUTION (VIDES) ===
            'Skewness Diff': np.nan,
            'Kurtosis Diff': np.nan,
            'Percentile 90 Diff': np.nan,
            'Percentile 10 Diff': np.nan,
            'IQR Diff': np.nan,
            
            # === NOUVELLES MÉTRIQUES DE TENDANCE (VIDES) ===
            'Monotonicity Preserved': np.nan,
            'Trend Correlation': np.nan,
            'Turning Points Diff': np.nan,
            
            # === NOUVELLES MÉTRIQUES DE FORME (VIDES) ===
            'Shape Similarity': np.nan,
            'Frechet Distance': np.nan,
            'Hausdorff Distance': np.nan,
            
            # === NOUVELLES MÉTRIQUES DE COUVERTURE (VIDES) ===
            'Coverage X': np.nan,
            'Coverage Y': np.nan,
            'Data Density Ratio': np.nan,
            'Missing Regions Count': np.nan,
            
            # === NOUVELLES MÉTRIQUES DE QUALITÉ (VIDES) ===
            'Noise Level Diff': np.nan,
            'Smoothness Diff': np.nan,
            'Outlier Ratio Diff': np.nan,
            
            # === SCORE COMPOSITE (VIDE) ===
            'Composite Similarity Score': np.nan
        }
    
    def _empty_result_dataframe(self) -> pd.DataFrame:
        """Créer un DataFrame vide avec TOUTES les colonnes."""
        return pd.DataFrame(columns=[
            'Plot Type', 'Image Name', 'Series Name', 'Extracted Name',
            'MAE', 'MAE rel', 'Left_miss_rel', 'Right_miss_rel', 'x_cutoff',
            'Match Method', 'Match Score',
            
            # Colonnes statistiques GT
            'GT Nb Points', 'GT X Min', 'GT X Max', 'GT X Std', 'GT X Range',
            'GT Y Min', 'GT Y Max', 'GT Y Std', 'GT Y Range',
            
            # Colonnes statistiques extraites
            'Extracted Nb Points',
            
            # Nouvelles métriques de distribution
            'Skewness Diff', 'Kurtosis Diff', 'Percentile 90 Diff', 
            'Percentile 10 Diff', 'IQR Diff',
            
            # Nouvelles métriques de tendance
            'Monotonicity Preserved', 'Trend Correlation', 'Turning Points Diff',
            
            # Nouvelles métriques de forme
            'Shape Similarity', 'Frechet Distance', 'Hausdorff Distance',
            
            # Nouvelles métriques de couverture
            'Coverage X', 'Coverage Y', 'Data Density Ratio', 'Missing Regions Count',
            
            # Nouvelles métriques de qualité
            'Noise Level Diff', 'Smoothness Diff', 'Outlier Ratio Diff',
            
            # Score composite
            'Composite Similarity Score'
        ])
    
# =============================================================================
# LOG
# =============================================================================
class ValidationLogger:
    """Classe pour logger le processus de validation avec détails algorithmiques."""
    
    def __init__(self, save_dir: str, image_name: str):
        self.save_dir = save_dir
        # NOUVEAU: Nettoyer le nom de l'image pour le fichier de log
        self.image_name = sanitize_filename(image_name)
        self.log_entries = []
        self.start_time = datetime.now()
        
        # Créer le répertoire s'il n'existe pas
        os.makedirs(save_dir, exist_ok=True)
        
        # Nom du fichier de log avec nom nettoyé
        self.log_file = os.path.join(save_dir, f"validation_log_{self.image_name}.txt")
        
        # Initialiser le log
        self._init_log()
    
    def _init_log(self):
        """Initialiser le fichier de log."""
        header = f"""
{'='*80}
VALIDATION LOG - {self.image_name}
{'='*80}
Timestamp: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
Save Directory: {self.save_dir}
{'='*80}

"""
        self._write_to_file(header)
        self.log_entries.append(("INIT", header.strip()))
    
    def log_step(self, step_name: str, details: str, data: dict = None):
        """Logger une étape avec détails."""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        entry = f"""
[{timestamp}] === {step_name.upper()} ===
{details}
"""
        
        if data:
            entry += "\nData:\n"
            for key, value in data.items():
                entry += f"  - {key}: {value}\n"
        
        entry += "-" * 50
        
        self._write_to_file(entry)
        self.log_entries.append((step_name, entry.strip()))
    
    def log_error(self, step_name: str, error: Exception, details: str = ""):
        """Logger une erreur."""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        entry = f"""
[{timestamp}] === ERROR IN {step_name.upper()} ===
Error: {str(error)}
Type: {type(error).__name__}
Details: {details}
{'='*50}
"""
        
        self._write_to_file(entry)
        self.log_entries.append(("ERROR", entry.strip()))
        print(f"[ERROR] {step_name}: {str(error)}")

    def log_bounds_calculation(self, bounds: dict):
        """Logger le calcul des bornes globales."""
        details = "Calcul des bornes globales pour normalisation"
        data = {
            "Y Range": f"{bounds['y_range']:.4f}",
            "Y Min-Max": f"[{bounds['y_min']:.4f}, {bounds['y_max']:.4f}]",
            "X Range": f"{bounds.get('x_range', 'N/A (categorical)')}" if 'x_range' in bounds else "N/A (categorical)"
        }
        self.log_step("bounds_calculation", details, data)
    
    def log_matching_results(self, matched_pairs: list):
        """Logger les résultats du matching."""
        details = f"{len(matched_pairs)} appariements trouvés"
        data = {
            "Total Matches": len(matched_pairs),
            "Match Details": [f"{m.gt_name} <-> {m.extracted_name} (method={m.method.value}, score={m.score:.3f})" 
                             for m in matched_pairs]
        }
        self.log_step("matching_results", details, data)
    
 # Classe: ValidationLogger
    def log_plot_type_detection(self, original_type: str, enum_type: str, final_type: str):
        """Logger la détection du type de plot."""
        details = f"Détection du type de plot: {original_type} -> {enum_type}"
        data = {
            "Original Type": original_type,
            "Enum Type": enum_type,
            "Final Type": final_type,
            "Loader Used": "BoxPlotDataLoader" if enum_type in ['vertical_box', 'horizontal_box'] else "DataLoader"
        }
        self.log_step("plot_type_detection", details, data)

    def log_bounds(self, bounds: dict, bound_type: str = "global"):
        """Logger les bornes calculées."""
        details = f"Bornes {bound_type} calculées pour normalisation MAE relative"
        data = {
            "X Range": f"{bounds.get('x_range', 'N/A'):.4f}" if 'x_range' in bounds else "N/A (categorical X)",
            "Y Range": f"{bounds['y_range']:.4f}",
            "X Min-Max": f"[{bounds.get('x_min', 'N/A')}, {bounds.get('x_max', 'N/A')}]",
            "Y Min-Max": f"[{bounds['y_min']:.4f}, {bounds['y_max']:.4f}]"
        }
        self.log_step("bounds_calculation", details, data)
    
    def log_matching(self, gt_names: list, extracted_names: list, matches: list, method_used: str):
        """Logger le processus de matching."""
        details = f"Appariement des séries utilisant: {method_used}"
        data = {
            "GT Series": f"{len(gt_names)} séries: {gt_names}",
            "Extracted Series": f"{len(extracted_names)} séries: {extracted_names}",
            "Matches Found": f"{len(matches)} appariements",
            "Match Details": [f"{m.gt_name} <-> {m.extracted_name} (method={m.method.value}, score={m.score:.3f})" for m in matches]
        }
        self.log_step("series_matching", details, data)
    
    def log_comparison(self, match: 'MatchResult', metrics: 'ComparisonMetrics', comparator_type: str):
        """Logger une comparaison de série avec infos sur les points."""
        details = f"Comparaison {match.gt_name} vs {match.extracted_name} avec {comparator_type}"
        data = {
            "MAE Absolute": f"{metrics.mae:.6f}" if not np.isnan(metrics.mae) else "NaN",
            "MAE Relative": f"{metrics.mae_rel:.6f}" if not np.isnan(metrics.mae_rel) else "NaN",
            "Left Miss (rel)": f"{metrics.left_miss_rel:.6f}" if not np.isnan(metrics.left_miss_rel) else "N/A",
            "Right Miss (rel)": f"{metrics.right_miss_rel:.6f}" if not np.isnan(metrics.right_miss_rel) else "N/A",
            "X Cutoff": f"{metrics.x_cutoff:.6f}" if metrics.x_cutoff is not None else "N/A",
            # NOUVEAU: Infos sur les points
            "GT Points": f"{metrics.gt_num_points}",
            "Extracted Points": f"{metrics.extracted_num_points}",
            "Points Ratio": f"{metrics.extracted_num_points/metrics.gt_num_points:.2f}" if metrics.gt_num_points > 0 else "N/A"
        }
        self.log_step("series_comparison", details, data)
    
    def log_data_loading(self, json_path: str, df: pd.DataFrame, data_type: str):
        """Logger le chargement des données."""
        details = f"Chargement des données {data_type} depuis {os.path.basename(json_path)}"
        
        # Analyser les types de données X
        x_types = df['x_type'].value_counts().to_dict() if 'x_type' in df.columns else {"unknown": len(df)}
        
        data = {
            "File Path": json_path,
            "Series Count": len(df),
            "Series Names": df['name'].tolist(),
            "X Data Types": x_types,
            "Total Data Points": sum(len(row['x']) for _, row in df.iterrows()) if not df.empty else 0
        }
        self.log_step("data_loading", details, data)

    def finalize_log(self, total_results: int, successful: int, errors: int):
        """Finaliser le log avec un résumé des résultats."""
        try:
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            # CORRECTION: Éviter la division par zéro
            if total_results > 0:
                success_rate = f"{(successful/total_results*100):.1f}%"
            else:
                success_rate = "N/A (aucun résultat)"

            summary = f"""

{'='*80}
VALIDATION SUMMARY
{'='*80}
End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Total Duration: {duration:.2f} seconds
Total Series Processed: {total_results}
Successful Comparisons: {successful}
Errors: {errors}
Success Rate: {success_rate}

Generated Files:
- Validation Log: {os.path.basename(self.log_file)}
- Comparison Plots: *.png files
- Statistics Files: GT_VS_Extr_*.txt files

{'='*80}
END OF LOG
{'='*80}
"""

            self._write_to_file(summary)
            self.log_entries.append(("SUMMARY", summary.strip()))
            
        except Exception as e:
            print(f"Erreur lors de la finalisation du log: {e}")
            
    def _write_to_file(self, content: str):
        """Écrire dans le fichier de log."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(content + '\n')

# =============================================================================
# FONCTIONS DE COMPATIBILITÉ AVEC L'API EXISTANTE
# =============================================================================

# Instance globale pour maintenir la compatibilité
_validator = MAEValidator()

# Fonctions wrapper pour maintenir l'API existante
def load_series(json_path: str) -> pd.DataFrame:
    """Wrapper pour maintenir la compatibilité."""
    return DataLoader.load_series(json_path)

def compare_single_image(gt_json_path: str, extracted_json_path: str, 
                        image_name: str, plot_type: str, 
                        save: bool = False, save_dir: str = None, 
                        plot: bool = False) -> pd.DataFrame:
    """Wrapper pour maintenir la compatibilité."""
    return _validator.compare_single_image(
        gt_json_path, extracted_json_path, image_name, plot_type, save, save_dir, plot
    )

def is_generic_name(name: str) -> bool:
    """Wrapper pour maintenir la compatibilité."""
    return NameAnalyzer.is_generic_name(name)

def has_mostly_generic_names(names: List[str], threshold: float = 0.7) -> bool:
    """Wrapper pour maintenir la compatibilité."""
    return NameAnalyzer.has_mostly_generic_names(names, threshold)


