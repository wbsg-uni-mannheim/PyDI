"""
Evaluation utilities for information extraction tasks.

Adapted from SelfRefinement4ExtractGPT repository:
https://github.com/wbsg-uni-mannheim/SelfRefinement4ExtractGPT

Provides evaluation metrics including precision, recall, and F1 score
for comparing predicted vs. target attribute values.
"""

import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import json


def calculate_evaluation_metrics(target_value: Any, predicted_value: Any) -> str:
    """
    Calculate evaluation metrics by comparing target and predicted values.
    
    Parameters
    ----------
    target_value : Any
        Ground truth value
    predicted_value : Any
        Predicted value
        
    Returns
    -------
    str
        Evaluation category:
        - 'NN': No target, No prediction
        - 'NV': No target, Valid prediction  
        - 'VN': Valid target, No prediction
        - 'VC': Valid correct prediction
        - 'VW': Valid wrong prediction
    """
    has_target = target_value is not None and str(target_value).strip() != ""
    has_prediction = predicted_value is not None and str(predicted_value).strip() != ""
    
    if not has_target and not has_prediction:
        return 'NN'
    elif not has_target and has_prediction:
        return 'NV'
    elif has_target and not has_prediction:
        return 'VN'
    elif has_target and has_prediction:
        if str(target_value).strip().lower() == str(predicted_value).strip().lower():
            return 'VC'
        else:
            return 'VW'
    
    return 'NN'  # fallback


def calculate_precision_recall_f1(evaluation_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score from evaluation counts.
    
    Parameters
    ----------
    evaluation_counts : Dict[str, int]
        Dictionary with counts for each evaluation category
        
    Returns
    -------
    Dict[str, float]
        Dictionary with precision, recall, and f1 values
    """
    # Get counts, defaulting to 0 if not present
    nn = evaluation_counts.get('NN', 0)
    nv = evaluation_counts.get('NV', 0) 
    vn = evaluation_counts.get('VN', 0)
    vc = evaluation_counts.get('VC', 0)
    vw = evaluation_counts.get('VW', 0)
    
    # Calculate precision
    total_predictions = nv + vc + vw
    precision = (vc / total_predictions * 100) if total_predictions > 0 else 0.0
    
    # Calculate recall
    total_targets = vn + vc + vw
    recall = (vc / total_targets * 100) if total_targets > 0 else 0.0
    
    # Calculate F1
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_predictions(predictions_df: pd.DataFrame, 
                        target_df: pd.DataFrame,
                        attribute_columns: List[str]) -> Dict[str, Any]:
    """
    Evaluate predictions against target values for multiple attributes.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame containing predicted values
    target_df : pd.DataFrame
        DataFrame containing target/ground truth values
    attribute_columns : List[str]
        List of attribute column names to evaluate
        
    Returns
    -------
    Dict[str, Any]
        Evaluation results including micro and macro scores
    """
    results = {
        'attribute_results': {},
        'micro_scores': {},
        'macro_scores': {},
        'total_counts': {}
    }
    
    total_evaluation_counts = {'NN': 0, 'NV': 0, 'VN': 0, 'VC': 0, 'VW': 0}
    attribute_scores = []
    
    for attribute in attribute_columns:
        if attribute not in predictions_df.columns or attribute not in target_df.columns:
            print(f"Warning: Attribute '{attribute}' not found in both DataFrames")
            continue
            
        evaluation_counts = {'NN': 0, 'NV': 0, 'VN': 0, 'VC': 0, 'VW': 0}
        
        # Compare predictions vs targets for this attribute
        for i in range(len(predictions_df)):
            pred_val = predictions_df[attribute].iloc[i] if i < len(predictions_df) else None
            target_val = target_df[attribute].iloc[i] if i < len(target_df) else None
            
            eval_category = calculate_evaluation_metrics(target_val, pred_val)
            evaluation_counts[eval_category] += 1
            total_evaluation_counts[eval_category] += 1
        
        # Calculate scores for this attribute
        attr_scores = calculate_precision_recall_f1(evaluation_counts)
        
        results['attribute_results'][attribute] = {
            'counts': evaluation_counts,
            'scores': attr_scores
        }
        
        attribute_scores.append(attr_scores)
    
    # Calculate micro scores (overall performance)
    results['micro_scores'] = calculate_precision_recall_f1(total_evaluation_counts)
    results['total_counts'] = total_evaluation_counts
    
    # Calculate macro scores (average across attributes)
    if attribute_scores:
        macro_precision = sum(s['precision'] for s in attribute_scores) / len(attribute_scores)
        macro_recall = sum(s['recall'] for s in attribute_scores) / len(attribute_scores)
        macro_f1 = sum(s['f1'] for s in attribute_scores) / len(attribute_scores)
        
        results['macro_scores'] = {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        }
    
    return results


def print_evaluation_results(results: Dict[str, Any]) -> None:
    """
    Print formatted evaluation results.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary from evaluate_predictions
    """
    print("="*60)
    print("INFORMATION EXTRACTION EVALUATION RESULTS")
    print("="*60)
    
    print("\n--- MICRO SCORES (Overall Performance) ---")
    micro = results['micro_scores']
    print(f"Precision: {micro['precision']:.2f}%")
    print(f"Recall:    {micro['recall']:.2f}%")
    print(f"F1 Score:  {micro['f1']:.2f}%")
    
    if 'macro_scores' in results and results['macro_scores']:
        print("\n--- MACRO SCORES (Average Across Attributes) ---")
        macro = results['macro_scores']
        print(f"Precision: {macro['precision']:.2f}%")
        print(f"Recall:    {macro['recall']:.2f}%")
        print(f"F1 Score:  {macro['f1']:.2f}%")
    
    print("\n--- ATTRIBUTE-LEVEL RESULTS ---")
    for attr, attr_results in results['attribute_results'].items():
        scores = attr_results['scores']
        counts = attr_results['counts']
        print(f"\n{attr}:")
        print(f"  Precision: {scores['precision']:.2f}%")
        print(f"  Recall:    {scores['recall']:.2f}%") 
        print(f"  F1 Score:  {scores['f1']:.2f}%")
        print(f"  Counts: VC={counts['VC']}, VW={counts['VW']}, VN={counts['VN']}, NV={counts['NV']}, NN={counts['NN']}")
    
    print("\n--- TOTAL COUNTS ---")
    total = results['total_counts']
    print(f"Valid Correct (VC):      {total['VC']}")
    print(f"Valid Wrong (VW):        {total['VW']}")
    print(f"Valid Missing (VN):      {total['VN']}")
    print(f"Invalid Extra (NV):      {total['NV']}")
    print(f"No Target/Prediction:    {total['NN']}")
    print("="*60)


def load_jsonl_targets(file_path: str) -> pd.DataFrame:
    """
    Load target data from JSONL format used in OA-Mine dataset.
    
    Parameters
    ----------
    file_path : str
        Path to JSONL file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with flattened target attributes
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            
            # Extract input text and category
            row = {
                'input': record.get('input', ''),
                'category': record.get('category', '')
            }
            
            # Flatten target_scores to individual columns
            target_scores = record.get('target_scores', {})
            for attr, values_dict in target_scores.items():
                # Take the first key as the value (assuming single value per attribute)
                if values_dict:
                    row[attr] = list(values_dict.keys())[0]
                else:
                    row[attr] = None
            
            data.append(row)
    
    return pd.DataFrame(data)