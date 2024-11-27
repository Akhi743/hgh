import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
import faiss
import warnings
warnings.filterwarnings('ignore')

def create_subset(df, exact_match):
    if not exact_match:
        df['exact_match_key'] = '1'
    else:
        df['exact_match_key'] = df[exact_match].astype(str).agg('_'.join, axis=1)
    return [group for _, group in df.groupby('exact_match_key')], df['exact_match_key']

def preprocess_and_score(df, x_variables, treatment_var, outcome_var):
    df_clean = df.copy()
    
    # Handle missing values
    for col in x_variables:
        if df_clean[col].dtype in ['int64', 'float64']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    # Create dummy variables
    df_encoded = df_clean.copy()
    categorical_cols = []
    
    for col in x_variables:
        if col in df_encoded.columns:
            if df_encoded[col].nunique() == 2 and df_encoded[col].dtype in ['int64', 'float64']:
                df_encoded[col] = df_encoded[col].map({0: 'A', 1: 'B'})
            
            if df_encoded[col].dtype == 'object' or df_encoded[col].nunique() < 10:
                categorical_cols.append(col)
                unique_vals = df_encoded[col].dropna().unique()
                for val in unique_vals:
                    df_encoded[f"{col}_{val}"] = (df_encoded[col] == val).astype(int)
                df_encoded = df_encoded.drop(columns=[col])
    
    # Prepare features
    features = [col for col in x_variables if col not in categorical_cols]
    for col in categorical_cols:
        features.extend([c for c in df_encoded.columns if c.startswith(col + '_')])
    
    df_encoded = df_encoded.dropna(subset=features + [treatment_var, outcome_var])
    
    # Calculate propensity scores
    model = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', random_state=42, max_iter=1000)
    model.fit(df_encoded[features], df_encoded[treatment_var])
    df_encoded['pscore'] = model.predict_proba(df_encoded[features])[:, 1]
    
    return df_encoded

def perform_matching(df_encoded, treatment_var, n_neighbors, matching_ratio, caliper_sd):
    treatment_df = df_encoded[df_encoded[treatment_var] == 1]
    control_df = df_encoded[df_encoded[treatment_var] == 0]
    
    # Single KNN for all data
    treated_pscores = treatment_df['pscore'].values.reshape(-1, 1).astype('float32')
    control_pscores = control_df['pscore'].values.reshape(-1, 1).astype('float32')
    
    index = faiss.IndexHNSWFlat(1, 32)
    index.hnsw.efConstruction = 80
    index.add(control_pscores)
    
    index.hnsw.efSearch = 40
    k = min(n_neighbors, len(control_df))
    distances, indices = index.search(treated_pscores, k)
    
    # Find matches by subset
    matched_pairs = []
    matching_maps = {}
    
    for exact_key in df_encoded['exact_match_key'].unique():
        mask_treat = treatment_df['exact_match_key'] == exact_key
        mask_control = control_df['exact_match_key'] == exact_key
        
        subset_treat = treatment_df[mask_treat]
        subset_control = control_df[mask_control]
        
        if len(subset_treat) < 1 or len(subset_control) < matching_ratio:
            continue
            
        caliper = np.std(df_encoded[df_encoded['exact_match_key'] == exact_key]['pscore']) * caliper_sd
        subset_indices = np.where(mask_treat)[0]
        
        for i, treat_idx in enumerate(subset_indices):
            current_indices = indices[treat_idx]
            current_distances = distances[treat_idx]
            
            valid_matches = []
            for j, ctrl_idx in enumerate(current_indices):
                if ctrl_idx == -1 or not mask_control.iloc[ctrl_idx]:
                    continue
                if current_distances[j] <= caliper:
                    valid_matches.append(ctrl_idx)
                if len(valid_matches) == matching_ratio:
                    break
                    
            if len(valid_matches) == matching_ratio:
                treat_record = treatment_df.iloc[treat_idx].copy()
                treat_record['match_group'] = treat_idx
                treat_record['unit_role'] = 'treated'
                matched_pairs.append(treat_record)
                
                ctrl_indices = []
                for match in valid_matches:
                    ctrl_record = control_df.iloc[match].copy()
                    ctrl_record['match_group'] = treat_idx
                    ctrl_record['unit_role'] = 'control'
                    matched_pairs.append(ctrl_record)
                    ctrl_indices.append(control_df.index[match])
                
                matching_maps[treatment_df.index[treat_idx]] = ctrl_indices
    
    return pd.DataFrame(matched_pairs), matching_maps

def psm_function(df, exact_match=[]):
    x_variables = ["age", "social_risk_score", "hpd_hyp", "hpd_hyc", "hpd_ast", "hpd_dia"]
    treatment_var = "grp_binary"
    outcome_var = "social_risk_score"
    matching_ratio = 2
    n_neighbors = 25
    caliper_sd = 0.25
    
    # Create subsets but keep data in single DataFrame
    _, df['exact_match_key'] = create_subset(df, exact_match)
    
    # Preprocess and calculate scores
    df_encoded = preprocess_and_score(df, x_variables, treatment_var, outcome_var)
    
    # Perform matching
    matched_results, matching_maps = perform_matching(
        df_encoded, treatment_var, n_neighbors, matching_ratio, caliper_sd)
    
    return matched_results, matching_maps

if __name__ == "__main__":
    df = pd.read_csv("marketing_test_case.csv")
    matched_results, matching_maps = psm_function(df, exact_match=['gender_cd'])
