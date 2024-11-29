import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import faiss
import warnings
import time

warnings.filterwarnings('ignore')

def preprocess_all_data(df, exact_match):
    processed_data = []
    
    # Split into subsets based on exact matching
    if exact_match:
        df['exact_match_key'] = df[exact_match].astype(str).agg('_'.join, axis=1)
        groups = [group for _, group in df.groupby('exact_match_key')]
    else:
        groups = [df]
    
    for i, dfx in enumerate(groups):
        if len(dfx) <= 10:
            continue
            
        # Create subset description
        subset_description = f"subset_{i}"
        if exact_match:
            match_values = dfx[exact_match].iloc[0].values
            subset_description = "_".join(f"{col}_{val}" for col, val in zip(exact_match, match_values))
        
        # Preprocess
        df_clean = dfx.copy()
        for col in X_VARIABLES:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        # Create dummy variables
        df_encoded = df_clean.copy()
        categorical_cols = []
        for col in X_VARIABLES:
            if col in df_encoded.columns:
                if df_encoded[col].nunique() == 2 and df_encoded[col].dtype in ['int64', 'float64']:
                    df_encoded[col] = df_encoded[col].map({0: 'A', 1: 'B'})
                
                if df_encoded[col].dtype == 'object' or df_encoded[col].nunique() < 10:
                    categorical_cols.append(col)
                    unique_vals = df_encoded[col].dropna().unique()
                    for val in unique_vals:
                        col_name = f"{col}_{val}"
                        df_encoded[col_name] = (df_encoded[col] == val).astype(int)
                    df_encoded = df_encoded.drop(columns=[col])
        
        # Calculate propensity scores
        features = [col for col in X_VARIABLES if col not in categorical_cols]
        features.extend([c for col in categorical_cols for c in df_encoded.columns if c.startswith(col + '_')])
        df_encoded = df_encoded.dropna(subset=features + [TREATMENT_VAR, OUTCOME_VAR])
        
        model = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', random_state=42, max_iter=1000)
        model.fit(df_encoded[features], df_encoded[TREATMENT_VAR])
        df_encoded['pscore'] = model.predict_proba(df_encoded[features])[:, 1]
        
        caliper = np.std(df_encoded['pscore']) * CALIPER_SD
        processed_data.append({
            'description': subset_description,
            'data': df_encoded,
            'caliper': caliper,
            'features': features
        })
    
    return processed_data

def perform_knn_matching(processed_data):
    all_treated_data = []
    all_control_data = []
    treated_pscores = []
    control_pscores = []
    control_offset = 0
    
    for subset in processed_data:
        df = subset['data']
        treatment = df[df[TREATMENT_VAR] == 1]
        control = df[df[TREATMENT_VAR] == 0]
        
        if not treatment.empty and not control.empty:
            treated_pscores.append(treatment['pscore'].values.reshape(-1, 1).astype('float32'))
            control_pscores.append(control['pscore'].values.reshape(-1, 1).astype('float32'))
            all_treated_data.append(treatment)
            all_control_data.append(control)
            
            subset['treatment_size'] = len(treatment)
            subset['control_size'] = len(control)
            subset['control_offset'] = control_offset
            control_offset += len(control)
    
    if not treated_pscores or not control_pscores:
        return None, None, None, None
        
    treated_pscores = np.vstack(treated_pscores)
    control_pscores = np.vstack(control_pscores)
    
    index = faiss.IndexHNSWFlat(1, 32)
    index.hnsw.efConstruction = 80
    index.add(control_pscores)
    
    distances, indices = index.search(treated_pscores, N_NEIGHBORS)
    
    return distances, indices, pd.concat(all_treated_data), pd.concat(all_control_data)

def find_matches(distances, indices, treatment_df, control_df, caliper, with_replacement=False):
    matched_pairs = []
    used_control_indices = set()
    unmatched_treated = 0
    matching_map = {}
    
    treatment_idx_list = treatment_df.index.tolist()
    
    for i, treated_idx in enumerate(treatment_idx_list):
        matches = indices[i]
        treated_pscore = treatment_df.loc[treated_idx, 'pscore']
        valid_matches = []
        
        for control_idx in matches:
            if control_idx == -1 or control_idx >= len(control_df.index):
                continue
            
            control_pscore = control_df.iloc[control_idx]['pscore']
            ps_diff = abs(treated_pscore - control_pscore)
            
            if ps_diff <= caliper and (with_replacement or control_idx not in used_control_indices):
                valid_matches.append(control_idx)
                if len(valid_matches) == MATCHING_RATIO:
                    break
        
        if len(valid_matches) == MATCHING_RATIO:
            treated_record = treatment_df.loc[treated_idx].copy()
            treated_record['match_group'] = treated_idx
            treated_record['unit_role'] = 'treated'
            treated_record['original_index'] = treated_idx
            matched_pairs.append(treated_record)
            
            control_indices = []
            for match in valid_matches:
                control_record = control_df.iloc[match].copy()
                control_record['match_group'] = treated_idx
                control_record['unit_role'] = 'control'
                control_record['original_index'] = control_df.index[match]
                matched_pairs.append(control_record)
                used_control_indices.add(match)
                control_indices.append(control_df.index[match])
            
            matching_map[treated_idx] = control_indices
        else:
            unmatched_treated += 1
            
    return matched_pairs, unmatched_treated, matching_map

def psm_function(df, exact_match=[]):
    start_time = time.time()
    
    global X_VARIABLES, TREATMENT_VAR, OUTCOME_VAR, MATCHING_RATIO, N_NEIGHBORS, CALIPER_SD
    X_VARIABLES = ["age", "social_risk_score", "hpd_hyp", "hpd_hyc", "hpd_ast", "hpd_dia"]
    TREATMENT_VAR = "grp_binary"
    OUTCOME_VAR = "social_risk_score"
    MATCHING_RATIO = 2
    N_NEIGHBORS = 25
    CALIPER_SD = 0.25
    
    # Process all data first
    processed_data = preprocess_all_data(df, exact_match)
    
    # Perform KNN matching once for all data
    distances, indices, combined_treatment, combined_control = perform_knn_matching(processed_data)
    
    if distances is None:
        return pd.DataFrame(), {}
    
    # Process matches for each subset
    all_results = pd.DataFrame()
    all_matching_maps = {}
    current_idx = 0
    
    for subset in processed_data:
        subset_size = subset['treatment_size']
        control_offset = subset['control_offset']
        
        subset_indices = indices[current_idx:current_idx + subset_size]
        subset_indices = subset_indices + control_offset
        
        subset_treatment = combined_treatment.iloc[current_idx:current_idx + subset_size]
        subset_control = combined_control.iloc[control_offset:control_offset + subset['control_size']]
        
        matched_pairs, unmatched_treated, matching_map = find_matches(
            distances[current_idx:current_idx + subset_size],
            subset_indices,
            subset_treatment,
            subset_control,
            subset['caliper'],
            with_replacement=True
        )
        
        if matched_pairs:
            matched_df = pd.DataFrame(matched_pairs)
            all_results = pd.concat([all_results, matched_df])
            all_matching_maps.update(matching_map)
            
            print(f"\nMatching Statistics for {subset['description']}:")
            print(f"Successfully matched treated units: {len(matched_pairs) // (MATCHING_RATIO + 1)}")
            print(f"Unmatched treated units: {unmatched_treated}")
            print(f"Total matched control units: {len(matched_pairs) - (len(matched_pairs) // (MATCHING_RATIO + 1))}")
        
        current_idx += subset_size
    
    print(f"\nTotal PSM function execution time: {time.time() - start_time:.2f} seconds")
    return all_results, all_matching_maps

if __name__ == "__main__":
    start_time = time.time()
    print("Loading data...")
    load_start = time.time()
    df = pd.read_csv("marketing_test_case.csv")
    print(f"Data loading time: {time.time() - load_start:.2f} seconds")
    
    matched_results, matching_maps = psm_function(df, exact_match=['gender_cd', 'hpd_hyp'])
    print(f"\nTotal program execution time: {time.time() - start_time:.2f} seconds")
