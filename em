import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import faiss
import warnings
import time
warnings.filterwarnings('ignore')

def create_subset(df, exact_match):
    if not exact_match:
        return [df]
    df['exact_match_key'] = df[exact_match].astype(str).agg('_'.join, axis=1)
    result = [group for _, group in df.groupby('exact_match_key')]
    return result

def prepare_data_for_matching(df_subset):
    df_clean = df_subset.copy()
    for col in X_VARIABLES:
        if df_clean[col].dtype in ['int64', 'float64']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
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
    
    features_for_model = [col for col in X_VARIABLES if col not in categorical_cols]
    for col in categorical_cols:
        dummy_cols = [c for c in df_encoded.columns if c.startswith(col + '_')]
        features_for_model.extend(dummy_cols)
    
    df_encoded = df_encoded.dropna(subset=features_for_model + [TREATMENT_VAR, OUTCOME_VAR])
    
    model = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', random_state=42, max_iter=1000)
    model.fit(df_encoded[features_for_model], df_encoded[TREATMENT_VAR])
    ps = model.predict_proba(df_encoded[features_for_model])[:, 1]
    df_encoded['pscore'] = ps
    
    return df_encoded, features_for_model

def perform_matching(treatment_groups, control_groups):
    all_treated_pscores = np.vstack([treatment_groups[subset_id]['pscore'].values.reshape(-1, 1) for subset_id in treatment_groups]).astype('float32')
    all_control_pscores = np.vstack([control_groups[subset_id]['pscore'].values.reshape(-1, 1) for subset_id in control_groups]).astype('float32')
    
    dimension = 1
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 80
    index.add(all_control_pscores)
    
    index.hnsw.efSearch = 40
    k = min(N_NEIGHBORS, len(all_control_pscores))
    distances, indices = index.search(all_treated_pscores, k)
    
    start_idx = 0
    all_matches = {}
    for subset_id in treatment_groups:
        n_treated = len(treatment_groups[subset_id])
        all_matches[subset_id] = (
            distances[start_idx:start_idx + n_treated],
            indices[start_idx:start_idx + n_treated]
        )
        start_idx += n_treated
    
    return all_matches

def find_matches(distances, indices, treatment_df, control_df, caliper, with_replacement=False):
    matched_pairs = []
    used_control_indices = set()
    unmatched_treated = 0
    matching_map = {}
    
    treatment_idx_by_ps = treatment_df['pscore'].sort_values().index
    
    for treated_idx in treatment_idx_by_ps:
        i = treatment_df.index.get_loc(treated_idx)
        matches = indices[i]
        treated_pscore = treatment_df['pscore'].iloc[i]
        valid_matches = []
        
        for control_idx in matches:
            if control_idx == -1:
                continue
                
            control_pscore = control_df['pscore'].iloc[control_idx]
            ps_diff = abs(treated_pscore - control_pscore)
            
            if ps_diff <= caliper and (with_replacement or control_idx not in used_control_indices):
                valid_matches.append(control_idx)
                if len(valid_matches) == MATCHING_RATIO:
                    break
        
        if len(valid_matches) == MATCHING_RATIO:
            treated_record = treatment_df.iloc[i].copy()
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
    
    list_data_frame = create_subset(df, exact_match)
    
    treatment_groups = {}
    control_groups = {}
    subset_descriptions = {}
    encoded_dataframes = {}
    
    for i, dfx in enumerate(list_data_frame):
        if len(dfx) > 10:
            subset_description = f"subset_{i}"
            if exact_match:
                match_values = dfx[exact_match].iloc[0].values
                subset_description = "_".join(f"{col}_{val}" for col, val in zip(exact_match, match_values))
            
            print(f"\nProcessing {subset_description}")
            df_encoded, _ = prepare_data_for_matching(dfx)
            encoded_dataframes[subset_description] = df_encoded
            
            treatment_df = df_encoded[df_encoded[TREATMENT_VAR] == 1].copy()
            control_df = df_encoded[df_encoded[TREATMENT_VAR] == 0].copy()
            
            treatment_groups[subset_description] = treatment_df
            control_groups[subset_description] = control_df
            subset_descriptions[i] = subset_description
    
    print("\nPerforming KNN matching")
    knn_results = perform_matching(treatment_groups, control_groups)
    
    all_results = pd.DataFrame()
    all_matching_maps = {}
    
    for i, dfx in enumerate(list_data_frame):
        if len(dfx) > 10:
            subset_description = subset_descriptions[i]
            df_encoded = encoded_dataframes[subset_description]
            
            treatment_df = df_encoded[df_encoded[TREATMENT_VAR] == 1].copy()
            control_df = df_encoded[df_encoded[TREATMENT_VAR] == 0].copy()
            
            caliper = np.std(df_encoded['pscore']) * CALIPER_SD
            distances, indices = knn_results[subset_description]
            
            print(f"\nFinding matches for {subset_description}")
            matched_pairs, unmatched_treated, matching_map = find_matches(
                distances, indices, treatment_df, control_df, caliper, with_replacement=True
            )
            
            matched_df = pd.DataFrame(matched_pairs)
            
            if not matched_df.empty:
                all_results = pd.concat([all_results, matched_df])
                all_matching_maps.update(matching_map)
                
            print(f"Matched treated units: {len(matched_pairs) // (MATCHING_RATIO + 1)}")
            print(f"Unmatched treated units: {unmatched_treated}")
            print(f"Matched control units: {len(matched_pairs) - (len(matched_pairs) // (MATCHING_RATIO + 1))}")
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    return all_results, all_matching_maps

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv("marketing_test_case.csv")
    matched_results, matching_maps = psm_function(df, exact_match=['gender_cd'])
