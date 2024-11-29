import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import faiss
import warnings
import time

warnings.filterwarnings('ignore')

def create_subset(df, exact_match):
    start_time = time.time()
    if not exact_match:
        return [df]
    
    df['exact_match_key'] = df[exact_match].astype(str).agg('_'.join, axis=1)
    result = [group for _, group in df.groupby('exact_match_key')]
    print(f"Subset creation time: {time.time() - start_time:.2f} seconds")
    return result

def preprocess_data(df_subset):
    start_time = time.time()
    df_clean = df_subset.copy()
    for col in X_VARIABLES:
        if df_clean[col].dtype in ['int64', 'float64']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    print(f"Preprocessing time: {time.time() - start_time:.2f} seconds")
    return df_clean

def create_dummy_variables(df_clean):
    start_time = time.time()
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
    print(f"Dummy variable creation time: {time.time() - start_time:.2f} seconds")
    return df_encoded, categorical_cols

def calculate_propensity_scores(df_encoded, features_for_model):
    start_time = time.time()
    model = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', random_state=42, max_iter=1000)
    model.fit(df_encoded[features_for_model], df_encoded[TREATMENT_VAR])
    result = model.predict_proba(df_encoded[features_for_model])[:, 1]
    print(f"Propensity score calculation time: {time.time() - start_time:.2f} seconds")
    return result

def perform_matching_batch(treated_pscores_list, control_pscores_list):
    start_time = time.time()
    
    all_treated_pscores = np.vstack(treated_pscores_list).astype('float32')
    all_control_pscores = np.vstack(control_pscores_list).astype('float32')
    
    dimension = 1
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 80
    index.add(all_control_pscores)
    
    index.hnsw.efSearch = 40
    k = min(N_NEIGHBORS, len(all_control_pscores))
    distances, indices = index.search(all_treated_pscores, k)
    
    print(f"Batch FAISS matching time: {time.time() - start_time:.2f} seconds")
    return distances, indices

def find_matches(distances, indices, treatment_df, control_df, caliper, with_replacement=False):
    start_time = time.time()
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
    
    print(f"Find matches time: {time.time() - start_time:.2f} seconds")
    return matched_pairs, unmatched_treated, matching_map

def prepare_subset_data(df_subset):
    df_clean = preprocess_data(df_subset)
    df_encoded, categorical_cols = create_dummy_variables(df_clean)
    
    features_for_model = [col for col in X_VARIABLES if col not in categorical_cols]
    for col in categorical_cols:
        dummy_cols = [c for c in df_encoded.columns if c.startswith(col + '_')]
        features_for_model.extend(dummy_cols)
    
    df_encoded = df_encoded.dropna(subset=features_for_model + [TREATMENT_VAR, OUTCOME_VAR])
    
    ps = calculate_propensity_scores(df_encoded, features_for_model)
    df_encoded['pscore'] = ps
    
    caliper = np.std(df_encoded['pscore']) * CALIPER_SD
    treatment_df = df_encoded[df_encoded[TREATMENT_VAR] == 1].copy()
    control_df = df_encoded[df_encoded[TREATMENT_VAR] == 0].copy()
    
    treated_pscores = treatment_df['pscore'].values.reshape(-1, 1)
    control_pscores = control_df['pscore'].values.reshape(-1, 1)
    
    return treatment_df, control_df, caliper, treated_pscores, control_pscores

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
    
    subset_data = []
    treated_pscores_list = []
    control_pscores_list = []
    
    for i, dfx in enumerate(list_data_frame):
        if len(dfx) > 10:
            subset_description = f"subset_{i}"
            if exact_match:
                match_values = dfx[exact_match].iloc[0].values
                subset_description = "_".join(f"{col}_{val}" for col, val in zip(exact_match, match_values))
            
            treatment_df, control_df, caliper, treated_pscores, control_pscores = prepare_subset_data(dfx)
            subset_data.append((treatment_df, control_df, caliper, subset_description))
            treated_pscores_list.append(treated_pscores)
            control_pscores_list.append(control_pscores)
    
    distances, indices = perform_matching_batch(treated_pscores_list, control_pscores_list)
    
    all_results = pd.DataFrame()
    all_matching_maps = {}
    current_idx = 0
    
    for i, (treatment_df, control_df, caliper, subset_description) in enumerate(subset_data):
        subset_size = len(treatment_df)
        subset_distances = distances[current_idx:current_idx + subset_size]
        subset_indices = indices[current_idx:current_idx + subset_size]
        
        matched_pairs, unmatched_treated, matching_map = find_matches(
            subset_distances, subset_indices,
            treatment_df, control_df,
            caliper, with_replacement=True
        )
        
        if matched_pairs:
            matched_df = pd.DataFrame(matched_pairs)
            all_results = pd.concat([all_results, matched_df])
            all_matching_maps.update(matching_map)
            
            print(f"\nMatching Statistics for {subset_description}:")
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
