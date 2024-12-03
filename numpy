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

def calculate_global_propensity_scores(df):
    start_time = time.time()
    model = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', random_state=42, max_iter=1000)
    
    df_clean = preprocess_data(df)
    df_encoded, categorical_cols = create_dummy_variables(df_clean)
    
    features_for_model = [col for col in X_VARIABLES if col not in categorical_cols]
    for col in categorical_cols:
        dummy_cols = [c for c in df_encoded.columns if c.startswith(col + '_')]
        features_for_model.extend(dummy_cols)
    
    model.fit(df_encoded[features_for_model], df_encoded[TREATMENT_VAR])
    ps = model.predict_proba(df_encoded[features_for_model])[:, 1]
    
    df_encoded['pscore'] = ps
    print(f"Global propensity score calculation time: {time.time() - start_time:.2f} seconds")
    return df_encoded, features_for_model

def calculate_global_knn(df_encoded):
    start_time = time.time()
    control_df = df_encoded[df_encoded[TREATMENT_VAR] == 0].copy()
    treatment_df = df_encoded[df_encoded[TREATMENT_VAR] == 1].copy()
    
    treated_pscores = treatment_df['pscore'].values.reshape(-1, 1).astype('float32')
    control_pscores = control_df['pscore'].values.reshape(-1, 1).astype('float32')
    
    dimension = 1
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 80
    index.add(control_pscores)
    index.hnsw.efSearch = 40
    
    k = min(N_NEIGHBORS, len(control_df))
    distances, indices = index.search(treated_pscores, k)
    
    print(f"Global KNN calculation time: {time.time() - start_time:.2f} seconds")
    return distances, indices, treatment_df, control_df

def find_matches(distances, indices, treatment_df, control_df, caliper, with_replacement=False):
    start_time = time.time()
    
    n_treated = len(treatment_df)
    n_neighbors = indices.shape[1]
    
    treated_ps = treatment_df['pscore'].values
    control_ps = control_df['pscore'].values
    
    match_matrix = np.full((n_treated, MATCHING_RATIO), -1)
    match_distances = np.full((n_treated, MATCHING_RATIO), np.inf)
    used_controls = np.zeros(len(control_df), dtype=bool)
    
    ps_diff_matrix = np.abs(treated_ps.reshape(-1, 1) - control_ps[indices])
    valid_mask = (ps_diff_matrix <= caliper) & (indices != -1)
    
    for i in range(n_treated):
        valid_indices = np.where(valid_mask[i])[0]
        if not with_replacement:
            valid_indices = valid_indices[~used_controls[indices[i, valid_indices]]]
        
        if len(valid_indices) >= MATCHING_RATIO:
            diffs = ps_diff_matrix[i, valid_indices]
            top_k_idx = np.argpartition(diffs, MATCHING_RATIO)[:MATCHING_RATIO]
            match_indices = valid_indices[top_k_idx]
            
            match_matrix[i] = indices[i, match_indices]
            match_distances[i] = ps_diff_matrix[i, match_indices]
            
            if not with_replacement:
                used_controls[indices[i, match_indices]] = True
    
    matched_pairs = []
    matching_map = {}
    unmatched = 0
    
    for i in range(n_treated):
        if match_matrix[i, 0] != -1:
            treated_idx = treatment_df.index[i]
            
            treated_record = treatment_df.iloc[i].copy()
            treated_record['match_group'] = treated_idx
            treated_record['unit_role'] = 'treated'
            treated_record['original_index'] = treated_idx
            matched_pairs.append(treated_record)
            
            control_indices = []
            for j in range(MATCHING_RATIO):
                control_idx = match_matrix[i, j]
                control_record = control_df.iloc[control_idx].copy()
                control_record['match_group'] = treated_idx
                control_record['unit_role'] = 'control'
                control_record['original_index'] = control_df.index[control_idx]
                matched_pairs.append(control_record)
                control_indices.append(control_df.index[control_idx])
            
            matching_map[treated_idx] = control_indices
        else:
            unmatched += 1
    
    print(f"NumPy find matches time: {time.time() - start_time:.2f} seconds")
    return matched_pairs, unmatched, matching_map

def process_subset_matches(subset_df, global_distances, global_indices, 
                         global_treatment_df, global_control_df, subset_description):
    start_time = time.time()
    subset_treated_indices = subset_df[subset_df[TREATMENT_VAR] == 1].index
    global_treated_indices = global_treatment_df.index
    
    subset_to_global = {idx: pos for pos, idx in enumerate(global_treated_indices) 
                       if idx in subset_treated_indices}
    
    subset_distances = global_distances[[subset_to_global[idx] for idx in subset_treated_indices]]
    subset_indices = global_indices[[subset_to_global[idx] for idx in subset_treated_indices]]
    
    caliper = np.std(global_treatment_df['pscore']) * CALIPER_SD
    matched_pairs, unmatched_treated, matching_map = find_matches(
        subset_distances, subset_indices,
        global_treatment_df.loc[subset_treated_indices],
        global_control_df, caliper, True)
    
    print(f"Subset {subset_description} matching time: {time.time() - start_time:.2f} seconds")
    return matched_pairs, matching_map

def psm_function(df, exact_match=[]):
    total_start_time = time.time()
    
    global X_VARIABLES, TREATMENT_VAR, OUTCOME_VAR, MATCHING_RATIO, N_NEIGHBORS, CALIPER_SD
    X_VARIABLES = ["age", "social_risk_score", "hpd_hyp", "hpd_hyc", "hpd_ast", "hpd_dia"]
    TREATMENT_VAR = "grp_binary"
    OUTCOME_VAR = "social_risk_score"
    MATCHING_RATIO = 2
    N_NEIGHBORS = 25
    CALIPER_SD = 0.25
    
    print("\nCalculating global scores and KNN...")
    df_encoded, features = calculate_global_propensity_scores(df)
    distances, indices, treatment_df, control_df = calculate_global_knn(df_encoded)
    
    list_data_frame = create_subset(df, exact_match)
    all_results = pd.DataFrame()
    all_matching_maps = {}
    
    for i, dfx in enumerate(list_data_frame):
        if len(dfx) > 10:
            subset_description = f"subset_{i}"
            if exact_match:
                match_values = dfx[exact_match].iloc[0].values
                subset_description = "_".join(f"{col}_{val}" for col, val in zip(exact_match, match_values))
            
            matched_df, matching_map = process_subset_matches(
                dfx, distances, indices, treatment_df, control_df, subset_description)
            
            if matched_df:
                all_results = pd.concat([all_results, pd.DataFrame(matched_df)])
                all_matching_maps.update(matching_map)
    
    print(f"\nTotal PSM function execution time: {time.time() - total_start_time:.2f} seconds")
    return all_results, all_matching_maps

if __name__ == "__main__":
    start_time = time.time()
    print("Loading data...")
    df = pd.read_csv("marketing_test_case.csv")
    print(f"Data loading time: {time.time() - start_time:.2f} seconds")
    
    matched_results, matching_maps = psm_function(df, exact_match=['gender_cd', 'hpd_hyp'])
    print(f"\nTotal program execution time: {time.time() - start_time:.2f} seconds")
