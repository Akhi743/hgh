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

def process_subset_matches(subset_df, global_distances, global_indices, 
                         global_treatment_df, global_control_df, subset_description):
    start_time = time.time()
    
    subset_treated_indices = subset_df[subset_df[TREATMENT_VAR] == 1].index
    global_treated_indices = global_treatment_df.index
    
    # Map subset indices to global indices
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

def find_matches(distances, indices, treatment_df, control_df, caliper, with_replacement=False):
    start_time = time.time()
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
    
    print(f"Find matches time: {time.time() - start_time:.2f} seconds")
    return matched_pairs, unmatched_treated, matching_map

def psm_function(df, exact_match=[]):
    total_start_time = time.time()
    
    global X_VARIABLES, TREATMENT_VAR, OUTCOME_VAR, MATCHING_RATIO, N_NEIGHBORS, CALIPER_SD
    X_VARIABLES = ["age", "social_risk_score", "hpd_hyp", "hpd_hyc", "hpd_ast", "hpd_dia"]
    TREATMENT_VAR = "grp_binary"
    OUTCOME_VAR = "social_risk_score"
    MATCHING_RATIO = 2
    N_NEIGHBORS = 25
    CALIPER_SD = 0.25
    
    # Calculate global propensity scores and KNN once
    print("\nCalculating global scores and KNN...")
    df_encoded, features = calculate_global_propensity_scores(df)
    distances, indices, treatment_df, control_df = calculate_global_knn(df_encoded)
    
    # Create subsets
    list_data_frame = create_subset(df, exact_match)
    
    # Process each subset using global KNN results
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

# Main execution
if __name__ == "__main__":
    start_time = time.time()
    print("Loading data...")
    df = pd.read_csv("marketing_test_case.csv")
    print(f"Data loading time: {time.time() - start_time:.2f} seconds")
    
    matched_results, matching_maps = psm_function(df, exact_match=['gender_cd', 'hpd_hyp'])
    print(f"\nTotal program execution time: {time.time() - start_time:.2f} seconds")






import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import faiss
import warnings
import time
from typing import List, Dict, Tuple, Set

warnings.filterwarnings('ignore')

def create_subset(df: pd.DataFrame, exact_match: List[str]) -> List[pd.DataFrame]:
    """Creates subsets of data based on exact matching criteria"""
    start_time = time.time()
    if not exact_match:
        return [df]
    
    df['exact_match_key'] = df[exact_match].astype(str).agg('_'.join, axis=1)
    result = [group for _, group in df.groupby('exact_match_key')]
    print(f"Subset creation time: {time.time() - start_time:.2f} seconds")
    return result

def preprocess_data(df_subset: pd.DataFrame, X_VARIABLES: List[str]) -> pd.DataFrame:
    """Handle missing values and create dummy variables"""
    start_time = time.time()
    df_clean = df_subset.copy()
    for col in X_VARIABLES:
        if df_clean[col].dtype in ['int64', 'float64']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    print(f"Preprocessing time: {time.time() - start_time:.2f} seconds")
    return df_clean

def create_dummy_variables(df_clean: pd.DataFrame, X_VARIABLES: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Create dummy variables for categorical features"""
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

def calculate_propensity_scores(df_encoded: pd.DataFrame, features_for_model: List[str], TREATMENT_VAR: str) -> np.ndarray:
    """Calculate propensity scores using logistic regression"""
    start_time = time.time()
    model = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', random_state=42, max_iter=1000)
    model.fit(df_encoded[features_for_model], df_encoded[TREATMENT_VAR])
    result = model.predict_proba(df_encoded[features_for_model])[:, 1]
    print(f"Propensity score calculation time: {time.time() - start_time:.2f} seconds")
    return result

def perform_matching(treatment_df: pd.DataFrame, control_df: pd.DataFrame, N_NEIGHBORS: int) -> Tuple[np.ndarray, np.ndarray]:
    """Perform nearest neighbor matching using FAISS"""
    start_time = time.time()
    treated_pscores = treatment_df['pscore'].values.reshape(-1, 1).astype('float32')
    control_pscores = control_df['pscore'].values.reshape(-1, 1).astype('float32')
    
    dimension = 1
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 80
    index.add(control_pscores)
    
    index.hnsw.efSearch = 40
    k = min(N_NEIGHBORS, len(control_df))
    distances, indices = index.search(treated_pscores, k)
    print(f"FAISS matching time: {time.time() - start_time:.2f} seconds")
    return distances, indices

def find_matches_batch(
    distances: np.ndarray,
    indices: np.ndarray,
    treatment_df: pd.DataFrame,
    control_df: pd.DataFrame,
    caliper: float,
    MATCHING_RATIO: int,
    batch_size: int = 1000,
    with_replacement: bool = False
) -> Tuple[List[pd.Series], int, Dict]:
    """Find matching pairs within caliper using batch processing"""
    start_time = time.time()
    matched_pairs = []
    used_control_indices = set()
    unmatched_treated = 0
    matching_map = {}
    
    # Sort treatment cases by propensity score
    treatment_idx_by_ps = treatment_df['pscore'].sort_values().index
    
    # Calculate number of batches
    n_batches = len(treatment_idx_by_ps) // batch_size + (1 if len(treatment_idx_by_ps) % batch_size != 0 else 0)
    
    for batch_num in range(n_batches):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, len(treatment_idx_by_ps))
        batch_indices = treatment_idx_by_ps[batch_start:batch_end]
        
        # Process batch
        batch_matches = []
        batch_used_controls = set()
        
        for treated_idx in batch_indices:
            i = treatment_df.index.get_loc(treated_idx)
            matches = indices[i]
            treated_pscore = treatment_df['pscore'].iloc[i]
            valid_matches = []
            
            for control_idx in matches:
                if control_idx == -1:
                    continue
                
                if not with_replacement and (control_idx in used_control_indices or control_idx in batch_used_controls):
                    continue
                
                control_pscore = control_df['pscore'].iloc[control_idx]
                ps_diff = abs(treated_pscore - control_pscore)
                
                if ps_diff <= caliper:
                    valid_matches.append(control_idx)
                    if len(valid_matches) == MATCHING_RATIO:
                        break
            
            if len(valid_matches) == MATCHING_RATIO:
                # Store treatment unit
                treated_record = treatment_df.iloc[i].copy()
                treated_record['match_group'] = treated_idx
                treated_record['unit_role'] = 'treated'
                treated_record['original_index'] = treated_idx
                batch_matches.append(treated_record)
                
                # Store control units
                control_indices = []
                for match in valid_matches:
                    control_record = control_df.iloc[match].copy()
                    control_record['match_group'] = treated_idx
                    control_record['unit_role'] = 'control'
                    control_record['original_index'] = control_df.index[match]
                    batch_matches.append(control_record)
                    batch_used_controls.add(match)
                    control_indices.append(control_df.index[match])
                
                matching_map[treated_idx] = control_indices
            else:
                unmatched_treated += 1
        
        # Update global matches and used controls after batch processing
        matched_pairs.extend(batch_matches)
        used_control_indices.update(batch_used_controls)
    
    print(f"Find matches time: {time.time() - start_time:.2f} seconds")
    return matched_pairs, unmatched_treated, matching_map

def run_analysis(
    df_subset: pd.DataFrame,
    subset_description: str,
    X_VARIABLES: List[str],
    TREATMENT_VAR: str,
    OUTCOME_VAR: str,
    MATCHING_RATIO: int,
    N_NEIGHBORS: int,
    CALIPER_SD: float,
    BATCH_SIZE: int
) -> Tuple[pd.DataFrame, Dict]:
    """Main analysis function"""
    start_time = time.time()
    print(f"\nAnalyzing {subset_description} subset...")
    
    # Preprocessing
    df_clean = preprocess_data(df_subset, X_VARIABLES)
    
    # Feature engineering
    df_encoded, categorical_cols = create_dummy_variables(df_clean, X_VARIABLES)
    
    # Prepare features for model
    features_for_model = [col for col in X_VARIABLES if col not in categorical_cols]
    for col in categorical_cols:
        dummy_cols = [c for c in df_encoded.columns if c.startswith(col + '_')]
        features_for_model.extend(dummy_cols)
    
    # Handle missing values
    df_encoded = df_encoded.dropna(subset=features_for_model + [TREATMENT_VAR, OUTCOME_VAR])
    
    # Calculate propensity scores
    ps = calculate_propensity_scores(df_encoded, features_for_model, TREATMENT_VAR)
    df_encoded['pscore'] = ps
    
    # Prepare for matching
    caliper = np.std(df_encoded['pscore']) * CALIPER_SD
    treatment_df = df_encoded[df_encoded[TREATMENT_VAR] == 1].copy()
    control_df = df_encoded[df_encoded[TREATMENT_VAR] == 0].copy()
    
    # Perform matching
    distances, indices = perform_matching(treatment_df, control_df, N_NEIGHBORS)
    matched_pairs, unmatched_treated, matching_map = find_matches_batch(
        distances, indices, treatment_df, control_df, caliper, 
        MATCHING_RATIO, BATCH_SIZE, with_replacement=True
    )
    
    # Create matched dataset
    matched_df = pd.DataFrame(matched_pairs)
    
    # Print results
    print(f"\nMatching Statistics for {subset_description}:")
    print(f"Successfully matched treated units: {len(matched_pairs) // (MATCHING_RATIO + 1)}")
    print(f"Unmatched treated units: {unmatched_treated}")
    print(f"Total matched control units: {len(matched_pairs) - (len(matched_pairs) // (MATCHING_RATIO + 1))}")
    print(f"Total analysis time for {subset_description}: {time.time() - start_time:.2f} seconds")
    
    return matched_df, matching_map

def psm_function(df: pd.DataFrame, exact_match: List[str] = []) -> Tuple[pd.DataFrame, Dict]:
    """Main function to perform PSM with exact matching"""
    start_time = time.time()
    
    # Config
    X_VARIABLES = ["age", "social_risk_score", "hpd_hyp", "hpd_hyc", "hpd_ast", "hpd_dia"]
    TREATMENT_VAR = "grp_binary"
    OUTCOME_VAR = "social_risk_score"
    MATCHING_RATIO = 2
    N_NEIGHBORS = 25
    CALIPER_SD = 0.25
    BATCH_SIZE = 1000  # Added batch size parameter
    
    # Create subsets based on exact matching criteria
    list_data_frame = create_subset(df, exact_match)
    
    # Initialize results
    all_results = pd.DataFrame()
    all_matching_maps = {}
    
    # Process each subset
    for i, dfx in enumerate(list_data_frame):
        if len(dfx) > 10:
            subset_description = f"subset_{i}"
            if exact_match:
                match_values = dfx[exact_match].iloc[0].values
                subset_description = "_".join(f"{col}_{val}" for col, val in zip(exact_match, match_values))
            
            matched_df, matching_map = run_analysis(
                dfx, subset_description, X_VARIABLES, TREATMENT_VAR, 
                OUTCOME_VAR, MATCHING_RATIO, N_NEIGHBORS, CALIPER_SD, BATCH_SIZE
            )
            
            if not matched_df.empty:
                all_results = pd.concat([all_results, matched_df])
                all_matching_maps.update(matching_map)
    
    print(f"\nTotal PSM function execution time: {time.time() - start_time:.2f} seconds")
    return all_results, all_matching_maps

# Usage example
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    load_start = time.time()
    df = pd.read_csv("marketing_test_case.csv")
    print(f"Data loading time: {time.time() - load_start:.2f} seconds")
    
    # Run matching
    matched_results, matching_maps = psm_function(df, exact_match=['gender_cd', 'hpd_hyp'])



















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
    
    # Convert to numpy arrays for faster operations
    treatment_pscores = treatment_df['pscore'].values
    control_pscores = control_df['pscore'].values
    treatment_indices = treatment_df.index.values
    control_indices = control_df.index.values
    
    # Sort treatment cases by propensity score
    sort_idx = np.argsort(treatment_pscores)
    treatment_pscores = treatment_pscores[sort_idx]
    treatment_indices = treatment_indices[sort_idx]
    indices = indices[sort_idx]
    
    n_treated = len(treatment_indices)
    matched_pairs = []
    used_control_mask = np.zeros(len(control_df), dtype=bool)
    matching_map = {}
    
    # Vectorized distance calculation
    control_idx_matrix = indices.reshape(n_treated, -1)
    control_pscore_matrix = control_pscores[control_idx_matrix]
    ps_diff_matrix = np.abs(treatment_pscores.reshape(-1, 1) - control_pscore_matrix)
    valid_matches_mask = ps_diff_matrix <= caliper
    
    if not with_replacement:
        for i in range(n_treated):
            valid_matches = control_idx_matrix[i][valid_matches_mask[i]]
            unused_matches = valid_matches[~used_control_mask[valid_matches]]
            
            if len(unused_matches) >= MATCHING_RATIO:
                selected_matches = unused_matches[:MATCHING_RATIO]
                used_control_mask[selected_matches] = True
                
                treated_record = treatment_df.loc[treatment_indices[i]].copy()
                treated_record['match_group'] = treatment_indices[i]
                treated_record['unit_role'] = 'treated'
                treated_record['original_index'] = treatment_indices[i]
                matched_pairs.append(treated_record)
                
                matching_map[treatment_indices[i]] = control_indices[selected_matches].tolist()
                
                for match in selected_matches:
                    control_record = control_df.loc[control_indices[match]].copy()
                    control_record['match_group'] = treatment_indices[i]
                    control_record['unit_role'] = 'control'
                    control_record['original_index'] = control_indices[match]
                    matched_pairs.append(control_record)
    else:
        for i in range(n_treated):
            valid_matches = control_idx_matrix[i][valid_matches_mask[i]]
            
            if len(valid_matches) >= MATCHING_RATIO:
                selected_matches = valid_matches[:MATCHING_RATIO]
                
                treated_record = treatment_df.loc[treatment_indices[i]].copy()
                treated_record['match_group'] = treatment_indices[i]
                treated_record['unit_role'] = 'treated'
                treated_record['original_index'] = treatment_indices[i]
                matched_pairs.append(treated_record)
                
                matching_map[treatment_indices[i]] = control_indices[selected_matches].tolist()
                
                for match in selected_matches:
                    control_record = control_df.loc[control_indices[match]].copy()
                    control_record['match_group'] = treatment_indices[i]
                    control_record['unit_role'] = 'control'
                    control_record['original_index'] = control_indices[match]
                    matched_pairs.append(control_record)
    
    unmatched_treated = n_treated - len(matching_map)
    print(f"Find matches time: {time.time() - start_time:.2f} seconds")
    return matched_pairs, unmatched_treated, matching_map

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
