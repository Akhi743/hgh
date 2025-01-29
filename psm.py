import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def run_psm_analysis(data_path="Dataset/lalonde.csv"):
    """
    Run simple PSM analysis on LaLonde dataset
    """
    # Read data
    df = pd.read_csv(data_path)
    
    # Create features for PSM
    features = ['age', 'educ', 'married', 'nodegree', 're74', 're75']
    race_dummies = pd.get_dummies(df['race'], prefix='race')
    df = pd.concat([df, race_dummies], axis=1)
    features.extend(race_dummies.columns)
    
    # Prepare data
    X = df[features]
    T = df['treat']
    Y = df['re78']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Estimate propensity scores
    ps_model = LogisticRegression(random_state=42)
    ps_model.fit(X_scaled, T)
    ps_scores = ps_model.predict_proba(X_scaled)[:, 1]
    
    # Add propensity scores to dataframe
    df['ps_score'] = ps_scores
    
    # Match treated units to control units
    treated = df[df['treat'] == 1].copy()
    control = df[df['treat'] == 0].copy()
    
    matches = []
    caliper = 0.2 * np.std(ps_scores)  # Standard caliper
    
    for idx, treated_unit in treated.iterrows():
        potential_matches = control[abs(control['ps_score'] - treated_unit['ps_score']) < caliper]
        if not potential_matches.empty:
            best_match_idx = abs(potential_matches['ps_score'] - treated_unit['ps_score']).idxmin()
            matches.append((idx, best_match_idx))
            control = control.drop(best_match_idx)
    
    # Create matched samples
    treated_matched = df.loc[[m[0] for m in matches]]
    control_matched = df.loc[[m[1] for m in matches]]
    
    # Calculate treatment effects
    att = np.mean(treated_matched['re78'] - control_matched['re78'])
    se = np.std(treated_matched['re78'] - control_matched['re78']) / np.sqrt(len(matches))
    
    # Calculate ATE
    ate = np.mean(treated_matched['re78']) - np.mean(control_matched['re78'])
    
    print("\nPropensity Score Matching Results")
    print("-" * 50)
    print(f"Number of treated units: {len(treated)}")
    print(f"Number of matched pairs: {len(matches)}")
    print(f"Matching rate: {len(matches)/len(treated):.2%}")
    
    print("\nTreatment Effects:")
    print(f"ATE: ${ate:.2f}")
    print(f"ATT: ${att:.2f}")
    print(f"Standard Error: ${se:.2f}")
    print(f"95% CI: (${att - 1.96*se:.2f}, ${att + 1.96*se:.2f})")
    print("-" * 50)

if __name__ == "__main__":
    run_psm_analysis()
