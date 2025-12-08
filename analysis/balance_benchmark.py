from pathlib import Path   
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

original_benchmark_path = Path("../data/benchmarks/original")


def greedy_coreset(X: np.ndarray, k: int, random_state: int = 0):
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]

    if k <= 0 or k > n_samples:
        raise ValueError("k must be in [1, n_samples].")

    # Start from one random point
    first_idx = rng.integers(0, n_samples)
    selected = [first_idx]

    # Pre-compute pairwise squared distances for efficiency
    # (n_samples, n_samples)
    dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)

    # For each point, distance to nearest selected point
    min_dist_to_selected = dists[:, first_idx]

    # Greedily add the farthest point from current selected set
    for _ in range(1, k):
        next_idx = np.argmax(min_dist_to_selected)
        selected.append(next_idx)
        # Update distances: min distance to any selected point
        min_dist_to_selected = np.minimum(min_dist_to_selected, dists[:, next_idx])

    return np.array(selected, dtype=int)




if __name__ == "__main__":
    meta_data_entries = []

    # Loop over all datasets in directory
    for dataset_dir in original_benchmark_path.iterdir():
        if dataset_dir.is_file():
            df = pd.read_parquet(dataset_dir)

            # Drop Empty Columns
            df = df.dropna(axis=1, how="all")

            # Analyze dataset meta data
            entry = {
                "data_set_path": dataset_dir,
                "num_rows": df.shape[0],
                "num_cols": df.shape[1],
                "std_target": df['target'].std(),
                "mean_target": df['target'].mean(),
            }

            train = df.sample(frac=0.8, random_state=42)
            test = df.drop(train.index)

            # Evaluate XGBoost Regressor
            xgb = XGBRegressor()
            xgb.fit(train.drop(columns=['target']), train['target'])
            xgb_score = xgb.score(test.drop(columns=['target']), test['target'])
            entry["xgb_r2"] = xgb_score

            # Evaluate Linear Regression
            lr = LinearRegression()
            lr.fit(train.drop(columns=['target']), train['target'])
            lr_score = lr.score(test.drop(columns=['target']), test['target'])
            entry["lr_r2"] = lr_score

            meta_data_entries.append(entry)

    # Create Dataframe Directory path mapping to meta data
    meta_data_df = pd.DataFrame(meta_data_entries)
    # Save meta data to CSV
    meta_data_df.to_csv("../analysis/cache/analysis/balance_benchmark_meta_data.csv", index=False)
    # TODO Apply Core Set
     

    # TODO Copy resulting set over to new directory
    # TODO Plot Each pair Distribution old meta data distribution and add mark selected samples  