from pathlib import Path   
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


original_benchmark_path = Path("../data/benchmarks/original/TODO")


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
    selected_indices = greedy_coreset(
        X=meta_data_df[['num_rows', 'num_cols', 'std_target', 'mean_target', 'xgb_r2', 'lr_r2']].to_numpy(),
        k=50,
        random_state=42
    )

    selected_meta_data = meta_data_df.iloc[selected_indices]
    # Copy resulting set over to new directory
    for file in selected_meta_data['data_set_path']:
        dest_path = Path("../data/benchmarks/balanced/") / original_benchmark_path.name / file.name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file, dest_path)
     
    # Anonimize Sensitive data by min max scalling
    scaler = MinMaxScaler()
    meta_data_df_scaled = meta_data_df.copy()
    meta_data_df_scaled[['num_rows', 'num_cols', 'std_target', 'mean_target', 'xgb_r2', 'lr_r2']] = scaler.fit_transform(
        meta_data_df[['num_rows', 'num_cols', 'std_target', 'mean_target', 'xgb_r2', 'lr_r2']]
    )
            
    selected_meta_data = meta_data_df.iloc[selected_indices]

    # TODO Plot Each pair Distribution old meta data distribution and add mark selected samples  
    plot_pairs = [
        ("num_rows", "num_cols"),
        ("std_target", "mean_target"),
        ("xgb_r2", "lr_r2"),
        ("num_rows", "xgb_r2"),
        ("num_rows", "lr_r2"),
        ("num_rows", "std_target"),
        ("num_cols", "std_target"),
    ]

    for x_col, y_col in plot_pairs:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=meta_data_df, x=x_col, y=y_col, label='Original', alpha=0.5)
        sns.scatterplot(data=selected_meta_data, x=x_col, y=y_col, label='Selected Coreset', color='red', s=100)
        plt.title(f'Scatter Plot of {x_col} vs {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_path = Path(f"../analysis/cache/analysis/benchmarks/plots/{x_col}_vs_{y_col}_coreset_selection.pdf")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()
