import yaml
from utils import (
    get_configspace, 
    load_data, 
    make_dict_storable, 
    get_hardware_resources, 
    store_complex_dict, 
    hash_dict, 
    create_sqlite_table_from_dict,
    archive_config
    )
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
import models
import metrics
import time
from tqdm.auto import tqdm
import pandas as pd
import sqlite3
from collections import defaultdict
import uuid
import typer
from typer_config import use_yaml_config

app = typer.Typer(pretty_exceptions_enable=False)

# @use_yaml_config(param_name="config")
@app.command()
def main(
        model_name: str = None, # Which model to evaluate
        eval_config_adr: str = None, # How to evaluate the model
        data_config_adr: str = None, # on which data to evaluate
        search_space_adr: str = None,
        data_id: int = None
):
    with open(eval_config_adr, 'r') as f:
        eval_config = yaml.safe_load(f)
    n_trials = eval_config["n_trials"]
    metric_collection = {metric: getattr(metrics, metric) for metric in eval_config["metrics"]}
    preferences: dict = eval_config["preferences"]

    search_space_hash = archive_config("results.db", config_path=search_space_adr, table_name="search_spaces", extras={"model": model_name})
    data_config_hash = archive_config("results.db", config_path=data_config_adr, table_name="data_configs")
    search_id = str(uuid.uuid4())

    # load dataset
    data_loader = load_data(data_config_adr, id=data_id) 
    X_train, y_train, _, _ = next(data_loader)
    validation_split = len(y_train) // 10
    X_val, y_val = X_train[-validation_split:], y_train[-validation_split:]
    X_train, y_train = X_train[:-validation_split], y_train[:-validation_split]

    # preprocess dataset
    feature_generator = AutoMLPipelineFeatureGenerator()
    X_train = feature_generator.fit_transform(X=X_train, y=y_train)
    X_val = feature_generator.transform(X_val)

    # Optimize Model
    cs = get_configspace(search_space_adr)
    configs = cs.sample_configuration(n_trials) 
    for config in tqdm(configs):
        # Train Model
        model = getattr(models, model_name)(**dict(config))

        start = time.time()
        model.train(X_train, y_train)
        train_duration = time.time() - start
        # Evaluate Model
        start = time.time()
        prediction = model.predict(X_val)
        inference_duration = ((time.time() - start) / len(y_val)) * 1000  
        results = {
            "model": str(model_name),
            "search_id": search_id,
            "search_space_hash": search_space_hash,
            "data_config_hash": data_config_hash,
            "data_id": data_id,
            "config": dict(config),
            "train_duration(s)": train_duration,
            "inference_duration(ms/sample)": inference_duration,
            **get_hardware_resources()
        }
        for metric_name, metric in metric_collection.items():
            results[metric_name] = metric(y_val, prediction)
        # Write Results to DB 
        store_complex_dict(results, database_path="results.db", table_name="trials")

    # Load Best Config
    conn = sqlite3.connect("results.db")
    results = pd.read_sql(
        "SELECT * FROM trials where search_space_hash = ? and search_id = ? and data_config_hash = ? and data_id = ?", 
        conn, 
        params=(search_space_hash, search_id, data_config_hash, data_id)
        )
    conn.close()
    assert len(results) == n_trials, f"Expected {n_trials} results but got {len(results)}. Something went wrong during the evaluation."
    results["preference_score"] = sum(results[metric] * weight for metric, weight in preferences.items())
    incumbent = results.sort_values("preference_score", ascending=True).iloc[0]
    best_config = yaml.safe_load(incumbent["config"].values[0])
    print("Best Config found:", best_config)


    # TODO Evaluate best Model (Time Series Cross Validation)
    data_loader = load_data(data_config_adr, id=data_id) 
    results = defaultdict(list)
    for X_train, y_train, X_test, y_test in data_loader:
        model = getattr(models, model_name)(**best_config)
        # preprocess dataset
        feature_generator = AutoMLPipelineFeatureGenerator()
        X_train = feature_generator.fit_transform(X=X_train, y=y_train)
        X_test = feature_generator.transform(X_test)
        # Train Model on full train set
        model.train(X_train, y_train)
        # Evaluate Model on test set
        prediction = model.predict(X_test)
        for metric_name, metric in metric_collection.items():
            results[metric_name].append(metric(y_test, prediction))
        
    # Commit Raw Results to DB
    store_complex_dict(
        {
            "model": str(model_name),
            "search_id": search_id,
            "search_space_hash": search_space_hash,
            "data_config_hash": data_config_hash,
            "data_id": data_id,
            "config": best_config,
            **results
        }, 
        database_path="results.db", 
        table_name="results"
    )

    # TODO Evaluate best Model (k-fold Cross Validation)
    # Do they agree?


if __name__ == "__main__":
    app()