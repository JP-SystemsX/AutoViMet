import yaml
from utils import (
    load_data, 
    store_complex_dict, 
    archive_config,
    random_search,
    automl_search,
    hebo_search,
    dehb_search,
    get_preprocessor,
    already_finished,
    setup_ray
    )
import models
import metrics
from collections import defaultdict
import typer
import time
from pathlib import Path
import os
import uuid

app = typer.Typer(pretty_exceptions_enable=False)

# @use_yaml_config(param_name="config")
@app.command()
def main(
        model_name: str = None, # Which model to evaluate
        eval_config_adr: str = None, # How to evaluate the model
        data_config_adr: str = None, # on which data to evaluate
        search_space_adr: str = None,
        data_id: int = None,
        search_algo: str = "random", # "automl"
        max_splits: int = 10,
):
    with open(eval_config_adr, 'r') as f:
        eval_config = yaml.safe_load(f)
    n_trials = eval_config["n_trials"]
    metric_collection = {metric: getattr(metrics, metric) for metric in eval_config["metrics"]}
    preferences: dict = eval_config["preferences"]

    search_space_hash = archive_config("results.db", config_path=search_space_adr, table_name="search_spaces", extras={"model": model_name})
    data_config_hash = archive_config("results.db", config_path=data_config_adr, table_name="data_configs")

    # Check if result already exists --> If so abord
    if already_finished(data_id=data_id, search_space_hash=search_space_hash, data_config_hash=data_config_hash, search_algo=search_algo):
        print("Already Done!")
        return
    
    if search_algo == "automl":
        setup_ray()

    # Evaluate best Model (Time Series Cross Validation)
    data_loader = load_data(data_config_adr, id=data_id) 
    results = defaultdict(list)
    best_configs = []
    search_ids = []
    i = 0
    experiment_id = str(uuid.uuid4()) 
    any_succeded = False # Did anymodel get trained successfully?
    for X_train, y_train, X_test, y_test in data_loader:
        if i >= max_splits: # limit number of evals to max 10 for feasability reasons
            break
        # Repeat HPO for every split (only fair because AG can (or must) re-optimize as well)
        model = None
        search_start = time.time()
        match search_algo:
            case "random":
                best_config, search_id = random_search(
                    X_train=X_train,
                    y_train=y_train,
                    search_space_adr=search_space_adr,
                    data_config_hash=data_config_hash,
                    metric_collection=metric_collection,
                    model_name=model_name,
                    n_trials=n_trials,
                    preferences=preferences,
                    data_id=data_id,
                    fold=i,
                    experiment_id=experiment_id
                )
            case "automl":
                model, search_id, best_config= automl_search(
                    X_train=X_train,
                    y_train=y_train,
                    model_config_adr=search_space_adr,
                    model_name=model_name,
                    preferences=preferences,
                )
                trained = True
            case "DEHB":
                best_config, search_id = dehb_search(
                    X_train=X_train,
                    y_train=y_train,
                    search_space_adr=search_space_adr,
                    data_config_hash=data_config_hash,
                    metric_collection=metric_collection,
                    model_name=model_name,
                    n_trials=n_trials,
                    preferences=preferences,
                    data_id=data_id,
                    fold=i,
                    n_workers=4, #TODO make accesible via CLI
                    mode="DEHB",
                    experiment_id=experiment_id
                )
            case "DE":
                best_config, search_id = dehb_search(
                    X_train=X_train,
                    y_train=y_train,
                    search_space_adr=search_space_adr,
                    data_config_hash=data_config_hash,
                    metric_collection=metric_collection,
                    model_name=model_name,
                    n_trials=n_trials,
                    preferences=preferences,
                    data_id=data_id,
                    fold=i,
                    n_workers=4, #TODO make accesible via CLI
                    mode="DE",
                    experiment_id=experiment_id
                )
            case "HEBO":
                best_config, search_id = hebo_search(
                    X_train=X_train,
                    y_train=y_train,
                    search_space_adr=search_space_adr,
                    data_config_hash=data_config_hash,
                    metric_collection=metric_collection,
                    model_name=model_name,
                    n_trials=n_trials,
                    preferences=preferences,
                    data_id=data_id,
                    fold=i,
                    experiment_id=experiment_id
                )
            case _:
                raise NotImplementedError(f"Search Algorithm {search_algo} not implemented.")
        i += 1
        search_duration = time.time() - search_start
        results["search_durations"].append(search_duration)
        if model is None:
            # preprocess dataset
            feature_generator = get_preprocessor(best_config)
            model = getattr(models, model_name)(**best_config)

            X_train = feature_generator.fit_transform(X=X_train, y=y_train)
            X_test = feature_generator.transform(X_test)
            try:
                # Train Model on full train set
                model.train(X_train, y_train)
                trained = True
            except Exception as e:
                print(f"Training failed with error: {e}. Continuing without training.")
                trained = False
        # Evaluate Model on test set
        prediction_start_time = time.time()
        try:
            assert trained, "Model training failed, cannot predict."
            prediction = model.predict(X_test)
            prediction_duration = time.time() - prediction_start_time
            results["prediction_time"].append(prediction_duration)
            for metric_name, metric in metric_collection.items():
                results[metric_name].append(metric(y_test, prediction))
            any_succeded = True
        except Exception as e:
            print(f"Prediction failed with error: {e}. Filling with Nones.")
            results["prediction_time"].append(None)
            for metric_name in metric_collection.keys():
                results[metric_name].append(None)
        results["train_samples"].append(len(X_train))
        results["test_samples"].append(len(X_test))
        best_configs.append(best_config)
        search_ids.append(search_id)
    if not any_succeded:
        print("No model was trained successfully on any split. Aborting result storage.")
        return
    # Commit Raw Results to DB
    store_complex_dict(
        {
            "model": str(model_name),
            "search_space_name": Path(search_space_adr).stem,
            "search_ids": search_ids,
            "experiment_id": experiment_id,
            "search_space_hash": search_space_hash,
            "data_config_hash": data_config_hash,
            "data_id": data_id,
            "best_configs": best_configs,
            "search_algo": search_algo,
            "timestamp": time.time(),
            **results
        }, 
        database_path="results.db", 
        table_name="results"
    )


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    app()