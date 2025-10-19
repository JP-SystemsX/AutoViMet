import yaml
from utils import (
    load_data, 
    store_complex_dict, 
    archive_config,
    random_search,
    automl_search,
    )
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
import models
import metrics
from collections import defaultdict
import typer
import time


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
):
    with open(eval_config_adr, 'r') as f:
        eval_config = yaml.safe_load(f)
    n_trials = eval_config["n_trials"]
    metric_collection = {metric: getattr(metrics, metric) for metric in eval_config["metrics"]}
    preferences: dict = eval_config["preferences"]

    search_space_hash = archive_config("results.db", config_path=search_space_adr, table_name="search_spaces", extras={"model": model_name})
    data_config_hash = archive_config("results.db", config_path=data_config_adr, table_name="data_configs")


    # Evaluate best Model (Time Series Cross Validation)
    data_loader = load_data(data_config_adr, id=data_id) 
    results = defaultdict(list)
    for X_train, y_train, X_test, y_test in data_loader:
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
                    data_id=data_id
                )
            case "automl":
                model, search_id, best_config= automl_search(
                    X_train=X_train,
                    y_train=y_train,
                    model_config_adr=search_space_adr,
                    model_name=model_name,
                    preferences=preferences,
                )
            case _:
                raise NotImplementedError(f"Search Algorithm {search_algo} not implemented.")
        search_duration = time.time() - search_start
        results["search_durations"].append(search_duration)
        if model is None:
            model = getattr(models, model_name)(**best_config)
            # preprocess dataset
            feature_generator = AutoMLPipelineFeatureGenerator()
            X_train = feature_generator.fit_transform(X=X_train, y=y_train)
            X_test = feature_generator.transform(X_test)
            # Train Model on full train set
            model.train(X_train, y_train)
        # Evaluate Model on test set
        prediction_start_time = time.time()
        prediction = model.predict(X_test)
        prediction_duration = time.time() - prediction_start_time
        results["prediction_time"].append(prediction_duration)
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


if __name__ == "__main__":
    app()