import ConfigSpace 
from pathlib import Path
import yaml
import openml
import datetime
import json
from autogluon.tabular import FeatureMetadata
import psutil
import platform
import hashlib
import sqlite3
import subprocess
import uuid
from time import sleep
from warnings import warn
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
from ConfigSpace import Configuration
import models as models
import metrics as metrics
import time
import pandas as pd
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.feature_extraction.text import CountVectorizer
import auto_models 
from math import inf
import pickle
from functools import cache, partial
from ConfigSpace import (
    Categorical, 
    Float, 
    Integer, 
    Constant, 
    UniformFloatHyperparameter, 
    UniformIntegerHyperparameter, 
    CategoricalHyperparameter
)
from autogluon.features.generators import PipelineFeatureGenerator, FillNaFeatureGenerator



def get_configspace(path: str | Path) -> ConfigSpace.ConfigurationSpace:
    """Load a ConfigSpace from a given file path."""
    cs = ConfigSpace.ConfigurationSpace.from_yaml(path)

    # Preprocessing Search Space
    # cs.add(Constant("enable_numeric_features", value=True))
    cs.add(Categorical("enable_categorical_features", [True, False],default=True))
    cs.add(Categorical("enable_datetime_features", [True, False],default=True))
    cs.add(Categorical("enable_text_special_features", [True, False],default=True))
    cs.add(Categorical("enable_text_ngram_features", [True, False],default=True))
    cs.add(Categorical("enable_raw_text_features", [True, False],default=False))
    cs.add(Categorical("allow_nans", [True, False],default=False))
    cs.add(Categorical("prepro_analyzer", ["word", "char", "char_wb"], default="word"))
    cs.add(Float("prepro_min_df", (0.005, 1.0), log=True, default=0.2))
    cs.add(Integer("prepro_max_features", (5, 20000), log=True, default=10000))

    return cs


@cache
def get_dataset(data_id: int, id: int):
    """HPC leads to hickups in standard caching procedure -> Custom Caching"""
    cache = Path("./cache")
    cache.mkdir(exist_ok=True, parents=True)
    suite_cache = Path(f"./cache/{data_id}.pkl")
    # if suite_cache.exists():
    try:
        with open(suite_cache, "rb") as f:
            suite = pickle.load(f)
    except: # To catch none exist AND version change
        suite = openml.study.get_suite(data_id)  # Get a curated list of tasks for classification
        with open(suite_cache, "wb") as f:
            pickle.dump(suite, f)
    task_id = suite.tasks[id]
    task_cache = Path(f"./cache/{data_id}_{task_id}.pkl")
    try:
        with open(task_cache, "rb") as f:
            task = pickle.load(f)
    except:
        task = openml.tasks.get_task(task_id, download_splits=True)
        with open(task_cache, "wb") as f:
            pickle.dump(task, f)
    ds_cache = Path(f"./cache/{data_id}_{task_id}_ds.pkl")
    try:
        with open(ds_cache, "rb") as f:
            dataset = pickle.load(f)
    except:
        dataset = task.get_dataset(force_refresh_cache=True)
        with open(ds_cache, "wb") as f:
            pickle.dump(dataset, f)
    return dataset, task



def load_data(data_config_adr: str, id: int = None):
    """Load data based on the provided configuration address."""
    with open(data_config_adr, 'r') as f:
        data_config = yaml.safe_load(f)

    if data_config["origin"] == "OpenML":
        if data_config["type"] == "Benchmark":
            assert id is not None, "ID must be provided for OpenML Benchmark datasets."
            dataset, task = get_dataset(data_id=data_config["data_id"], id=id)
            X, y, _, _ = dataset.get_data(target=task.target_name)
            repeats, folds, samples = task.get_split_dimensions()
            for repeat in range(repeats):
                for fold in range(folds):
                    for sample in range(samples):
                        train_indices, test_indices = task.get_train_test_split_indices(repeat=repeat, fold=fold, sample=sample)
                        yield X.iloc[train_indices], y.iloc[train_indices], X.iloc[test_indices], y.iloc[test_indices]
    elif data_config["origin"] == "Custom":
        if data_config["type"] == "Benchmark":
            # Get all Files in data directory
            data_files = list(Path(data_config["data_directory"]).glob("*.parquet"))
            # Order by Alphabet (to always have the same order)
            data_files = sorted(data_files)
            data_file = data_files[id]
            df = pd.read_parquet(data_file)

            if data_config["cv_strategy"]["name"] == "TimeSeriesSplit":
                # Sort by time
                df = df.sort_values(by=data_config["cv_strategy"]["order_by"])
                n_splits = data_config["cv_strategy"]["n_splits"]
                fold_size = len(df) // (n_splits + 1)
                for fold in range(n_splits):
                    train_end = fold_size * (fold + 1)
                    test_end = fold_size * (fold + 2)
                    train_data = df.iloc[:train_end]
                    test_data = df.iloc[train_end:test_end]
                    X_train = train_data.drop(columns=[data_config["target_column"]])
                    y_train = train_data[data_config["target_column"]]
                    X_test = test_data.drop(columns=[data_config["target_column"]])
                    y_test = test_data[data_config["target_column"]]
                    yield X_train, y_train, X_test, y_test
            else:
                # TODO Also Support Random K-Fold CV -- (k-fold Cross Validation vs time series split) --> Do they agree?
                raise NotImplementedError(f"CV Strategy {data_config['cv_strategy']['name']} not implemented yet.")
        else:
            raise NotImplementedError(f"Data config type {data_config['type']} not implemented yet.")



def make_dict_storable(advanced_dictionary: dict)->dict:
    """
    Takes in a dict with advanced values like datetime, dicts, lists, etc. 
    and converts it into a dict that can be stored in an sqlite database meaning strings, numbers, etc.

    Args:
        advanced_dictionary (dict): dict with potentially complex datatypes

    Returns:
        dict: A dictionary with only simple datatypes
    """
    simple_dict = {}
    for key, value in advanced_dictionary.items():
        if isinstance(value, (int, float, str)):
            pass
        elif isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, (bool, np.bool_)):
            value = 1 if value else 0
        elif isinstance(value, (bytes, Path, list, FeatureMetadata)) or value is None:
            value = str(value)
        elif isinstance(value, (datetime.datetime, datetime.date)):
            value = value.strftime("%d-%m-%Y %H:%M:%S")
        elif isinstance(value, dict):
            value = json.dumps(make_dict_storable(value))
        #elif isinstance(value, str):
        else:
            raise NotImplementedError(f"dtype {type(value)} is currently not storable, but can likely be easily added in make_dict_storable()")
        simple_dict[key] = value

    return simple_dict


def hash_dict(d: dict) -> str:
    """Coverts Simple Dict into hash 

    Args:
        d (dict): Dict that shall be hashed

    Returns:
        str: hashcode
    """
    # Not sure how more complex types get hashed (often represented as some kind of RANDOM id)
    # Hence Simplify first  
    d = make_dict_storable(d)  
    # Make Consistent ~ Convert dict to a sorted tuple 
    dict_tuple = tuple(sorted(d.items()))
    # Convert to binaries
    dict_string = str(dict_tuple).encode()
    # Hash binaries
    hash_object = hashlib.sha256(dict_string)
    
    return hash_object.hexdigest()

def store_complex_dict(d:dict, database_path: str | Path, table_name: str) -> str:
    storable_dictionary = make_dict_storable(d)
    hash = hash_dict(storable_dictionary)

    create_sqlite_table_from_dict(
        database_path=database_path,
        table_name=table_name,
        primary_keys=["hash_key"],#, "data_config_hash"],
        data_dict={
            # Both Hashs build Primary Key
            "hash_key": hash, 
            **storable_dictionary
        }
    ) 
    return hash


def get_hardware_resources() -> dict:
    hw_dict = {
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "architecture": str(platform.architecture()),
        "cpu_count_available": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3),
        "commit_hash": get_git_commit_hash(),
        "uncommited_changes": uncommited_changes_exist(),
    }
    return hw_dict


def create_sqlite_table_from_dict(database_path: str | Path, table_name: str, data_dict: dict, primary_keys=None):
    """
    Create a SQLite database and table from a dictionary.

    Args:
        database_path (str): Path to the SQLite database.
        table_name (str): Name of the table to create.
        data_dict (dict): Dictionary where each key is a column name and its value determines the column type.
    """
    # Infer column types based on dictionary values
    def infer_sql_type(value):
        if isinstance(value, int):
            return "INTEGER"
        elif isinstance(value, float):
            return "REAL"
        elif isinstance(value, str):
            return "TEXT"
        elif isinstance(value, bytes):
            return "BLOB"
        else:
            return "TEXT"  # Default to TEXT if the type is unknown
    
    if isinstance(database_path, str):
        database_path = Path(database_path)

    # Connect to SQLite database (create if it doesn't exist)
    database_path.parent.mkdir(parents=True, exist_ok=True) # Ensure the directory exists

    if (database_path.exists() and database_path.is_dir()) or ((not database_path.exists()) and database_path.suffix == ""):
        database_path.mkdir(exist_ok=True, parents=True)
        # Cache Transaction First
        transaction_id = str(uuid.uuid4())
        transaction_adr = database_path / f"{transaction_id}.yml"
        transaction_data = {
            "table_name": table_name,
            "data_dict": data_dict,
            "primary_keys": primary_keys
        }
        with open(transaction_adr, "w+") as file:
            yaml.safe_dump(transaction_data, file)

        return

    # Build the CREATE TABLE statement
    columns = [f"{key} {infer_sql_type(value)}" for key, value in data_dict.items()]# if key not in primary_keys]
    if not primary_keys:
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {', '.join(columns)}
        );
        """
    else:
        #primary_columns = [f"{key} {infer_sql_type(data_dict[key])} PRIMARY KEY" for key in primary_keys]
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join(columns)},
            PRIMARY KEY ({', '.join(primary_keys)})
        );
        """

    # Insert the dictionary as a row into the table
    columns_str = ', '.join(data_dict.keys())
    placeholders = ', '.join(['?' for _ in data_dict.values()])
    insert_sql = f"INSERT OR IGNORE INTO {table_name} ({columns_str}) VALUES ({placeholders});"


    # Active Waiting to connect
    i = 0
    while True:
        try:
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            # Execute the CREATE TABLE statement
            cursor.execute(create_table_sql)
            cursor.execute(insert_sql, tuple(data_dict.values()))

            conn.commit()
            conn.close()
            break
        except Exception as e:
            print(e, i)
            i += 1
            sleep(0.1)
            if i > 100:
                break


def get_git_commit_hash()->str:
    """Get The Hash to identify the executed Code Version Later on

    Returns:
        str: The hashcode is returned as string
    """
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL)
        return commit_hash.strip().decode('utf-8')
    except subprocess.CalledProcessError:
        warn("No Git repository found or Git not installed")
        return "Not Found"

    
def uncommited_changes_exist()->bool:
    try:
        result = subprocess.run(["git", "diff-index", "--quiet", "HEAD", "--"], capture_output=True)
        return result.returncode != 0  # Non-zero return code indicates changes
    except FileNotFoundError:
        print("Git is not installed or not available in PATH.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    

def archive_config(database_path: Path, config_path: Path, table_name: str, extras: dict = None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config_hash = hash_dict(config)
    create_sqlite_table_from_dict(
        database_path=database_path,
        table_name=table_name,
        primary_keys=["hash_key"],#, "data_config_hash"],
        data_dict={
            # Both Hashs build Primary Key
            "hash_key": config_hash, 
            "config": json.dumps(make_dict_storable(config)),
            **(extras if extras else {})
        }
    ) 
    return config_hash

# =========== #
# HPO Methods #
# =========== #


def train(
        config: Configuration,
        fidelity: float,
        X_train_: pd.DataFrame,
        y_train_: pd.Series,
        X_val_: pd.DataFrame,
        y_val_: pd.Series,
        metric_collection: dict,
        model_name: str,
        search_id: str,
        search_space_hash: str,
        data_config_hash: str,
        data_id: int,
        fold: int,
        preferences: dict,
):
    config = dict(config)
    config_ = deepcopy(config)

    # Setup Preprocessor + Remove related HPs from config
    feature_generator = get_preprocessor(config)
    
    X_train = feature_generator.fit_transform(X=X_train_, y=y_train_)
    X_val = feature_generator.transform(X_val_)

    # Make copy to avoid operating in place
    y_train = deepcopy(y_train_)
    y_val = deepcopy(y_val_)

    if fidelity < 1.0:
        idx = X_train.sample(frac=fidelity, random_state=123).index
        y_train = y_train.loc[idx]
        X_train = X_train.loc[idx]
    model = getattr(models, model_name)(**config)
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
        "fold": fold,
        "fidelity": fidelity,
        "config": config_,
        "train_duration": train_duration,
        "inference_duration": inference_duration,
        "timestamp": time.time(),
        **get_hardware_resources()
    }
    for metric_name, metric in metric_collection.items():
        results[metric_name] = metric(y_val, prediction)

    # Write Results to DB 
    store_complex_dict(results, database_path="results.db", table_name="trials")

    preference_score = sum(results[metric] * weight for metric, weight in preferences.items())

    return {
        "fitness": preference_score,  # DE/DEHB minimizes
        "cost": train_duration+inference_duration,
        "info": {
            "test_score": str(results),
            "fidelity": fidelity
        }
    }


def dehb_search(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        search_space_adr: str, 
        data_config_hash: str, 
        metric_collection: dict,
        model_name: str, 
        n_trials: int, 
        preferences: dict,
        data_id: int = None,
        fold: int = 0,
        n_workers=4, #TODO send through
        mode = "DEHB" #or 'DE'
        ):
    from dehb import DEHB, DE
    assert len(X_train) == len(y_train), "X_train and y_train must have the same length."
    # Split Data into Train and Validation
    validation_split = len(y_train) // 10
    X_val_, y_val_ = X_train[-validation_split:], y_train[-validation_split:]
    X_train_, y_train_ = X_train[:-validation_split], y_train[:-validation_split]

    # Optimize Model
    search_space_hash = archive_config("results.db", config_path=search_space_adr, table_name="search_spaces", extras={"model": model_name}) # Regenerate
    cs = get_configspace(search_space_adr)
    search_id = str(uuid.uuid4())               
    print(f"Evaluating {n_trials} configurations for model {model_name} on data config {data_config_hash}.")

    if mode == "DEHB":
        optimization_algo = DEHB 
    elif mode == "DE":
        optimization_algo = DE
    else:
        raise NotImplementedError(f"Optimization Algo {mode} does not exist") 

    optimizer = optimization_algo(
        f=partial(
            train,
            X_train_=X_train_,
            y_train_=y_train_,
            X_val_=X_val_,
            y_val_=y_val_,
            metric_collection=metric_collection,
            model_name=model_name,
            search_id=search_id,
            search_space_hash=search_space_hash,
            data_config_hash=data_config_hash,
            data_id=data_id,
            fold=fold,
            preferences=preferences,
        ),
        cs=cs, 
        dimensions=len(list(cs.values())), 
        min_fidelity=0.005, # Exact number is irrelevant just used to compute how many steps fit between lowest and highest fidelity
        max_fidelity=1,
        eta=4.64158883361, # Three steps from 1% to 100%
        n_workers=n_workers,
        mutation_factor=0.5,
        crossover_prob=0.5,
        )
    if mode == "DEHB":
        trajectory, runtime, history = optimizer.run(
            fevals=n_trials,
            seed=123,
        )
    elif mode == "DE":
        trajectory, runtime, history = optimizer.run(
            generations=max((n_trials // optimizer.pop_size)-1, 1), 
            fidelity=1.0,
        )

    best_config = optimizer.vector_to_configspace(optimizer.inc_config)
    best_config = dict(best_config)
    print("Best Config found:", best_config)    


    return best_config, search_id


def hebo_search(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        search_space_adr: str, 
        data_config_hash: str, 
        metric_collection: dict,
        model_name: str, 
        n_trials: int, 
        preferences: dict,
        data_id: int = None,
        fold: int = 0,
        max_walltime: int = 14400, # Abort search after 4h
        ):
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO
    def to_design_space(cs: ConfigSpace.ConfigurationSpace) -> DesignSpace:
        # Convert manually to HEBO space
        params = []
        for hp in list(cs.values()):
            if isinstance(hp, UniformFloatHyperparameter):
                params.append({'name': hp.name, 'type': 'num', 'lb': hp.lower, 'ub': hp.upper})
            elif isinstance(hp, UniformIntegerHyperparameter):
                params.append({'name': hp.name, 'type': 'int', 'lb': hp.lower, 'ub': hp.upper})
            elif isinstance(hp, CategoricalHyperparameter):
                params.append({'name': hp.name, 'type': 'cat', 'categories': hp.choices})

        space = DesignSpace().parse(params)
        return space

    assert len(X_train) == len(y_train), "X_train and y_train must have the same length."
    # Split Data into Train and Validation
    validation_split = len(y_train) // 10
    X_val_, y_val_ = X_train[-validation_split:], y_train[-validation_split:]
    X_train_, y_train_ = X_train[:-validation_split], y_train[:-validation_split]

    # Optimize Model
    search_space_hash = archive_config("results.db", config_path=search_space_adr, table_name="search_spaces", extras={"model": model_name}) # Regenerate
    cs = get_configspace(search_space_adr)
    space = to_design_space(cs)
    opt   = HEBO(space)

    search_id = str(uuid.uuid4())               
    print(f"Evaluating {n_trials} configurations for model {model_name} on data config {data_config_hash}.")
    step_size = 32
    start_time = time.time()
    for i in tqdm(range((n_trials+step_size-1)//step_size)):
        configs = opt.suggest(n_suggestions=step_size)
        fittnesses = []
        for config in configs.to_dict(orient="records"):
            if (time.time() - start_time) > max_walltime:
                break
            try:
                results = train(
                    config=config,
                    fidelity=1.0,
                    X_train_=X_train_,
                    y_train_=y_train_,
                    X_val_=X_val_,
                    y_val_=y_val_,
                    metric_collection=metric_collection,
                    model_name=model_name,
                    search_id=search_id,
                    search_space_hash=search_space_hash,
                    data_config_hash=data_config_hash,
                    data_id=data_id,
                    fold=fold,
                    preferences=preferences
                )
                fittnesses.append(results["fitness"])
            except Exception as e:
                warn(f"Config {config} failed, due to: {e}")
                fittnesses.append(inf)
        if (time.time() - start_time) > max_walltime:
            break
        opt.observe(configs, np.array(fittnesses))

    # Load Best Config
    conn = sqlite3.connect("results.db")
    results = pd.read_sql(
        "SELECT * FROM trials where search_space_hash = ? and search_id = ? and data_config_hash = ? and data_id = ?", 
        conn, 
        params=(search_space_hash, search_id, data_config_hash, data_id)
        )
    conn.close()
    # assert len(results) == n_trials, f"Expected {n_trials} results but got {len(results)}. Something went wrong during the evaluation."
    results["preference_score"] = sum(results[metric] * weight for metric, weight in preferences.items())
    incumbent = results.sort_values("preference_score", ascending=True).iloc[0]
    best_config = yaml.safe_load(incumbent["config"])
    # Type Cast back to original (e.g. we store bools as int two typecasts required to cast it back to original)
    best_config = Configuration(cs, values=best_config)
    best_config = Configuration(cs, vector=best_config._vector)
    best_config = dict(best_config)
    print("Best Config found:", best_config)    

    return best_config, search_id


def random_search(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        search_space_adr: str, 
        data_config_hash: str, 
        metric_collection: dict,
        model_name: str, 
        n_trials: int, 
        preferences: dict,
        data_id: int = None,
        fold: int = 0,
        max_walltime: int = 14400, # Abort search after 4h
        ):
    assert len(X_train) == len(y_train), "X_train and y_train must have the same length."
    # Split Data into Train and Validation
    validation_split = len(y_train) // 10
    X_val_, y_val_ = X_train[-validation_split:], y_train[-validation_split:]
    X_train_, y_train_ = X_train[:-validation_split], y_train[:-validation_split]

    # Optimize Model
    search_space_hash = archive_config("results.db", config_path=search_space_adr, table_name="search_spaces", extras={"model": model_name}) # Regenerate
    cs = get_configspace(search_space_adr)
    configs = [cs.get_default_configuration()] + cs.sample_configuration(n_trials-1) 
    search_id = str(uuid.uuid4())               
    print(f"Evaluating {n_trials} configurations for model {model_name} on data config {data_config_hash}.")
    start_time = time.time()
    for config in tqdm(configs):
        if (time.time() - start_time) > max_walltime:
            break
        try:
            train(
                config=config,
                fidelity=1.0,
                X_train_=X_train_,
                y_train_=y_train_,
                X_val_=X_val_,
                y_val_=y_val_,
                metric_collection=metric_collection,
                model_name=model_name,
                search_id=search_id,
                search_space_hash=search_space_hash,
                data_config_hash=data_config_hash,
                data_id=data_id,
                fold=fold,
                preferences=preferences
            )
        except Exception as e:
            warn(f"Config {config} failed, due to: {e}")

    # Load Best Config
    conn = sqlite3.connect("results.db")
    results = pd.read_sql(
        "SELECT * FROM trials where search_space_hash = ? and search_id = ? and data_config_hash = ? and data_id = ?", 
        conn, 
        params=(search_space_hash, search_id, data_config_hash, data_id)
        )
    conn.close()
    # assert len(results) == n_trials, f"Expected {n_trials} results but got {len(results)}. Something went wrong during the evaluation."
    results["preference_score"] = sum(results[metric] * weight for metric, weight in preferences.items())
    incumbent = results.sort_values("preference_score", ascending=True).iloc[0]
    best_config = yaml.safe_load(incumbent["config"])
    # Type Cast back to original (e.g. we store bools as int two typecasts required to cast it back to original)
    best_config = Configuration(cs, values=best_config)
    best_config = Configuration(cs, vector=best_config._vector)
    best_config = dict(best_config)
    print("Best Config found:", best_config)    

    return best_config, search_id


def automl_search(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
        model_config_adr: str = None, # alternative name for search_space_adr
        preferences: dict = None,
):
    X_train = deepcopy(X_train)
    y_train = deepcopy(y_train)
    search_id = str(uuid.uuid4())
    # Load Config
    with open(model_config_adr, 'r') as f:
        model_config = yaml.safe_load(f)
    metric = max(preferences, key=lambda k: abs(preferences[k])) # TODO Support Multi Objective Optimization
    model = getattr(auto_models, model_name)(metric=metric, config=model_config, search_id=search_id)

    model.train(X_train, y_train)

    info = model.info()
    return model, search_id, make_dict_storable(info)


# ============= #
# Preprocessing #
# ============= #

class AdvancedFillNaFeatureGenerator(FillNaFeatureGenerator):

    def __init__(self, fillna_map: dict):
        super().__init__(fillna_map=fillna_map)

    def _fit_transform(self, X: pd.DataFrame, **kwargs) :
        features = self.feature_metadata_in.get_features()
        self._fillna_feature_map = {}

        for feature in features:
            feature_raw_type = self.feature_metadata_in.get_feature_type_raw(feature)
            feature_fillna_val = self.fillna_map.get(feature_raw_type, self.fillna_default)

            if isinstance(feature_fillna_val, str):
                col = X[feature]
                if feature_fillna_val == "mean" and np.issubdtype(col.dtype, np.number):
                    feature_fillna_val = col.mean()
                elif feature_fillna_val == "median" and np.issubdtype(col.dtype, np.number):
                    feature_fillna_val = col.median()
                elif feature_fillna_val in {"mode", "most"}:
                    feature_fillna_val = col.mode(dropna=True).iloc[0] if not col.mode(dropna=True).empty else np.nan

            if feature_fillna_val is not np.nan:
                self._fillna_feature_map[feature] = feature_fillna_val

        return self._transform(X), self.feature_metadata_in.type_group_map_special


def get_preprocessor(config: dict):
    # preprocess dataset
    vectorizer = CountVectorizer( # only when enable_text_ngram_features
            min_df=config.pop("prepro_min_df", 0.2),
            ngram_range=(1, 3), # Uni, Bi, and Tri-grams
            max_features=config.pop("prepro_max_features", 10000), 
            dtype=np.uint8, 
            analyzer=config.pop("prepro_analyzer", "word") 
    )

    if config.pop("allow_nans", False):
        imputer = None
    else:
        imputer = AdvancedFillNaFeatureGenerator(
                                            fillna_map={
                                                "int": "median",
                                                "float": "mean",
                                                "category": "most",
                                                "object": "mode",
                                                "datetime": pd.Timestamp("1970-01-01"),
                                            }
                                        )

    feature_generator = AutoMLPipelineFeatureGenerator(
        enable_numeric_features=True, # config.pop("enable_numeric_features", True), # This always true
        enable_categorical_features=config.pop("enable_categorical_features", True),
        enable_datetime_features=config.pop("enable_datetime_features", True),
        enable_text_special_features=config.pop("enable_text_special_features", True),
        enable_text_ngram_features=config.pop("enable_text_ngram_features", True),
        enable_raw_text_features=config.pop("enable_raw_text_features", False),
        vectorizer=vectorizer,
        post_generators=imputer,
    )


    return feature_generator