import os
from autogluon.tabular import TabularPredictor, TabularDataset
from models import BaseModel
from autogluon.tabular.configs.presets_configs import tabular_presets_dict
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from autogluon.core.models.ensemble.fold_fitting_strategy import SequentialLocalFoldFittingStrategy


class AutoModel(BaseModel):
    def __init__(self, metric, config, search_id,**kwargs):
        pass

    def info(self) -> dict:
        return {"model": "Model"}

class AutoGluon(AutoModel):
    sklearn_metric_to_ag_metric = {
        "r2_score": "r2",
        "mean_absolute_error": "mean_absolute_error",
        "mean_squared_error": "mean_squared_error",
        "mean_absolute_percentage_error": "mean_absolute_percentage_error",
        "root_mean_squared_error": "root_mean_squared_error"
    }
    def __init__(self, metric, config, search_id, **kwargs):
        self.fit_kwargs = config.get("fit_kwargs", {})
        init_kwargs = config.get("init_kwargs", {})
        self.model = TabularPredictor("label", problem_type="regression", path=f"./tmp/AG/{search_id}", eval_metric=self.sklearn_metric_to_ag_metric[metric], **{**kwargs, **init_kwargs})
        self.cpu_count = 2

    def train(self, X, y):
        X["label"] = y
        df = TabularDataset(X)

        # For Bug with CPU count on GloFo Cluster which makes num_cpus ineffective
        preset = self.fit_kwargs["presets"]
        preset_hp = tabular_presets_dict[preset]
        hp_conf_name = preset_hp.get("hyperparameters", "default")
        model_hp: dict = get_hyperparameter_config(hp_conf_name)
        # Adjust HP to work with GloFos HPC (a bug took more cpus than allocated)
        for hp_name in ['RF', 'XT', 'KNN', 'NN_TORCH', 'GBM', 'XGB', 'FASTAI']:
            # If the model isn't included in the preset ignore it
            if hp_name not in model_hp:
                continue
            # Set use_child_off of the remaining models to false
            if isinstance(model_hp, list):
                old_hps = model_hp[hp_name][:]
            else:
                old_hps = [model_hp[hp_name]]
            for hp in old_hps:  # For every config in the search-space
                if isinstance(hp, dict):
                    if "ag_args_ensemble" not in hp:
                        hp["ag_args_ensemble"] = {}
                    hp["ag_args_ensemble"]["use_child_oof"] = False
                    hp["ag_args_ensemble"]["fold_fitting_strategy"] = "sequential_local" # SequentialLocalFoldFittingStrategy
            model_hp[hp_name] = old_hps

        
        self.model.fit(df, num_cpus=self.cpu_count, num_gpus=0, ds_args={'enable_ray_logging': False}, **self.fit_kwargs)

    def predict(self, X):
        df = TabularDataset(X)
        return self.model.predict(df)

    def info(self) -> dict:
        return self.model.info()