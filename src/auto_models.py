import os
from autogluon.tabular import TabularPredictor, TabularDataset
from models import BaseModel
from autogluon.tabular.configs.presets_configs import tabular_presets_dict
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from autogluon.core.models.ensemble.fold_fitting_strategy import SequentialLocalFoldFittingStrategy
import flaml
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
import pandas as pd

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
   
        self.model.fit(
            df, 
            num_cpus=self.cpu_count, 
            num_gpus=0, 
            ds_args={'enable_ray_logging': False},
            ag_args_fit={
                "num_cpus": self.cpu_count,
            },
            **self.fit_kwargs)


    def predict(self, X):
        df = TabularDataset(X)
        return self.model.predict(df)

    def info(self) -> dict:
        return self.model.info()
    


class FLAML(AutoModel):

    def __init__(self, metric, config, search_id, **kwargs):
        self.fit_kwargs = config.get("fit_kwargs", {})
        self.feature_names = None
        self.categories = {}
        self.model = flaml.AutoML()

    def train(self, X, y):
        X = X.copy()
        self.feature_names = [str(i) for i in range(X.shape[1])]
        X.columns = self.feature_names

        for col in X.select_dtypes(include=["category", "object", "string"]).columns:
            X[col] = X[col].astype(str)
            self.categories[col] = X[col].unique().tolist()
            X[col] = pd.Categorical(X[col], categories=self.categories[col])

        self.model.fit(X, y, **self.fit_kwargs)

    def predict(self, X):
        X = X.copy()
        X.columns = self.feature_names

        for col, cats in self.categories.items():
            X[col] = pd.Categorical(X[col].astype(str), categories=cats)

        return self.model.predict(X)


class LightAutoML(AutoModel):

    def __init__(self, metric, config, search_id, **kwargs):
        self.target = "target"
        self.feature_names = None
        self.model = TabularAutoML(
            task=Task("reg", metric="mse"),
            **config.get("init_kwargs", {}),
        )

    def train(self, X, y):
        train = X.copy()

        self.feature_names = [str(i) for i in range(X.shape[1])]
        train.columns = self.feature_names
        train[self.target] = y

        self.model.fit_predict(
            train,
            roles={"target": self.target},
        )

    def predict(self, X):
        X = X.copy()
        X.columns = self.feature_names

        return self.model.predict(X).data[:, 0]