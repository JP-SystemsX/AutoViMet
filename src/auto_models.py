from autogluon.tabular import TabularPredictor, TabularDataset
from models import BaseModel


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

    def train(self, X, y):
        X["label"] = y
        df = TabularDataset(X)
        self.model.fit(df, num_cpus=2, num_gpus=0, ds_args={'enable_ray_logging': False}, **self.fit_kwargs)

    def predict(self, X):
        df = TabularDataset(X)
        return self.model.predict(df)

    def info(self) -> dict:
        return self.model.info()