from abc import ABC, abstractmethod
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from m5py import M5Prime
from cubist import Cubist
from sklearn.ensemble import GradientBoostingRegressor

import m5py.main as m5main # Bug in M5 --> Monkey patching
from m5py.main import LinRegLeafModel
import numpy as np
from sklearn.tree import _tree
from sklearn.tree._tree import DOUBLE
import uuid
from pathlib import Path



class BaseModel(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def train(self, X, y):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, X):
        raise NotImplementedError
    
    
class Lasso(BaseModel):
    def __init__(self, **kwargs):
        self.model = linear_model.Lasso(**kwargs)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    

class PolynomialRegression(BaseModel):
    def __init__(self, degree, **kwargs):
        self.degree = degree
        self.model = linear_model.LinearRegression(**kwargs)
    
    def train(self, X, y):
        if self.degree > 1:
            self.poly = PolynomialFeatures(degree=self.degree)
            X = self.poly.fit_transform(X)
        self.model.fit(X, y)
    
    def predict(self, X):
        if self.degree > 1:
            X = self.poly.transform(X)
        return self.model.predict(X)
    
class PartialLeastSquares(BaseModel):
    def __init__(self, **kwargs):
        self.model = PLSRegression(**kwargs)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    

class GaussianProcess(BaseModel):
    def __init__(self, kernel, **kwargs):
        match kernel:
            case "DotProduct+RBF":
                kernel = DotProduct() + RBF() + WhiteKernel()
            case "DotProduct":
                kernel = DotProduct() + WhiteKernel()
            case "RBF":
                kernel = RBF() + WhiteKernel()
            case "Matern":
                kernel = Matern() + WhiteKernel()
            case "WhiteKernel":
                kernel = WhiteKernel()
        self.model = GaussianProcessRegressor(kernel=kernel, **kwargs)
    
    def train(self, X, y):
        if len(X) > 1000:
            X = X.sample(n=1000, random_state=42)
            y = y.loc[X.index]
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X, return_std=False)
    

class SupportVectorRegression(BaseModel):
    def __init__(self, **kwargs):
        self.model = SVR(**kwargs)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    

class XGBoost(BaseModel):
    def __init__(self, **kwargs):
        self.model = XGBRegressor(enable_categorical=True, **kwargs)
    
    def train(self, X, y):
        self.model.fit(X=X, y=y)
    
    def predict(self, X):
        return self.model.predict(X)
    

class LightGBM(BaseModel):
    def __init__(self, **kwargs):
        self.model = LGBMRegressor(**kwargs)
    
    def train(self, X, y):
        self.model.fit(X=X, y=y)
    
    def predict(self, X):
        return self.model.predict(X)
    
class CatBoost(BaseModel):
    def __init__(self, **kwargs):
        train_dir = Path("./cache/catboost/" + str(uuid.uuid4()))
        train_dir.mkdir(exist_ok=True, parents=True)
        self.model = CatBoostRegressor(verbose=0, train_dir=train_dir, **kwargs) # This thing blocks itself if trained in parallel
    
    def train(self, X, y):
        cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
        self.model.fit(X=X, y=y, cat_features=cat_cols)
    
    def predict(self, X):
        return self.model.predict(X)
    

class RandomForest(BaseModel):
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)
    
    def train(self, X, y):
        self.model.fit(X=X, y=y)
    
    def predict(self, X):
        return self.model.predict(X)
    
class ExtraTrees(BaseModel):
    def __init__(self, **kwargs):
        self.model = ExtraTreesRegressor(**kwargs)
    
    def train(self, X, y):
        self.model.fit(X=X, y=y)
    
    def predict(self, X):
        return self.model.predict(X)


class TabPFN(BaseModel):
    def __init__(self, **kwargs):
        from tabpfn import TabPFNRegressor # TODO Move up
        self.model = TabPFNRegressor(device='auto', **kwargs)
    
    def train(self, X, y):
        self.model.fit(X=X, y=y)
    
    def predict(self, X):
        return self.model.predict(X)


class MLP(BaseModel):
    def __init__(self, **kwargs):
        self.model = MLPRegressor(**kwargs)
    
    def train(self, X, y):
        self.model.fit(X=X, y=y)
    
    def predict(self, X):
        return self.model.predict(X)


class M5(BaseModel):
    def __init__(self, leaf_model="None", **kwargs):
        match leaf_model:
            case "Lasso":
                leaf_model = linear_model.Lasso()
            case "Ridge":
                leaf_model = linear_model.Ridge()
            case "LinearRegression":
                leaf_model = linear_model.LinearRegression()
            case "RandomForestRegressor":
                leaf_model = RandomForestRegressor()
            case "None":
                leaf_model = None

        self.model = M5Prime(leaf_model=leaf_model, **kwargs)
    
    def train(self, X, y):
        # Reset Index
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        self.model.fit(X=X, y=y)
    
    def predict(self, X):
        return self.model.predict(X)


class CubistModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = Cubist(auto=False, **kwargs)
    
    def train(self, X, y):
        self.model.fit(X=X, y=y)
    
    def predict(self, X):
        return self.model.predict(X)


class GradientBoosting(BaseModel): # bstTrees 
    def __init__(self, **kwargs):
        self.model = GradientBoostingRegressor(**kwargs)
    
    def train(self, X, y):
        self.model.fit(X=X, y=y)
    
    def predict(self, X):
        return self.model.predict(X)
    

class Ridge(BaseModel):
    def __init__(self, **kwargs):
        self.model = linear_model.Ridge(**kwargs)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
class BayesianRidge(BaseModel):
    def __init__(self, **kwargs):
        self.model = linear_model.BayesianRidge(**kwargs)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class ElasticNet(BaseModel):
    def __init__(self, **kwargs):
        self.model = linear_model.ElasticNet(**kwargs)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class LARS(BaseModel):
    def __init__(self, **kwargs):
        self.model = linear_model.Lars(**kwargs)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class LassoLars(BaseModel):
    def __init__(self, **kwargs):
        self.model = linear_model.LassoLars(**kwargs)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)


class Huber(BaseModel):
    def __init__(self, **kwargs):
        self.model = linear_model.HuberRegressor(**kwargs)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    

class Dummy(BaseModel):
    def __init__(self, **kwargs):
        from sklearn.dummy import DummyRegressor
        self.model = DummyRegressor(**kwargs)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    

class PassiveAggressive(BaseModel):
    def __init__(self, **kwargs):
        self.model = linear_model.PassiveAggressiveRegressor(**kwargs)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class Tweedie(BaseModel): # ~ Generalized Linear Model with Tweedie Distribution
    def __init__(self, **kwargs):
        self.model = linear_model.TweedieRegressor(**kwargs)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

# ! Monkey Patch for M5
def predict_from_leaves(m5p, X, smoothing=True, smoothing_constant=15):
    """
    Predicts using the M5P tree, without using the compiled sklearn
    `tree.apply` subroutine.

    The main purpose of this function is to apply smoothing to a M5P model tree
    where smoothing has not been pre-installed on the models. For examples to
    enable a model to be used both without and with smoothing for comparisons
    purposes, or for models whose leaves are not Linear Models and therefore
    for which no pre-installation method exist.

    Note: this method is slower than `predict_from_leaves_no_smoothing` when
    `smoothing=False`.

    Parameters
    ----------
    m5p : M5Prime
        The model to use for prediction
    X : array-like
        The input data
    smoothing : bool
        Whether to apply smoothing
    smoothing_constant : int
        The smoothing constant `k` used as the prediction weight of parent node model.
        (k=15 in the articles).


    Returns
    -------

    """
    # validate and converts dtype just in case this was directly called
    # e.g. in unit tests
    X = m5p._validate_X_predict(X, check_input=True)

    tree = m5p.tree_
    node_models = m5p.node_models
    nb_samples = X.shape[0]
    y_predicted = -np.ones((nb_samples, 1), dtype=DOUBLE)

    # sample_ids_to_leaf_node_ids = tree.apply(X)
    def smooth_predictions(ancestor_nodes, X_at_node, y_pred_at_node):
        # note: y_pred_at_node can be a constant
        current_node_model_id = ancestor_nodes[-1]
        for _i, parent_model_id in enumerate(reversed(ancestor_nodes[:-1])):
            node_nb_train_samples = tree.n_node_samples[current_node_model_id]
            parent_model = node_models[parent_model_id]
            parent_predictions = parent_model.predict(X_at_node) if len(X_at_node) > 0 else np.array([]) #! Catch empty leafs

            # --- minimal shape alignment fix ---
            if y_pred_at_node.shape != parent_predictions.shape:
                y_pred_at_node = y_pred_at_node.reshape(-1, 1)
                parent_predictions = parent_predictions.reshape(-1, 1)
            # -----------------------------------

            y_pred_at_node = (
                node_nb_train_samples * y_pred_at_node
                + smoothing_constant * parent_predictions
            ) / (node_nb_train_samples + smoothing_constant)
            current_node_model_id = parent_model_id

        return y_pred_at_node


    def apply_prediction(node_id, ids=None, ancestor_nodes=None):
        first_call = False
        if ids is None:
            ids = slice(None)
            first_call = True
        if smoothing:
            if ancestor_nodes is None:
                ancestor_nodes = [node_id]
            else:
                ancestor_nodes.append(node_id)

        left_id = tree.children_left[node_id]
        if left_id == _tree.TREE_LEAF:
            # ... and tree.children_right[node_id] == _tree.TREE_LEAF
            # LEAF node: predict
            # predict
            node_model = node_models[node_id]
            # assert (ids == (sample_ids_to_leaf_node_ids == node_id)).all()
            if isinstance(node_model, LinRegLeafModel):
                X_at_node = X[ids, :]
                predictions = node_model.predict(X_at_node) if len(X_at_node) > 0 else np.array([]) #! Catch empty leafs
            else:
                # isinstance(node_model, ConstantLeafModel)
                predictions = tree.value[node_id]
                if smoothing:
                    X_at_node = X[ids, :]
            if predictions.ndim == 1: #! Adjust Shape
                predictions = predictions[:, None]
            # finally apply smoothing
            if smoothing:
                y_predicted[ids] = smooth_predictions(ancestor_nodes, X_at_node, predictions)
            else:
                y_predicted[ids] = predictions
        else:
            right_id = tree.children_right[node_id]
            # non-leaf node: split samples and recurse
            left_group = np.zeros(nb_samples, dtype=bool)
            left_group[ids] = X[ids, tree.feature[node_id]] <= tree.threshold[node_id]
            right_group = (~left_group) if first_call else (ids & (~left_group))

            # important: copy ancestor_nodes BEFORE calling anything, otherwise
            # it will be modified
            apply_prediction(
                left_id, ids=left_group, ancestor_nodes=(ancestor_nodes.copy() if ancestor_nodes is not None else None)
            )
            apply_prediction(right_id, ids=right_group, ancestor_nodes=ancestor_nodes)

    # recurse to fill all predictions
    apply_prediction(0)

    return y_predicted

m5main.predict_from_leaves = predict_from_leaves