from abc import ABC, abstractmethod
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern


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
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X, return_std=False)