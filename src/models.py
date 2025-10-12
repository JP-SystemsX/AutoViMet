from abc import ABC, abstractmethod
from sklearn import linear_model


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
    def __init__(self, alpha, **kwargs):
        self.model = linear_model.Lasso(alpha=alpha)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    