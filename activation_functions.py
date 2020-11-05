from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):

    @staticmethod    
    @abstractmethod
    def g(u):
        pass
    
class DerivativeFunction(ABC):
    
    @staticmethod    
    @abstractmethod
    def dg(u):
        pass

class sigmoid(ActivationFunction):
    
    def g(self,u):
        return 1/(1+np.exp(-u))
    
class sigmoidDerivative(DerivativeFunction):
    
    def dg(self,u):
        return (1/(1+np.exp(-u))) * (1- (1/(1+np.exp(-u))) )