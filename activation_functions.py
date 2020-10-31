from abc import ABC, abstractmethod
import math

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
        return 1/(1+math.exp(-u))
    
class sigmoidDerivative(DerivativeFunction):
    
    def dg(self,u):
        return (1/(1+math.exp(-u))) * (1- (1/(1+math.exp(-u))) )