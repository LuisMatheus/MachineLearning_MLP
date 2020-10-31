import numpy as np


class Perceptron:
    
    def __init__(self, input_values, output_values, learning_rate, activation_function):
        
        ones_column = np.ones((len(input_values), 1)) * -1
        self.input_values = np.append(ones_column, input_values, axis=1)
        self.output_values = output_values
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.W = np.random.rand(self.input_values.shape[1])
        #self.theta = np.random.rand(1)[0]
        
    def train(self):
        epochs = 1
        error = True
        
        print(f'Initial W: {self.W}')
        print('')
        
        while error:
            error = False
            print(f'[EPOCH {epochs}]')
            for x, d in zip(self.input_values, self.output_values):
                #u = np.dot(x, self.W) - self.theta
                u = np.dot(x, self.W)
                y = self.activation_function.g(u)
                print(f'Input: {x}, Output: {y}, Expected: {d}')
                
                if y != d:
                    print(f'Output is different from Expected, recalculating W!')
                    print(f'Actual W: {self.W}')
                    #print(f'Actual Theta: {self.theta}')
                    
                    self.W = self.W + self.learning_rate * (d - y) * x
                    #self.theta = self.theta + self.learning_rate * (d - y) * -1
                    error = True
                    
                    print(f'New W: {self.W}')
                    #print(f'New Theta: {self.theta}')
                    print('')
                    
                    break
                    
            epochs += 1
            print('')
            
        #print(f'Final W: {self.W}, Final Theta: {self.theta}')
        print(f'Final W: {self.W}')
        
    def evaluate(self, input_value):
        input_value =  np.append(-1, input_value)
        u = np.dot(input_value, self.W)
        return self.activation_function.g(u)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
