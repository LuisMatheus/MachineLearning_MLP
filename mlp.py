import numpy as np
from activation_functions import sigmoid, sigmoidDerivative

class MLP:
    
    def __init__(self, input_values, output_values, layers, activation_function=sigmoid(), derivative_function=sigmoidDerivative(), learning_rate=0.1, precision=1e-6):
       ones_column = np.ones((len(input_values), 1)) * -1
       self.input_values = np.append(ones_column, input_values, axis=1)
       self.output_values = output_values
       self.learning_rate = learning_rate
       self.precision = precision
       self.activation_function = activation_function
       self.derivative_function = derivative_function
       
       self.W = []
       neuron_input = self.input_values.shape[1]
       for i in range(len(layers)):
           self.W.append(np.random.rand(layers[i], neuron_input))
           neuron_input = layers[i] + 1
          
       self.epochs = 0
       self.eqms = []
       
    def train(self):
        
        print('Initializing the train process....')
        error = True
        
        while error:
            print(f'[EPOCH] {self.epochs}')
            self.epochs += 1
           
            eqm_previous = self.eqm()
            
            for x, d in zip(self.input_values, self.output_values):
            
                I1 = np.dot(self.W[0], x)
                Y1 = np.zeros(I1.shape)
                for i in range(Y1.shape[0]):
                    Y1[i] = self.activation_function.g(I1[i])
                Y1 = np.append(-1, Y1)
                
                I2 = np.dot(self.W[1], Y1)
                Y2 = np.zeros(I2.shape)
                for i in range(Y2.shape[0]):
                    Y2[i] = self.activation_function.g(I2[i])
                    
                #delta2
                delta2 = np.zeros(I2.shape)
                for i in range(Y2.shape[0]):
                    delta2[i] = d[i]-Y2[i] * self.derivative_function.dg(I2[i])
                
                #ajustar W2
                self.W[1] = self.W[1] + self.learning_rate * np.dot(np.transpose(np.array([delta2])),np.array([Y1]))
                
                #delta1
                delta1 = np.dot(np.transpose(self.W[1]),np.transpose(np.array([delta2])))
                
                #remover o bias
                delta1 = delta1[1:,:]
                
                for i in range(delta1.shape[0]):
                    delta1[i] = delta1[i] * self.derivative_function.dg(I1[i]) 
                
                #ajustar W1
                self.W[0] = self.W[0] + self.learning_rate * np.dot(delta1, np.array([x]))
                
                #saida = W2ji = W2ji + n * ((sum(dj -Y2)) * g'(I2)) * Y1 -> TransPdelta2 * Y1
                #Wij1 = W1ji + n * (sum(delta2 * W2kj)*g'(I1)) *xi
                                
                
            #print(f'New W: {self.W}')
            eqm_actual = self.eqm()
            
            if abs(eqm_actual - eqm_previous) < self.precision:
                error = False
                
        #print(f'Final W: {self.W}')
         
        
    def eqm(self):
        
        eq = 0
        
        for x, d in zip(self.input_values, self.output_values):
            
            I1 = np.dot(self.W[0], x)
            Y1 = np.zeros(I1.shape)
            for i in range(Y1.shape[0]):
                Y1[i] = self.activation_function.g(I1[i])
            Y1 = np.append(-1, Y1)
            
            I2 = np.dot(self.W[1], Y1)
            Y2 = np.zeros(I2.shape)
            for i in range(Y2.shape[0]):
                Y2[i] = self.activation_function.g(I2[i])
                
            eq += 0.5 * sum((d - Y2) ** 2)
            
        return eq/len(self.output_values)
    
    def test(self,x):
        
        
        #x = np.transpose(np.array([x]))
        x = np.append(-1, x)
        
        I1 = np.dot(self.W[0], x)
        Y1 = np.zeros(I1.shape)
        for i in range(Y1.shape[0]):
            Y1[i] = self.activation_function.g(I1[i])
        Y1 = np.append(-1, Y1)
        
        I2 = np.dot(self.W[1], Y1)
        Y2 = np.zeros(I2.shape)
        for i in range(Y2.shape[0]):
            Y2[i] = self.activation_function.g(I2[i])
            
        return Y2
        