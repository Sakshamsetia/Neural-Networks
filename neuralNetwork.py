import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    
    def __init__(self,layers):
        np.random.seed(10) #Seed fixed so no variation in training same model on multiple runs
        self.params = {}
        for l in range(1,len(layers)):
            self.params["W"+str(l)] = np.random.randn(layers[l],layers[l-1])*0.1
            self.params["B"+str(l)] = np.zeros(layers[l])
        
    def _sigmoid(self,x): #For making sure things skew in between 0 and 1
        return 1/(1+np.exp(-1*x))
    
    def _loss(self,out,actual):
        eps = 1e-15
        out = np.clip(out, eps, 1 - eps)
        return -1*np.dot(actual,np.log(out))
    
    def forward_prop(self,inputData):
        n = len(self.params)//2
        for i in range(n):
            #Linear Hypothesis
            Z = np.dot(inputData,self.params["W"+str(i+1)])+self.params["B"+str(i+1)]
            inputData = self._sigmoid(Z)
        
        return inputData
    