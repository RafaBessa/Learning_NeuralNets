import matplotlib.pyplot as plt 
import random
#Perceptron para porta AND, com o plus da variação poder ser positiva ou negativa 

DataAnd = [
        [0,0,0],
        [0,1,0],
        [1,0,0],
        [1,1,1]
        ]

DataOR = [
          [0,0,0],
          [0,1,1],
          [1,0,1],
          [1,1,1]
        ]

#Classe responsalvel por receber os dados (parametros) e treinar seus pesos para resolver o problema
#modelagem para and =  Perceptron([-0.3, 0.5, 0.6],0,0.1)
class Perceptron():
    NumEpoch = 0
    def __init__(self, w, threshold,variation):
     self.w = w
     self.threshold = threshold
     self.variation = variation

    def TreinarEpoca(self,Data):
        print("Epoch - " + str(self.NumEpoch))
        ContCorrec = 0
        for v in Data:
          if  self.Calcular(v) == 1:
              ContCorrec += 1
        self.NumEpoch +=1
        print("Valores Finais na Epoch  -- " + str(self.w))
        print("---------------------------------")
        return ContCorrec

    def Calcular(self,vet) :
        print(vet)
        func = self.w[0] + vet[0] * self.w[1] + vet [1] * self.w[2] #u         
        print("  \"" + str(vet[0]) +"\" * "  + str(self.w[1]) + "  \" " + str(vet[1]) +"\" * " + str(self.w[2]) + " + Bias " + str(self.w[0]) + " = " + str(func) ) 
        
        if (func >= self.threshold and vet[2] == 1) or (func < self.threshold and vet[2] == 0) :
            return 0 #se funcionar
        else:
            if(func >= self.threshold and vet[2] == 0 ):
                self.Corrigir(vet,-1)
                return 1 #se nao funcionar
            else:  
                self.Corrigir(vet,1)
            return 1 #se nao funcioanr

    #Direction deve ser 1 para corrigir positivamento e -1 para corrigir negativamente
    def  Corrigir(self,vet,direction):
        self.w[0] += self.variation * direction
        print("Variacao Bias - " + str(self.w[0]))
        if(vet[0] > 0 ):
            self.w[1] += self.variation * direction
            print("Variacao W1 - " + str(self.w[1]))
        if(vet[1] > 0):
            self.w[2] += self.variation * direction
            print("Variacao W2 - " + str(self.w[2]))


#p = Perceptron([-0.3, 0.5, 0.6],0,0.1)
p = Perceptron([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)],0,0.1)
pOr = Perceptron([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)],0,0.1)

#for i in range(1, 10):
#while p.TreinarEpoca(DataAnd) != 0:
 #   continue

print("#####################################")


while pOr.TreinarEpoca(DataOR) != 0:
    continue


#graph plot
# W1x1 + W2x2 + W0 = y
# 1 ponto quanto x = 0
# 1 ponto quanto x2 = 0
# 

x = [p.w[0], p.w[1]*1+p.w[0]]
y = [p.w[2]*1+p.w[0] , p.w[0]]
#plt.plot(x,y)
#plt.show()