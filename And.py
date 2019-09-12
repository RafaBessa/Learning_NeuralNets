
#Perceptron para porta AND 







Data = [[0,0,0],
        [0,1,0],
        [1,0,0],
        [1,1,1]
        ]


#Classe responsalvel por receber os dados (parametros) e treinar seus pesos para resolver o problema
class Perceptron():
    threshold = 0
    variation = 0.1
    w = [-0.3, 0.5, 0.6] #iniciando meus works
    NumEpoch = 0
    def __init(self):
        pass

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
        func = self.w[0] + vet[0] * self.w[1] + vet [1] * self.w[2] #u         
        print("  \"" + str(vet[0]) +"\" * "  + str(self.w[1]) + "  \" " + str(vet[1]) +"\" * " + str(self.w[2]) + " + Bias " + str(self.w[0]) + " = " + str(func) ) 
        
        if (func >= self.threshold and vet[2] == 1) or (func < self.threshold and vet[2] == 0) :
            return 0 #se funcionar
        else:
            self.Corrigir(vet)
            return 1 #se nao funcioanr
    def  Corrigir(self,vet):
        self.w[0] -= self.variation
        print("Variacao Bias - " + str(self.w[0]))
        if(vet[0] > 0 ):
            self.w[1] -= self.variation
            print("Variacao W1 - " + str(self.w[1]))
        if(vet[1] > 0):
            self.w[2] -= self.variation
            print("Variacao W2 - " + str(self.w[2]))


p = Perceptron()

#for i in range(1, 10):
while p.TreinarEpoca(Data) != 0:
    continue

