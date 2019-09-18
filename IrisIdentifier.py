import matplotlib.pyplot as plt
import numpy as np
import random

data = np.loadtxt(fname="irisData.txt")
print(len(data))
x1 = []
x2 = []
y1 = []
y2 = []
for d in data:
    if d[2] == 1:
        x1.append(d[0])
        y1.append(d[1])
        d[2] = 0
    else:
        x2.append(d[0])
        y2.append(d[1])
        d[2] = 1
        
plt.scatter(x1,y1,color="red" ) 
plt.scatter(x2,y2,color = "blue")
#plt.show()
#print(data)




class Perceptron():
    #w = [Bias,W1,W2]
    def __init__(self, w, threshold,variation):
     self.w = w
     self.threshold = threshold
     self.variation = variation

    def Treinar(self,Data): #Data = v[x,y,resultado]
        print("Data - " + str(Data))
        return self.Calcular(Data)
        
    def Calcular(self,vet):
       
        print("Weights - " + str(self.w))
        func = self.w[0] + vet[0] * self.w[1] + vet [1] * self.w[2] #u         
        print("Func Value - " + str(func)) 
        if func >= self.threshold:
            func = 1
        else:
            func = 0
        erro = vet[2] - func    
        #func = 1/( 1 + np.exp(func))
        self.Corrigir(vet,erro)    
        return erro #se nao funcioanr

    #Direction deve ser 1 para corrigir positivamento e -1 para corrigir negativamente
    def  Corrigir(self,vet,corr):#variacao ajuste 
        self.w[0] += self.variation * corr
        self.w[1] += self.variation * corr * vet[0]
        self.w[2] += self.variation * corr * vet[1]
       

#p= Perceptron([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)],0,0.1)
#p= Perceptron([0.3353116356850554, -0.5594348883530098, -0.256859859593127],0,0.05)
#-3.63 5.61 2.10
#p = Perceptron([2.10,-3.63,5.61],0,0.1)
p = Perceptron([0.7,0.7,-1.3],0,0.1)
print("Weight Inicial = " + str(p.w))

epoch = 0
resposta = 1
erro = 100
file_object  = open("erro.txt", "w") 
while epoch<10 and erro > 1:
    print("epoch - " + str(epoch))  
    erro = 0
    for d in data:
        erro += abs(p.Treinar(d))

    erro = (erro/len(data))*100
    epoch +=1
    print("-"*90)
    print("Weight Final = " + str(p.w))
    file_object.write("Weight Final = " + str(p.w)+"%\n")
    print("erro = " + str(erro))
    file_object.write(str(epoch) +" - "+str(erro)+"%\n")
    print("-"*90)

file_object.close()

xp1 = ((-1)*p.w[0]/p.w[1] )
xp2 = ((-1)*p.w[0]/p.w[2] )
pontos = [[0,xp1],[xp2,0]]
z = np.polyfit(pontos[0], pontos[1], 1) #pega uma equacao para a reta
p = np.poly1d(z) #a transforma em polinomial
print(p)
x = np.arange(10)
y = p(x)
plt.plot(x,y, '--',color="black")
print(plt.axis())
plt.show()
