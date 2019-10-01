import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import random





data = np.loadtxt(fname="irisData.txt")
#print(len(data))

def KMeans(Data,k):
    centroid = []
   
    tipo = [-1]*Data.shape[0]
    numDif = -1
    for x in range(k):
        centroid.append((random.uniform(np.min(data[:,0]),np.max(data[:,0])),
        random.uniform(np.min(data[:,1 ]),np.max(data[:,1 ]))))
    i=0

    print(centroid)
    cont =0
    while numDif != 0:
        numDif=0
        #calcular tipos
        i=0
        for d in Data:
            a =  MenorDistancia(centroid,d)
            if(a != tipo[i]):
                numDif+=1
            tipo[i] = a
            i+=1
            
        print(tipo)
        #corrigir centroids
        Newcentroid = [(0,0)]*len(centroid)
        Numcentroid = [0]*len(centroid)
        i=0
        for d in Data:
            Newcentroid[tipo[i]] = (Newcentroid[tipo[i]][0]+d[0],Newcentroid[tipo[i]][1]+d[1])
            
            Numcentroid[tipo[i]] +=1
            i+=1
        i=0
        
        for x in Newcentroid:
            if(Numcentroid[i]!=0):
                centroid[i] = (x[0]/Numcentroid[i],x[1]/Numcentroid[i])               
            
            
            i+=1
        print(centroid)
       # centroid = Newcentroid
       # print(centroid)
        cont+=1
        #-----------------
    #plotgraf(Data,tipo,centroid)
    return calcInercia(Data,tipo,centroid)         
   

    

    

def MenorDistancia(centroid, d):
    menor = -1
    imenor = -1
    i=0
    for c in centroid:
      #  print("Menor: " + str(menor) +" - " + str( DistanciaEuclidiana(c[0],d[0],c[1],d[1])))
        if(menor == -1) or (menor > DistanciaEuclidiana(c[0],d[0],c[1],d[1])):
            imenor = i
            menor = DistanciaEuclidiana(c[0],d[0],c[1],d[1])
        i+=1
    return imenor

def DistanciaEuclidiana(x1,x2,y1,y2):
    return ((((x1-x2)**2 + (y1-y2)**2))**(1/2))

def plotgraf(data,tipo,centroid):
    a = mcolors.CSS4_COLORS
    a = list(a.keys())
    
    i=0
    for d in data:
        plt.scatter(d[0],d[1],color = a[tipo[i]+20], alpha = 1)
        i+=1
    i=0
    for c in centroid:
        plt.scatter(c[0],c[1],s =  plt.rcParams['lines.markersize'] ** 3, color = a[i+20], marker="s", alpha = 0.7)
        i+=1
    plt.show()
    #for d in data:
def calcInercia(data,tipo,centroid):
    Inercia = 0
    i = 0
    for d in data:
        #a inercia eh a soma das distancias de cada ponto ao se centroid ao quadrado
        Inercia += (DistanciaEuclidiana(d[0],centroid[tipo[i]][0] ,d[1],centroid[tipo[i]][1]))**2
        i+=1
    return Inercia



Inercia = []
for i in range(1,16):
    Inercia.append(KMeans(data,i))
   # print (i)
plt.plot(range(1,16),Inercia,color='red')
plt.show()
