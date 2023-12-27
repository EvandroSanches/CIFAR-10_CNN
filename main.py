import keras.optimizers.schedules
from keras.datasets import cifar10
from keras.layers import Dense,Dropout,Conv2D,MaxPool2D,BatchNormalization,Flatten
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import cv2
import matplotlib.pyplot as plt

#
nome_classes = ['Avião', 'Automovel', 'Passáro','Gato', 'Veado', 'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminhão']

batch_size = 50
epoches = 40

def Pre_Processamento(previsor):
    #Criando array vazio para adicionar a imagem
    img = np.empty((0,32,32,3))

    #Redimensionando o tamanho da imagem
    previsor = cv2.resize(previsor, (32,32))

    #Adicionando a imagem a um array
    previsor = np.append(img, np.array([previsor]), axis=0)

    #Tratando imagem
    previsor = previsor.astype('float32')
    previsor = previsor / 255

    return previsor

def CarregaDados():
    #Carregando base de daos CIFAR-10
    (x,y), (x_teste,y_teste) = cifar10.load_data()

    #Juntando dados de treino e de teste
    previsores = np.concatenate((x,x_teste))
    classes = np.concatenate((y,y_teste))

    #Tranformando dados em formato float32 e reduzindo valores para escala de 0 a 1
    previsores = previsores.astype('float32')
    previsores = previsores / 255

    #Codificando classes
    classes = to_categorical(classes)

    return previsores, classes

def CriaModelo():

    modelo = Sequential()

    #Primeira camada convolucional
    modelo.add(Conv2D(40, (3,3), activation='relu', input_shape=(32,32,3)))
    modelo.add(BatchNormalization())
    modelo.add(MaxPool2D(2,2))

    #Segunda camada convolucional
    modelo.add(Conv2D(40, (3,3), activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(MaxPool2D(2,2))

    #Transformando matriz de dados em vetor
    modelo.add(Flatten())

    #Camada Densa e saida
    modelo.add(Dense(units=150, activation='relu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(units=150, activation='relu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(units=10, activation='softmax'))

    modelo.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics='categorical_accuracy')

    return modelo

def Treinamento():
    #Carregando dados previsores
    previsores, classes = CarregaDados()

    #Gerando classe de modelo com KerasClassifier
    modelo = KerasClassifier(build_fn=CriaModelo, epochs=epoches, batch_size=batch_size)

    #fazendo treinamento com validação cruzada
    result = cross_val_score(estimator=modelo, X=previsores, y=classes, cv=10)

    #Gerando resultados de analise, média e desvio padrão
    print(result)
    print(result.mean())
    print(result.std())

def GeraModelo():
    #Carregando dados
    previsores, classes = CarregaDados()

    #Cria modelo
    modelo = CriaModelo()

    #Treina e salva modelo
    modelo.fit(previsores, classes, epochs=epoches, batch_size=batch_size)
    modelo.save('Modelo.0.1')

#Passar caminha da imagem
def Previsao(caminho):
    #Carregando modelo ja treinado
    modelo = load_model('Modelo.0.1')

    #Lendo imagem
    previsor = cv2.imread(caminho)

    previsor = Pre_Processamento(previsor)

    #Imrpimindo Resultado
    result = modelo.predict(previsor)
    print(nome_classes[np.argmax(result)])


Treinamento()
