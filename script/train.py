import os
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Caminho para o diretório de dados
DIRETORIO = 'Dataset/TRAIN/train'
TAMANHO_IMAGEM = (128, 128)

# Função para carregar anotações de XML
def carregar_anotacao(nome_arquivo):
    tree = ET.parse(nome_arquivo)
    root = tree.getroot()
    
    objetos = []
    for obj in root.findall('object'):
        nome = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objetos.append((nome, (xmin, ymin, xmax, ymax)))
    
    return objetos

# Função para carregar imagens e rótulos
def carregar_dados(diretorio, tamanho_imagem):
    imagens = []
    rotulos = []
    
    arquivos_imagem = [f for f in os.listdir(diretorio) if f.endswith('.jpg')]
    arquivos_anotacao = [f for f in os.listdir(diretorio) if f.endswith('.xml')]
    
    for arquivo_imagem in arquivos_imagem:
        base_nome = arquivo_imagem.replace('.jpg', '')
        nome_anotacao = f"{base_nome}.xml"
        
        if nome_anotacao in arquivos_anotacao:
            caminho_imagem = os.path.join(diretorio, arquivo_imagem)
            caminho_anotacao = os.path.join(diretorio, nome_anotacao)
            
            # Carregar a imagem
            imagem = Image.open(caminho_imagem).convert('RGB').resize(tamanho_imagem)
            imagem = np.array(imagem) / 255.0  # Normalizar
            
            # Carregar anotação
            anotacao = carregar_anotacao(caminho_anotacao)
            
            # Extrair rótulo (classe da primeira anotação encontrada)
            rotulo = anotacao[0][0]  # Supondo que estamos usando a primeira anotação
            if rotulo == 'apple':
                rotulo = 0
            elif rotulo == 'banana':
                rotulo = 1
            elif rotulo == 'orange':
                rotulo = 2
            elif rotulo == 'mixed':  # Classe para frutas mistas
                rotulo = 3
            
            imagens.append(imagem)
            rotulos.append(rotulo)
    
    return np.array(imagens), np.array(rotulos)

# Carregar dados
imagens, rotulos = carregar_dados(DIRETORIO, TAMANHO_IMAGEM)
rotulos = to_categorical(rotulos, num_classes=4)  # Atualizado para 4 classes

# Dividir em conjunto de treino e validação
X_train, X_val, y_train, y_val = train_test_split(imagens, rotulos, test_size=0.2, random_state=42)
# Função para construir o modelo
def construir_modelo():
    model = Sequential()

    # Primeira camada de convolução e pooling
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Segunda camada de convolução e pooling
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Terceira camada de convolução e pooling
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Camadas totalmente conectadas
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))  # Atualizado para 4 classes

    # Compilando o modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Construir o modelo
model = construir_modelo()
# Configuração do aumento de dados
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Ajustando o gerador ao conjunto de treinamento
datagen.fit(X_train)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('fruit_detect_model.keras', monitor='val_accuracy', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
# Treinar o modelo com data augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# Avaliar o modelo no conjunto de validação
scores = model.evaluate(X_val, y_val)
print(f"Acurácia no conjunto de validação: {scores[1] * 100:.2f}%")


# Salvar o modelo final
model.save('./models/models.keras')
