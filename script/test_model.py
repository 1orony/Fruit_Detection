import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import cv2  # Para desenhar as caixas delimitadoras
import time  # Para controlar o tempo do slideshow

# Caminho para o diretório que contém as imagens de teste
DIRETORIO_TESTE = 'Dataset/TEST/test'  # Atualize com o caminho correto

# Carregar o modelo
modelo = load_model('./models/models.keras')

# Função para carregar e processar as imagens
def carregar_imagem(caminho_imagem, tamanho_imagem=(128, 128)):
    imagem = load_img(caminho_imagem, target_size=tamanho_imagem)
    imagem = img_to_array(imagem)
    imagem = imagem / 255.0  # Normalizar para [0, 1]
    return imagem

# Listar imagens de teste
arquivos_imagem_teste = [f for f in os.listdir(DIRETORIO_TESTE) if f.endswith('.jpg')]

# Preparar os dados de teste
imagens_teste = []
nomes_teste = []

for arquivo in arquivos_imagem_teste:
    caminho_imagem = os.path.join(DIRETORIO_TESTE, arquivo)
    imagem = carregar_imagem(caminho_imagem)
    imagens_teste.append(imagem)
    nomes_teste.append(arquivo)  # Aqui você pode adicionar a lógica para extrair o rótulo do nome do arquivo

# Converter para array NumPy
imagens_teste = np.array(imagens_teste)

# Fazer previsões
previsoes = modelo.predict(imagens_teste)

# Obter as classes previstas
classes_previstas = np.argmax(previsoes, axis=1)

# Lista de nomes de frutas (deve corresponder ao índice das classes)
nomes_frutas = ['Apple', 'Banana', 'Orange']  # Adicione seus nomes de frutas reais aqui

# Exibir imagens com caixas tracejadas em formato de slideshow
def visualizar_slideshow(imagens, previsoes, nomes):
    plt.figure(figsize=(12, 12))
    
    for i in range(len(imagens)):
        # Carregar a imagem original para desenhar as caixas
        img_original = cv2.imread(os.path.join(DIRETORIO_TESTE, nomes[i]))
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

        # Simulação de coordenadas de bounding box (exemplo)
        height, width, _ = img_original.shape
        
        # Exemplo: aqui é onde você obtém as caixas delimitadoras das frutas
        frutas_detectadas = []
        if classes_previstas[i] < len(nomes_frutas):  # Verifica se a classe prevista é válida
            frutas_detectadas.append({
                "label": nomes_frutas[classes_previstas[i]],  # Nome da fruta correspondente
                "box": (10, 10, width - 20, height - 20)  # Exemplo: caixa cheia
            })

        # Desenhar as caixas tracejadas e rótulos
        for fruta in frutas_detectadas:
            box = fruta["box"]
            start_point = (box[0], box[1])
            end_point = (box[2], box[3])

            # Desenhar a caixa como linhas tracejadas
            for j in range(0, width, 10):  # Ajuste o intervalo para controle da "tracing"
                if j % 20 == 0:  # Desenhar apenas em posições específicas
                    cv2.line(img_original, (start_point[0] + j, start_point[1]), (start_point[0] + j + 10, start_point[1]), (0, 255, 0), thickness=2)
                    cv2.line(img_original, (start_point[0] + j, end_point[1]), (start_point[0] + j + 10, end_point[1]), (0, 255, 0), thickness=2)

            for j in range(0, height, 10):  # Ajuste o intervalo para controle da "tracing"
                if j % 20 == 0:  # Desenhar apenas em posições específicas
                    cv2.line(img_original, (start_point[0], start_point[1] + j), (start_point[0], start_point[1] + j + 10), (0, 255, 0), thickness=2)
                    cv2.line(img_original, (end_point[0], start_point[1] + j), (end_point[0], start_point[1] + j + 10), (0, 255, 0), thickness=2)

            # Desenhar o texto com tamanho aumentado
            cv2.putText(img_original, fruta["label"], (box[0] + 5, box[1] + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)  # Tamanho da fonte aumentado

        # Exibir a imagem
        plt.imshow(img_original)
        plt.axis('off')
        plt.show()
        plt.pause(2)  # Pausa de 2 segundos entre as imagens

    plt.close()

visualizar_slideshow(imagens_teste, classes_previstas, arquivos_imagem_teste)
