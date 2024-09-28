import os
import xml.etree.ElementTree as ET
from PIL import Image

# Caminho para o diretório que contém as imagens e anotações
diretorio = 'Dataset/TRAIN/train'

# Listar arquivos de imagem e de anotação
arquivos_imagem = [f for f in os.listdir(diretorio) if f.endswith('.jpg')]
arquivos_anotacao = [f for f in os.listdir(diretorio) if f.endswith('.xml')]

# Função para carregar anotações
def carregar_anotacao(nome_arquivo):
    tree = ET.parse(nome_arquivo)
    root = tree.getroot()
    
    # Extraia as informações que você precisa (exemplo: objetos, coordenadas, etc.)
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

# Processar cada imagem e carregar a respectiva anotação
for arquivo_imagem in arquivos_imagem:
    # Nome da imagem sem a extensão
    base_nome = arquivo_imagem.replace('.jpg', '')

    # Criar o nome da anotação correspondente
    nome_anotacao = f"{base_nome}.xml"  # O nome deve ser igual ao da imagem

    # Verificar se a anotação existe
    if nome_anotacao in arquivos_anotacao:
        caminho_imagem = os.path.join(diretorio, arquivo_imagem)
        caminho_anotacao = os.path.join(diretorio, nome_anotacao)

        # Carregar a imagem
        imagem = Image.open(caminho_imagem)

        # Carregar a anotação
        anotacao = carregar_anotacao(caminho_anotacao)

        # Aqui você pode fazer o que quiser com a imagem e as anotações
        print(f"Carregada a imagem: {caminho_imagem}")
        print(f"Anotações: {anotacao}")
    else:
        print(f"Anotação não encontrada para: {arquivo_imagem}. Esperado: {nome_anotacao}")

# Exibir uma mensagem ao final do processamento
print("Processamento concluído.")
