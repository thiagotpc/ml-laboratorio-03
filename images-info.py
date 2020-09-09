# O objetivo deste script é percorrer a pasta de imagens e obter informações para
# obter a média e mediana das larguras e alturas e outras informações.
# Tais informações poderiam trazer algum insight ou servir de base para testes
# com objetivo de melhorar a acurácia do classificador ao determinar o tamanho ideal
# das amostras de treinamento.

import os
from PIL import Image

import statistics


def get_dimensoes_info_from_images():
    count_alturas_usadas = [0]*250
    count_larguras_usadas = [0]*500
    list_alturas = []
    list_larguras = []
    list_proporcao =[]

    folder_images = "data/meses/data"

    for dirpath, _, filenames in os.walk(folder_images):
        for path_image in filenames:
            image = os.path.abspath(os.path.join(dirpath, path_image))
            with Image.open(image) as img:
                width, heigth = img.size
                count_alturas_usadas[heigth] = count_alturas_usadas[heigth] + 1
                count_larguras_usadas[width] = count_larguras_usadas[width] + 1
                list_alturas.append(heigth)
                list_larguras.append(width)
                list_proporcao.append(width/heigth)

    print('LARGURA')
    print('min: ', min(list_larguras))
    print('max: ', max(list_larguras))
    print('media: ', statistics.mean(list_larguras))
    print('mediana: ', statistics.median(list_larguras))

    print('ALTURA')
    print('min: ', min(list_alturas))
    print('max: ', max(list_alturas))
    print('media: ', statistics.mean(list_alturas))
    print('mediana: ', statistics.median(list_alturas))

    print('PROPORCAO')
    print('media: ', statistics.mean(list_proporcao))
    print('mediana: ', statistics.median(list_proporcao))


if __name__ == "__main__":
    get_dimensoes_info_from_images()
