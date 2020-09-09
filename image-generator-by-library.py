import os
import glob
import cv2
from augment import distort, stretch, perspective

filenames_for_training = []
classes_month = []

# entrada
input_filename_train = 'data/meses/train.txt'
input_images_dir = 'data/meses/data'

# diretorio de saída
output_path = 'out/new-data/'
images_dir = output_path + 'using-library'
fullpath_file_train = output_path + 'using-library-train.txt'


# limpa pasta de saída
def clear_out():
    if os.path.exists(fullpath_file_train):
        os.remove(fullpath_file_train)
    for file in os.scandir(images_dir):
        if file.name.endswith(".jpg"):
            os.remove(file.path)


# carrega imagens do treinamento original
def load_train():
    f = open(input_filename_train,"r")
    lines = f.readlines()
    for x in lines:
        filename, class_month = x.split(' ')
        filenames_for_training.append(filename)
        classes_month.append(int(class_month.replace("\n", "")))
    f.close()
    print(filenames_for_training)
    print(classes_month)


# registra a imagem e a classe
def adiciona_rotulo(class_month, filename):
    file_object = open(fullpath_file_train, 'a')
    file_object.write(f'{filename} {class_month}')
    file_object.write("\n")
    file_object.close()


def generate_variations():
    file_paths = glob.glob(input_images_dir + '/*.jpg')  # search for all jpg images in the folder
    counter = 1
    # percorre todos os arquivos de imagem da base original
    for filepath in file_paths:
        filename = os.path.basename(filepath)

        # verifica quais fazem parte da base de treinamento
        if filename in filenames_for_training:
            print(counter)  # apenas para controle do tempo de execucao
            index_filename = filenames_for_training.index(filename)
            filename_without_extension = filename.split('.')[0]
            im = cv2.imread(filepath)
            # para cinco níveis (2 a 11, números primos), gera novas imagens
            parameters = (2, 3, 5, 7, 11)
            for i in parameters:
                # gera e salva imagem distorcida
                distort_img = distort(im, i)
                new_filename = filename_without_extension + '-distort-' + str(counter) + '.jpg'
                cv2.imwrite(images_dir + '/' + new_filename, distort_img)
                adiciona_rotulo(classes_month[index_filename], new_filename)
                counter = counter + 1

                # gera e salva imagem esticada
                stretch_img = stretch(im, i)
                new_filename = filename_without_extension + '-distort-' + str(counter) + '.jpg'
                cv2.imwrite(images_dir + '/' + new_filename, stretch_img)
                adiciona_rotulo(classes_month[index_filename], new_filename)
                counter = counter + 1

            # gera e salva imagem em perspectiva
            perspective_img = perspective(im)
            new_filename = filename_without_extension + '-perspective-' + str(counter) + '.jpg'
            cv2.imwrite(images_dir + '/' + new_filename, perspective_img)
            adiciona_rotulo(classes_month[index_filename], new_filename)
            counter = counter + 1


clear_out()
load_train()
generate_variations()
