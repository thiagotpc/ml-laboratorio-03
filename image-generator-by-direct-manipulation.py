import os
import glob
from random import randrange

from PIL import Image, ImageOps

filenames_for_training = []
classes_month = []

# entrada
input_filename_train = 'data/meses/train.txt'
input_images_dir = 'data/meses/data'

# diretorio de saída
output_path = 'out/new-data/'
images_dir = output_path + 'direct-manipulation'
fullpath_file_train = output_path + 'direct-manipulation-train.txt'

background_color = (255, 255, 255)


# limpa pasta de saída
def clear_out():
    print('deleting...')
    if os.path.exists(fullpath_file_train):
        os.remove(fullpath_file_train)
    for file in os.scandir(images_dir):
        if file.name.endswith(".jpg"):
            os.remove(file.path)
    print('done!')


# carrega imagens do treinamento original
def load_train():
    f = open(input_filename_train, "r")
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

            # rotate
            angles = (-9, -5, -1, 1, 5, 9)
            img = Image.open(filepath)
            img2 = img.convert('RGBA')
            for angle in angles:
                new_img = img2.rotate(angle, expand=1)
                fff = Image.new('RGBA', new_img.size, (255,) * 4)
                out = Image.composite(new_img, fff, new_img)
                new_filename = filename_without_extension + '-angle-' + str(counter) + '.jpg'
                out.convert(img.mode).save(images_dir + '/' + new_filename)
                adiciona_rotulo(classes_month[index_filename], new_filename)
                counter = counter + 1

            # resize desproporcional
            zoom = (.75, 1.25)
            for mult in zoom:
                width, height = img.size
                new_img = img.resize((int(width * mult), int(height * 1)))
                new_filename = filename_without_extension + '-resize-' + str(counter) + '.jpg'
                new_img.save(images_dir + '/' + new_filename)
                adiciona_rotulo(classes_month[index_filename], new_filename)
                counter = counter + 1

                new_img = img.resize((int(width * 1), int(height * mult)))
                new_filename = filename_without_extension + '-resize-' + str(counter) + '.jpg'
                new_img.save(images_dir + '/' + new_filename)
                adiciona_rotulo(classes_month[index_filename], new_filename)
                counter = counter + 1

            # booth resize and angle
            img3 = img.convert('RGBA')
            mult_1_rand = randrange(-30, 30)
            mult_2_rand = randrange(-30, 30)
            angle_rand = randrange(-10, 10)
            mult_1 = (100+mult_1_rand)/100
            mult_2 = (100+mult_2_rand)/100
            width, height = img.size
            new_img = img3.rotate(angle_rand, expand=1).resize((int(width * mult_1), int(height * mult_2)))
            fff = Image.new('RGBA', new_img.size, (255,) * 4)
            out = Image.composite(new_img, fff, new_img)
            new_filename = filename_without_extension + '-angle-resize-' + str(counter) + '.jpg'
            out.convert(img.mode).save(images_dir + '/' + new_filename)
            adiciona_rotulo(classes_month[index_filename], new_filename)
            counter = counter + 1


# corta espaco em branco das imagens geradas
def crop_images():
    # cropping white background
    file_paths = glob.glob(images_dir + '/*-angle-*.jpg')  # search for all jpg images rotated in the folder

    for file_path in file_paths:
        print(file_path)
        image = Image.open(file_path)
        image.load()

        # remove alpha channel
        invert_im = image.convert("RGB")

        # invert image (so that white is 0)
        invert_im = ImageOps.invert(invert_im)
        image_box = invert_im.getbbox()

        cropped = image.crop(image_box)
        cropped.save(file_path)

    print('Images cropped!')


clear_out()
load_train()
generate_variations()
crop_images()
