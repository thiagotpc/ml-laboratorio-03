import os
import glob
from PIL import Image, ImageDraw, ImageFont, ImageOps


# Inicializacao de variáveis
# meses do ano
months = ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro']

# fontes para gerar imagens
fonts_dir = 'src/fonts/'
fonts_extension = '.ttf'
fonts = [
    'AlexBrush-Regular',
    'Allura-Regular',
    'ClickerScript-Regular',
    'Cookie-Regular',
    'DancingScript-VariableFont_wght',
    'DawningofaNewDay-Regular',
    'GreatVibes-Regular',
    'HomemadeApple-Regular',
    'MrDafoe-Regular',
    'MrsSaintDelafield-Regular',
    'Parisienne-Regular',
    'Sacramento-Regular',
    'Satisfy-Regular'
]

# parâmetros para variação das imagens
font_sizes = [36, 48, 60, 72, 84]
draw_angles = [-9, -5, -1, 1, 5, 9]
x_size = 300
y_syze = 150
draw_sizes = [1.2, 0.8]

# cores default
background_color = (255, 255, 255)
forecolor = (0, 0, 0)

# diretorio de saída
output_path = 'out/new-data/'
images_dir = output_path + 'fonts-based'
fullpath_file_train = output_path + 'fonts-based-train.txt'


# registra a imagem e a classe
def adiciona_rotulo(month_name, filename):
    class_number = months.index(month_name)
    file_object = open(fullpath_file_train, 'a')
    file_object.write(f'{filename} {class_number}')
    file_object.write("\n")
    file_object.close()


# limpa pasta de saída
def clear_out():
    print('deleting...')
    if os.path.exists(fullpath_file_train):
        os.remove(fullpath_file_train)
    for file in os.scandir(images_dir):
        if file.name.endswith(".jpg"):
            os.remove(file.path)
    print('done!')


# gera imagens conforme parametros
def generate_months_images():
    counter = 1
    for month in months:
        for font in fonts:
            for font_size in font_sizes:
                for angle in draw_angles:
                    for size in draw_sizes:
                        img = Image.new('RGB', (x_size, y_syze), color=background_color)
                        d = ImageDraw.Draw(img)
                        font_path = fonts_dir + font + fonts_extension
                        fnt = ImageFont.truetype(font_path, font_size)
                        d.text((15, 15), month, font=fnt, fill=forecolor)
                        filename = 'font-based-' + str(counter) + '.jpg'

                        img.rotate(angle, expand=1, fillcolor=background_color).resize(
                            (int(x_size * 1), int(y_syze * size))).save(images_dir + '/' + filename)
                        adiciona_rotulo(month, filename)
                        counter = counter + 1

                        img.rotate(angle, expand=1, fillcolor=background_color).resize(
                            (int(x_size * size), int(y_syze * 1))).save(images_dir + '/' + filename)
                        adiciona_rotulo(month, filename)
                        counter = counter + 1

    print(f'Generated {counter - 1} imagens')


# corta espaco em branco das imagens geradas
def crop_images():
    # cropping white background
    file_paths = glob.glob(images_dir + '/*.jpg')  # search for all jpg images in the folder

    for file_path in file_paths:
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


# main
clear_out()
generate_months_images()
crop_images()
print('Fim!')
