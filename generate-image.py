import os
import glob
from PIL import Image, ImageDraw, ImageFont, ImageOps

months = ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro']

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

font_sizes = [24, 28, 32, 36]

draw_angles = [1, 2, 3, 0, -1, -2, -3]
x_size = 300
y_syze = 10
draw_sizes = [1, 1.1, 0.9]

background_color = (255, 255, 255)
forecolor = (0, 0, 0)

counter = 1

for file in os.scandir('out/new-data'):
    if file.name.endswith(".png"):
        os.unlink(file.path)

for month in months:
    for font in fonts:
        for font_size in font_sizes:
            for angle in draw_angles:
                for size in draw_sizes:
                    img = Image.new('RGB', (x_size, y_syze), color=background_color)
                    d = ImageDraw.Draw(img)
                    font_path = ''
                    font_path = fonts_dir + font + fonts_extension
                    fnt = ImageFont.truetype(font_path, font_size)
                    d.text((5, 5), month, font=fnt, fill=forecolor)
                    img.rotate(angle, expand=1, fillcolor=background_color).resize((int(x_size*size),int(y_syze*size))).save('out/new-data/' + str(counter) + '.png')
                    counter = counter + 1

print(f'Generated {counter-1} imagens')

# cropping white background
filePaths = glob.glob('out/new-data/*.png')  # search for all png images in the folder

for filePath in filePaths:
    image = Image.open(filePath)
    image.load()
    imageSize = image.size

    # remove alpha channel
    invert_im = image.convert("RGB")

    # invert image (so that white is 0)
    invert_im = ImageOps.invert(invert_im)
    imageBox = invert_im.getbbox()

    cropped = image.crop(imageBox)
    # print(filePath, "Size:", imageSize, "New Size:", imageBox)
    cropped.save(filePath)

print('Fim!')
