import os
import shutil
import random

ORIGINAL_TRAIN_FILE = 'data/meses/train.txt'
ORIGINAL_TEST_FILE = 'data/meses/test.txt'
ORIGINAL_IMAGES_DIR = 'data/meses/data'

FONTS_TRAIN_FILE = 'out/new-data/fonts-based-train.txt'
FONTS_IMAGES_DIR = 'out/new-data/fonts-based'

THIRD_TRAIN_FILE = 'out/new-data/using-library-train.txt'
THIRD_IMAGES_DIR = 'out/new-data/using-library'

DIRECT_TRAIN_FILE = 'out/new-data/direct-manipulation-train.txt'
DIRECT_IMAGES_DIR = 'out/new-data/direct-manipulation'

NEW_IMAGES_DIR = 'out/new-data/all-images'
C1_TRAIN_FILE = 'out/new-data/train1.txt'
C2_TRAIN_FILE = 'out/new-data/train2.txt'
C3_TRAIN_FILE = 'out/new-data/train3.txt'
C4_TRAIN_FILE = 'out/new-data/train4.txt'


# limpa pasta de saída
def clear_out():
    print('deleting...')
    if os.path.exists(C1_TRAIN_FILE):
        os.remove(C1_TRAIN_FILE)
    if os.path.exists(C2_TRAIN_FILE):
        os.remove(C2_TRAIN_FILE)
    if os.path.exists(C3_TRAIN_FILE):
        os.remove(C3_TRAIN_FILE)
    if os.path.exists(C4_TRAIN_FILE):
        os.remove(C4_TRAIN_FILE)
    for file in os.scandir(NEW_IMAGES_DIR):
        if file.name.endswith(".jpg"):
            os.remove(file.path)
    print('done!')


def merge_train_files():
    print('creating training files...')
    c1 = [ORIGINAL_TRAIN_FILE, FONTS_TRAIN_FILE]
    with open(C1_TRAIN_FILE, 'w') as outfile:
        for fname in c1:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    print('c1 done!')

    c2 = [ORIGINAL_TRAIN_FILE, THIRD_TRAIN_FILE]
    with open(C2_TRAIN_FILE, 'w') as outfile:
        for fname in c2:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    print('c2 done!')

    c3 = [ORIGINAL_TRAIN_FILE, DIRECT_TRAIN_FILE]
    with open(C3_TRAIN_FILE, 'w') as outfile:
        for fname in c3:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    print('c3 done!')

    c4 = [ORIGINAL_TRAIN_FILE, FONTS_TRAIN_FILE, THIRD_TRAIN_FILE, DIRECT_TRAIN_FILE]
    with open(C4_TRAIN_FILE, 'w') as outfile:
        for fname in c4:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    print('c4 done!')
    print('training files created!')


def suffle_lines():
    files = [C1_TRAIN_FILE, C2_TRAIN_FILE, C3_TRAIN_FILE, C4_TRAIN_FILE]
    for file in files:
        lines = open(file).readlines()
        random.shuffle(lines)
        open(file, 'w').writelines(lines)


def merge_all_images_dir():
    print('coping image files...')
    images_dir = [ORIGINAL_IMAGES_DIR, FONTS_IMAGES_DIR, THIRD_IMAGES_DIR, DIRECT_IMAGES_DIR]
    for src_images_dir in images_dir:
        print(ORIGINAL_IMAGES_DIR + '...')
        src_files = os.listdir(src_images_dir)
        for file_name in src_files:
            full_file_name = os.path.join(src_images_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, NEW_IMAGES_DIR)
    print('done!')


merge_train_files()
merge_all_images_dir()
suffle_lines()
