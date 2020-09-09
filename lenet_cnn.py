import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from contextlib import redirect_stdout

from sklearn.metrics import confusion_matrix

from PIL import Image

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D


ORIGINAL_TRAIN_FILE = "data/meses/train.txt"
ORIGINAL_TEST_FILE = "data/meses/test.txt"
NUM_CLASSES = 12

ALL_IMAGES_DIR = 'out/new-data/all-images/'
C1_TRAIN_FILE = 'out/new-data/train1.txt'
C2_TRAIN_FILE = 'out/new-data/train2.txt'
C3_TRAIN_FILE = 'out/new-data/train3.txt'
C4_TRAIN_FILE = 'out/new-data/train4.txt'


#  redimensiona imagens
def resize_data(data, size, convert):
    if convert:
        data_upscaled = np.zeros((data.shape[0], size[0], size[1], 3))
    else:
        data_upscaled = np.zeros((data.shape[0], size[0], size[1]))
    for i, img in enumerate(data):
        large_img = cv2.resize(img, dsize=(size[1], size[0]), interpolation=cv2.INTER_CUBIC)
        data_upscaled[i] = large_img
    return data_upscaled


#  carrega imagens
def load_images(image_paths, convert=False):
    x = []
    y = []
    for image_path in image_paths:
        path, label = image_path.split(' ')
        path = ALL_IMAGES_DIR + path
        if convert:
            image_pil = Image.open(path).convert('RGB')
        else:
            image_pil = Image.open(path).convert('L')
        img = np.array(image_pil, dtype=np.uint8)
        x.append(img)
        y.append([int(label)])
    x = np.array(x)
    y = np.array(y)
    if np.min(y) != 0:
        y = y - 1
    return x, y


#  carrega conjunto de treino e validação
def load_dataset(train_file, test_file, resize, convert=False, size=(224, 224)):

    # Abre arquivo de treinamento e lê conteúdo
    print("Loading training set...")
    arq = open(train_file, 'r')
    texto = arq.read()
    train_paths = texto.split('\n')
    train_paths.remove('')
    x_train, y_train = load_images(train_paths, convert)

    # Abre arquivo de validação e lê conteúdo
    print("Loading testing set...")
    arq = open(test_file, 'r')
    texto = arq.read()
    test_paths = texto.split('\n')
    test_paths.remove('')
    x_test, y_test = load_images(test_paths, convert)

    # verifica necessidade e faz resize
    if resize:
        print("Resizing images...")
        x_train = resize_data(x_train, size, convert)
        x_test = resize_data(x_test, size, convert)
    if not convert:
        x_train = x_train.reshape(x_train.shape[0], size[0], size[1], 1)
        x_test = x_test.reshape(x_test.shape[0], size[0], size[1], 1)

    return (x_train, y_train), (x_test, y_test)


#  gera etiquetas para grafico
def generate_labels(x_test, y_test):
    labels = []
    for i in range(len(x_test)):
        labels.append(y_test[i][0])
    return labels


#  normalizacao das imagens
def normalize_images(x):
    x = x.astype('float32')
    x /= 255
    return x


#  converte em vetor
def convert_vector(x, num_classes):
    return keras.utils.to_categorical(x, num_classes)


#  treina o modelo
def fit_model(model, x_train, y_train, x_test, y_test, epochs, batch_size=128, verbose=1):
    return model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
                     verbose=verbose)


#  gera matriz de confusao
def get_confusion_matrix(model, x_test, labels):
    pred = []
    y_pred = model.predict_classes(x_test)
    for i in range(len(x_test)):
        pred.append(y_pred[i])
    return confusion_matrix(labels, pred)


#  gera e salva gráficos
def plot_graphs(history, title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.figure(figsize=(12, 9))
    plt.plot(epochs, acc, 'b', label='Acurácia do treinamento')
    plt.plot(epochs, val_acc, 'r', label='Acurácia da validação')
    plt.title('Acurácia do treinamento e validação')
    plt.legend()
    plt.savefig('out/graphs/' + title + '_acc.png')
    plt.figure(figsize=(12, 9))
    plt.plot(epochs, loss, 'b', label='Perda do treinamento')
    plt.plot(epochs, val_loss, 'r', label='Perda da validação')
    plt.title('Perda do treinamento e validação')
    plt.legend()
    plt.savefig('out/graphs/' + title + '_loss.png')


#  arredondamento
def round_float(value):
    return float("{:.5f}".format(value))


#  retorna o timestamp atual
def get_time():
    return time.time()


#  calcula a diferenca de timestamps
def get_time_diff(start_time):
    end_time = time.time()
    return round_float(end_time - start_time)


# Modelo LeNet5
def lenet5(img_rows, img_cols, num_classes):
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(img_rows, img_cols, 3),
                     padding="same"))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(Flatten())
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# Modelo para Outra CNN
def other_cnn(img_rows, img_cols, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(8, 8), activation='relu', input_shape=(img_rows, img_cols, 3)))
    model.add(Conv2D(128, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.20))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# Execucao da LeNet5
def execute_lenet5(img_cols, img_rows, epochs, conj=0):
    # Star Time
    checkpoint_time = get_time()

    # Título do Experimento: tipo da rede + conjunto de treino + largura + altura + epocas
    title = 'lenet5_c' + str(conj) + '_' + str(img_cols) + 'x' + str(img_rows) + '_' + str(epochs) + 'epochs'

    # Loading Inital Data
    if conj == 1:
        train_file = C1_TRAIN_FILE
    elif conj == 2:
        train_file = C2_TRAIN_FILE
    elif conj == 3:
        train_file = C3_TRAIN_FILE
    elif conj == 4:
        train_file = C4_TRAIN_FILE
    else:
        train_file = ORIGINAL_TRAIN_FILE

    (x_train, y_train), (x_test, y_test) = load_dataset(train_file, ORIGINAL_TEST_FILE, resize=True,
                                                        convert=True,
                                                        size=(img_rows, img_cols))

    # Normalize images
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)

    # Generating Labels for Confusion Matrix
    labels = generate_labels(x_test, y_test)

    # Convert class vectros to binary class matrices
    y_train = convert_vector(y_train, NUM_CLASSES)
    y_test = convert_vector(y_test, NUM_CLASSES)

    # Get LeNet 5 Model
    model = lenet5(img_rows, img_cols, NUM_CLASSES)

    # Saving Model
    with open('out/reports/' + title + '_model.txt', 'w') as f:
        with redirect_stdout(f):
            # Printing Summary
            model.summary()

    # Compiling Model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=["accuracy"])

    # Trainning model
    history = fit_model(model, x_train, y_train, x_test, y_test, epochs)

    # Getting Score
    score = model.evaluate(x_test, y_test, verbose=0)

    # Saving Outputs
    # Saving Model
    with open('out/reports/' + title + '_results.txt', 'w') as f:
        with redirect_stdout(f):
            # Loss
            print('Loss:', score[0])

            # Accuracy
            print('Accuracy:', score[1])

            # Confusion Matrix
            cm = get_confusion_matrix(model, x_test, labels)
            print(f'Confusion Matrix: \n', cm)

            # Execution Time
            print(f'Execution Time: {get_time_diff(checkpoint_time)}')

    # Gera graficos
    plot_graphs(history, title)


# Executa Outra rede CNN
def execute_other_cnn(img_cols, img_rows, epochs, conj=0):

    # Star Time
    checkpoint_time = get_time()

    # Título do Experimento: tipo da rede + conjunto de treino + largura + altura + epocas
    title = 'other_cnn_c' + str(conj) + '_' + str(img_cols) + 'x' + str(img_rows) + '_' + str(epochs) + 'epochs'

    # Loading Inital Data
    if conj == 1:
        train_file = C1_TRAIN_FILE
    elif conj == 2:
        train_file = C2_TRAIN_FILE
    elif conj == 3:
        train_file = C3_TRAIN_FILE
    elif conj == 4:
        train_file = C4_TRAIN_FILE
    else:
        train_file = ORIGINAL_TRAIN_FILE

    # Loading Inital Data
    (x_train, y_train), (x_test, y_test) = load_dataset(train_file, ORIGINAL_TEST_FILE, resize=True,
                                                        convert=True,
                                                        size=(img_rows, img_cols))

    # Normalize images
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)

    # Generating Labels for Confusion Matrix
    labels = generate_labels(x_test, y_test)

    # Convert class vectros to binary class matrices
    y_train = convert_vector(y_train, NUM_CLASSES)
    y_test = convert_vector(y_test, NUM_CLASSES)

    # Get Other CNN Model
    model = other_cnn(img_rows, img_cols, NUM_CLASSES)

    # Saving Model
    with open('out/reports/' + title + '_model.txt', 'w') as f:
        with redirect_stdout(f):
            # Printing Summary
            model.summary()

    # Compiling Model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=["accuracy"])

    # Trainning model
    history = fit_model(model, x_train, y_train, x_test, y_test, epochs)

    # Getting Score
    score = model.evaluate(x_test, y_test, verbose=0)

    # Saving Outputs
    # Saving Model
    with open('out/reports/' + title + '_results.txt', 'w') as f:
        with redirect_stdout(f):
            # Loss
            print('Loss:', score[0])

            # Accuracy
            print('Accuracy:', score[1])

            # Confusion Matrix
            cm = get_confusion_matrix(model, x_test, labels)
            print(f'Confusion Matrix: \n', cm)

            # Execution Time
            print(f'Execution Time: {get_time_diff(checkpoint_time)}')

    # Graphs
    plot_graphs(history, title)


# Sequencias de Execucoes
# 4 dimensoes, 3 tamanhos de epocas, 5 conjuntos de treinamento diferentes = 60 experimentos
arr_dimensoes = [(32, 32), (64, 64), (60, 17), (120, 34)]
arr_epochs = (10, 20, 40)
arr_conj = (0, 1, 2, 3, 4)

for largura, altura in arr_dimensoes:
    for num_epochs in arr_epochs:
        for c in arr_conj:
            execute_lenet5(largura, altura, num_epochs, c)
            execute_other_cnn(largura, altura, num_epochs, c)
