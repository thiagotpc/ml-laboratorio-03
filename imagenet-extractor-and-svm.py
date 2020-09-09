import numpy as np

from contextlib import redirect_stdout

from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import accuracy_score
from sklearn import svm

from keras.preprocessing import image
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_preprocess


# INICIALIZAÇÃO DE VARIÁVEIS
ORIGINAL_TRAIN_FILE = "data/meses/train.txt"
ORIGINAL_TEST_FILE = "data/meses/test.txt"
NUM_CLASSES = 12

ALL_IMAGES_DIR = 'out/new-data/all-images/'
C1_TRAIN_FILE = 'out/new-data/train1.txt'
C2_TRAIN_FILE = 'out/new-data/train2.txt'
C3_TRAIN_FILE = 'out/new-data/train3.txt'
C4_TRAIN_FILE = 'out/new-data/train4.txt'

OUTPUT_SVM_DIR = 'out/svm/'
OUTPUT_FILE_TEST = OUTPUT_SVM_DIR + 'test.svm'
OUTPUT_FILE_TRAIN_C0 = OUTPUT_SVM_DIR + 'c0_train.svm'
OUTPUT_FILE_TRAIN_C1 = OUTPUT_SVM_DIR + 'c1_train.svm'
OUTPUT_FILE_TRAIN_C2 = OUTPUT_SVM_DIR + 'c2_train.svm'
OUTPUT_FILE_TRAIN_C3 = OUTPUT_SVM_DIR + 'c3_train.svm'
OUTPUT_FILE_TRAIN_C4 = OUTPUT_SVM_DIR + 'c4_train.svm'
IMG_ROWS = 32
IMG_COLS = 32


# arredondamento
def round_float(value):
    return float("{:.5f}".format(value))


# extrai características
def extract_features(input_extract_file, output_extract_file, img_rows, img_cols, dir_dataset):
    file_input = open(input_extract_file, 'r')
    input_extract = file_input.readlines()
    file_input.close()
    output = open(output_extract_file, 'w')
    model = Xception(weights='imagenet', include_top=False)
    for i in input_extract:
        sample_name, sample_class = i.split()
        img_path = dir_dataset + sample_name
        img = image.load_img(img_path, target_size=(img_rows, img_cols))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = xception_preprocess(img_data)
        model_features = model.predict(img_data)
        features_np = np.array(model_features)
        features_np = features_np.flatten()
        output.write(sample_class + ' ')
        for j in range(features_np.size):
            output.write(str(j + 1) + ':' + str(features_np[j]) + ' ')
        output.write('\n')
    output.close()


# executa SVM
def do_svm(input_trained_svm, input_test_svm, conj_train=0):
    x_train, y_train = load_svmlight_file(input_trained_svm)
    x_test, y_test = load_svmlight_file(input_test_svm)
    x_train = x_train.toarray()
    x_test = x_test.toarray()
    classificator = svm.SVC()
    classificator.fit(x_train, y_train)
    predict = classificator.predict(x_test)
    f1_score = round_float(sklearn_f1_score(y_test, predict, labels=np.unique(predict), average='weighted'))
    accuracy = round_float(accuracy_score(y_test, predict))
    cm = confusion_matrix(y_test, predict)
    # Saving Results
    title = 'svm_c' + str(conj_train)
    with open('out/reports/' + title + '_results.txt', 'w') as f:
        with redirect_stdout(f):
            print(f'Accuracy:  {accuracy}')
            print(f'F1Score:  {f1_score}')
            # Confusion Matrix
            print(f'Confusion Matrix: \n', cm)


# Train
train_files = [(ORIGINAL_TRAIN_FILE, OUTPUT_FILE_TRAIN_C0), (C1_TRAIN_FILE, OUTPUT_FILE_TRAIN_C1), (C2_TRAIN_FILE, OUTPUT_FILE_TRAIN_C2), (C3_TRAIN_FILE, OUTPUT_FILE_TRAIN_C3), (C4_TRAIN_FILE, OUTPUT_FILE_TRAIN_C4)]
for input_file, output_file in train_files:
    print(input_file)
    extract_features(input_file, output_file, IMG_ROWS, IMG_COLS, ALL_IMAGES_DIR)

# Test
print('extraindo caracteristicas para validação')
extract_features(ORIGINAL_TEST_FILE, OUTPUT_FILE_TEST, IMG_ROWS, IMG_COLS, ALL_IMAGES_DIR)
print('done!')

svm_trained_files = [OUTPUT_FILE_TRAIN_C0, OUTPUT_FILE_TRAIN_C1, OUTPUT_FILE_TRAIN_C2, OUTPUT_FILE_TRAIN_C3, OUTPUT_FILE_TRAIN_C4]
svm_test_file = OUTPUT_FILE_TEST
print('validando!')
conj = 0
for input_trained in svm_trained_files:
    print(input_trained)
    do_svm(input_trained, svm_test_file, conj)
    conj = conj + 1
print('done!')
