# ml-laboratorio-03
Laboratório 03 da disciplina de Aprendizagem de Máquina do PPGInf da UFPR

## Objetivo

Dado um conjunto de imagens de amostras para treinamento e validação, executar experimentos com 
Convolutional Neural Networks (CNNs). As amostras são imagens de texto manuscrito para os meses do ano
no idioma português (pt-br).

Usar técnicas de Data-Augmentation para melhor os resultados.

## Enunciado

> ## CNN
>
> Para esse laboratório considere a base de dados de meses do ano (12 classes) apresentado nas práticas de Deep Learning.
>
> 1. Implemente funções para aumentar o número de amostras do conjunto de TREINAMENTO (Data Augmentation);
>
> 2. Implemente duas redes neurais convolucionais: (a) LeNet 5 [Yann LeCun (1998)]; (b) CNN de sua escolha;
>
> 3. Escreva um breve relatório que:
>    a) Descreva as CNNs utilizadas e as funções de Data Augmentation;
>    b) Compare o desempenho dessas redes variando os diferentes parâmetros apresentados em aula. Realize treinamentos com e sem Data Augmentation. Analise os resultados obtidos nos diferentes experimentos apresentando suas conclusões (apresente gráficos e matriz de confusão);
>    c) Repita os experimentos e análises utilizando 2 redes pré-treinadas na ImageNet (Transfer Learning/Fine-Tuning). Utilize essas redes para gerar vetores de características, realizando a classificação em outro classificador (Ex: SVM).
>
> O relatório reportando seus experimentos deve entregue em formato PDF.


## Estrutura de Arquivos e Pastas

- 📂data
  - 📂meses
    - data
  - test.txt
  - train.txt
- 📂src
- 📂out
  - 📂new-data
    - 📂fonts-based
    - 📂direct-manipulation
    - 📂using-library
    - 📂all-imagens
    train1.txt
    train2.txt
    train3.txt
    train4.txt
  - 📂svm
  - 📂graphs
  - 📂reports
- image-generator-by-fonts.py
- image-generator-by-direct-manipulation.py
- image-generator-by-library.py
- lenet-cnn.py
- imagenet-extrator-and-svm.py
- 📦 augment.py
- 📦 warp_mls.py
- relatorio.pdf

### Data
A pasta **data** contém a base de treinamento e testes, fornecidas pelo professor.

Em data\\meses\\data estão as imagens e em meses\\*.txt a rotulagem para base de testes e validação, já divididos.

### Src
A pasta src\fonts contém os arquivos de fontes manuscritas, escolhidas no Google Fonts.

### Out
A pasta **out** armazena os arquivos gerados pelos scripts do programa.

Na pasta **new-data** ficarão as imagens de data-augmentation e os arquivos de texto para rotulação das imagens geraadas e das novas da bases de treinamento.

Na pasta **svm** ficarão os arquivos de extração de características para treino e validação da SVM.

Na pasta **reports** ficarão as saídas dos experimentos, como modelo e matriz de confusão.

Na pasta **graphs** ficarão os gráficos para acurácia e perda a cada época, gerados durante os experimentos.

Devido a grande quantidade de imagens o projeto não inclui as imagens geradas na execução dos experimentos. Também não inclui os arquivos de vetores de características pois somam mais de 1GB.

### Scripts
- **image-generator-by-fonts.py**, gera imagens por fontes.
- **image-generator-by-direct-manipulation.py**, gera imagens modificando a base com funções de rotação e resize.
- **image-generator-by-library.py**, gera imagens usando biblioteca [Text-Image-Augmentation-python](https://github.com/RubanSeven/Text-Image-Augmentation-python).
- **merge-training-data.py**, junta os rótulos e imagens para formar novos conjuntos de treinamento. 
- **lenet-cnn.py**, roda os experimentos com LeNet5 e Outra CNN.
- **imagenet-extrator-and-svm.py**, roda os experimentos com extração de características da ImageNet e SVM.

### Outros
- **augment.py** e **warp_mls.py**, fazem parte da biblioteca [Text-Image-Augmentation-python](https://github.com/RubanSeven/Text-Image-Augmentation-python);
- **relatorio.pdf** descreve os experimentos e comenta os resultados.

## Principais Dependências
- Biblioteca keras e scikit-learn para execução dos treinos e validação;
- Biblioteca cv2 e PIL para manipulação das imagens;
- Biblioteca [Text-Image-Augmentation-python](https://github.com/RubanSeven/Text-Image-Augmentation-python), para um dos conjuntos de *data augmentation*;

## Modo de Funcionamento
Execute os scripts nesta ordem: image-generators, merge, experimentos.
Verifique as saídas na pasta out, principalmente graphs e reports.
