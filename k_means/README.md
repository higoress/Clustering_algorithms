# Data Clustering: K MEANS

## Requerimentos
    1 - bibliotecas python: numpy e pandas
        1.1 - podem ser instalas fazendo: pip3 install numpy, pandas 

## Modo de Usar
    1 - execute o código: python3 k_means.py PATH_NAME K
        1.1 - eg.: python3 k_means.py data/iris_unlabelled.csv 3 ONDE 
            data/iris.csv é o path + o nome do seu arquivo é K o número de clusters;
    2 - para mais opções de execução faça: python3 k_means.py --help
    3 - para executar o exemplo do dataset Iris: python3 k_means.py --example
    4 - para executar o algoritmo interativamente python3 k_means.py --interactive

    Nota: O algoritmo funciona somente para dados numéricos e sem missing values.

## Outras Informações
    - Uma versão do google Colab também está disponível, contendo inclusive visualização dos 
    centroids em pairplot. Para saber mais acesse:
        [Clique aqui](https://colab.research.google.com/drive/1lOMXif-petzSWLwzoV8jUMDXYb55702V?usp=sharing)

    - Esta implementação do K Means possui mais opções, como é possível escolher a norma usada
    para calcular as distâncias opções (L1 - manhattan, L2 - euclidiana), o algoritmo parar 
    quando não houver alterações nos clusters durante a última interação e escolher o número
    máximo de interações que ele deve executar, caso não convirja. Para saber mais, veja na 
    documentação do código. 


