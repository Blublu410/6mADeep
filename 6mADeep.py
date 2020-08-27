
# import matplotlib as mpl
# mpl.use('Agg')

from keras.optimizers import SGD
from group_norm import GroupNormalization
import random
import pandas as pd
from keras import regularizers
from keras.metrics import binary_accuracy
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os, sys,  argparse
from keras import losses
import tensorflow as tf


from scipy import interp
from keras.layers import *
from keras.models import *

import itertools
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection  import train_test_split

from keras import backend as K
from numpy import sort
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel




kmers = 6
MAX_LEN_H = 41 - kmers + 1
MAX_LEN =   MAX_LEN_H
NB_WORDS = 4 ** kmers + 1
EMBEDDING_DIM = 100
embedding_matrix = np.load('embeddingMatrix6.npy')
folds = 10





def sentence2word(str_set):
    word_seq = []
    for sr in str_set:
        tmp = []
        for i in range(len(sr) - kmers + 1):
            if ('N' in sr[i:i + kmers]):
                tmp.append('null')
            else:
                tmp.append(sr[i:i + kmers])
        word_seq.append(' '.join(tmp))
    return word_seq


def word2num(wordseq, tokenizer, MAX_LEN):
    sequences = tokenizer.texts_to_sequences(wordseq)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN)
    return numseq


def sentence2num(str_set, tokenizer, MAX_LEN):
    wordseq = sentence2word(str_set)
    numseq = word2num(wordseq, tokenizer, MAX_LEN)
    return numseq


def get_tokenizer():
    f = ['A', 'C', 'G', 'T']
    c = itertools.product(f, f, f ,f ,f ,f)
    res = []
    for i in c:
        temp = i[0 ] +i[1 ] +i[2 ] +i[3 ] +i[4 ] +i[5]
        res.append(temp)
    res = np.array(res)
    tokenizer = Tokenizer(num_words=NB_WORDS)
    tokenizer.fit_on_texts(res)
    acgt_index = tokenizer.word_index
    acgt_index['null'] = 0
    return tokenizer





def get_data(trainData ,maxLen):
    tokenizer = get_tokenizer()
    X_en = sentence2num(trainData ,tokenizer ,maxLen)
    return X_en


def analyze(temp, OutputDir):


    trainning_result, validation_result, testing_result = temp;

    file = open(OutputDir + '/performance.txt', 'w')

    index = 0
    for x in [trainning_result, validation_result, testing_result]:

        title = ''

        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'

        index += 1;

        file.write(title + 'results\n')

        for j in ['sn', 'sp', 'acc', 'MCC', 'AUC', 'precision', 'F1', 'lossValue']:

            total = []

            for val in x:
                total.append(val[j])

            file.write(j + ' : mean : ' + str(np.mean(total)) + ' std : ' + str(np.std(total)) + '\n')

        file.write('\n\n______________________________\n')
    file.close();

    index = 0

    for x in [trainning_result, validation_result, testing_result]:

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0

        for val in x:
            tpr = val['tpr']
            fpr = val['fpr']
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i + 1, roc_auc))

            i += 1

        print;

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")

        title = ''

        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'

        plt.savefig(OutputDir + '/' + title + 'ROC.png')
        plt.close('all');

        index += 1;


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def calculate(sequence):
    X = []
    dictNum = {'A': 0, 'T': 0, 'C': 0, 'G': 0};

    for i in range(len(sequence)):

        if sequence[i] in dictNum.keys():
            dictNum[sequence[i]] += 1;
            X.append(dictNum[sequence[i]] / float(i + 1));

    return np.array(X)


def dataProcessing(path):
    data = pd.read_csv(path);
    alphabet = np.array(['A', 'G', 'T', 'C'])
    X = [];
    for line in data['data']:

        line = list(line.strip('\n'));
        scoreSequence = calculate(line);

        seq = np.array(line, dtype='|U1').reshape(-1, 1);
        seq_data = []

        for i in range(len(seq)):
            if seq[i] == 'A':
                seq_data.append([1, 0, 0, 0])
            if seq[i] == 'T':
                seq_data.append([0, 1, 0, 0])
            if seq[i] == 'C':
                seq_data.append([0, 0, 1, 0])
            if seq[i] == 'G':
                seq_data.append([0, 0, 0, 1])

        X.append(np.array(seq_data));

    X = np.array(X);
    y = np.array(data['label'], dtype=np.int32);
    print(X ,y)

    return X, y;


def prepareData(PositiveCSV, NegativeCSV):
    Positive_X, Positive_y = dataProcessing(PositiveCSV);
    Negitive_X, Negitive_y = dataProcessing(NegativeCSV);

    return Positive_X, Positive_y, Negitive_X, Negitive_y


def shuffleData(X, y):
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X = X[index]
    y = y[index]
    return X, y;




def getMode():
    input_shape = (MAX_LEN,)
    inputs = Input(shape=input_shape)
    emb_inputs = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)(inputs)

    convLayer = Conv1D(filters=16, kernel_size=4, activation='elu', input_shape=input_shape,
                       kernel_regularizer=regularizers.l2(1e-4), bias_regularizer=regularizers.l2(1e-4))(emb_inputs);
    normalizationLayer = GroupNormalization(groups=4, axis=-1)(convLayer)
    poolingLayer = MaxPooling1D(pool_size=4)(normalizationLayer)
    flattenLayer = Flatten()(poolingLayer)

    dropoutLayer = Dropout(0.25)(flattenLayer)

    denseLayer = Dense(32, activation='elu', kernel_regularizer=regularizers.l2(1e-4),
                       bias_regularizer=regularizers.l2(1e-4))(dropoutLayer)
    outLayer = Dense(1, activation='sigmoid')(denseLayer)

    model = Model(inputs=inputs, outputs=outLayer)
    model.compile(loss='binary_crossentropy', optimizer=SGD(momentum=0.95, lr=0.005), metrics=[binary_accuracy]);

    print(model.summary())

    return model;


def calculateScore(X, y, model):
    score = model.evaluate(X, y)
    pred_y = model.predict(X)

    accuracy = score[1];

    tempLabel = np.zeros(shape=y.shape, dtype=np.int32)

    for i in range(len(y)):
        if pred_y[i] < 0.5:
            tempLabel[i] = 0;
        else:
            tempLabel[i] = 1;
    confusion = confusion_matrix(y, tempLabel)
    TN, FP, FN, TP = confusion.ravel()

    sensitivity = recall_score(y, tempLabel)
    specificity = TN / float(TN + FP)
    MCC = matthews_corrcoef(y, tempLabel)

    F1Score = (2 * TP) / float(2 * TP + FP + FN)
    precision = TP / float(TP + FP)

    pred_y = pred_y.reshape((-1,))

    ROCArea = roc_auc_score(y, pred_y)
    fpr, tpr, thresholds = roc_curve(y, pred_y)
    lossValue = None;

    print(y.shape)
    print(pred_y.shape)

    y_true = tf.convert_to_tensor(y, np.float32)
    y_pred = tf.convert_to_tensor(pred_y, np.float32)

    with tf.Session():
        lossValue = losses.binary_crossentropy(y_true, y_pred).eval()

    return {'sn': sensitivity, 'sp': specificity, 'acc': accuracy, 'MCC': MCC, 'AUC': ROCArea, 'precision': precision,
            'F1': F1Score, 'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'lossValue': lossValue}





##########################################################################





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="deep learning 6mA analysis in rice genome")

    parser.add_argument("--output", type=str, help="output folder", required=True)
    parser.add_argument("--positive", type=str, help="positive 6mA csv", required=True)
    parser.add_argument("--negative", type=str, help="negative 6mA csv", required=True)

    args = parser.parse_args()

    PositiveCSV = os.path.abspath(args.positive)
    NegativeCSV = os.path.abspath(args.negative)
    OutputDir = os.path.abspath(args.output)

    if not os.path.exists(OutputDir):
        print("The OutputDir not exist! Error\n")
        sys.exit()
    if not os.path.exists(PositiveCSV) or not os.path.exists(NegativeCSV):
        print("The csv data not exist! Error\n")
        sys.exit()

    # Read data
    data = pd.read_csv(PositiveCSV);
    Positive_X = get_data(data['data'], MAX_LEN_H  )# kmer
    Positive_y = np.array(data['label'], dtype=np.int32);
    data = pd.read_csv(NegativeCSV);
    Negitive_X = get_data(data['data'], MAX_LEN_H)
    Negitive_y = np.array(data['label'], dtype=np.int32);

    X = np.concatenate((Positive_X, Negitive_X))
    Y = np.concatenate((Positive_y, Negitive_y))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    thresholds = sort(model.feature_importances_)

    for thresh in thresholds:
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        selection_model = XGBClassifier()
        selection_model.fit(select_X_train, y_train)
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)

        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))



    b = sorted(enumerate(model.feature_importances_), key=lambda x: x[1], reverse=True)
    a = np.array(b)[:, 0][0:MAX_LEN].astype(np.uint8)

    Positive_X = Positive_X[:,a]
    Negitive_X = Negitive_X[:,a]

    random.shuffle(Positive_X);
    random.shuffle(Negitive_X);

    Positive_X_Slices = chunkIt(Positive_X, folds);
    Positive_y_Slices = chunkIt(Positive_y, folds);

    Negative_X_Slices = chunkIt(Negitive_X, folds);
    Negative_y_Slices = chunkIt(Negitive_y, folds);


    trainning_result = []
    validation_result = []
    testing_result = []

    for test_index in range(folds):

        test_X = np.concatenate((Positive_X_Slices[test_index], Negative_X_Slices[test_index]))
        test_y = np.concatenate((Positive_y_Slices[test_index], Negative_y_Slices[test_index]))

        validation_index = (test_index + 1) % folds;

        valid_X = np.concatenate((Positive_X_Slices[validation_index], Negative_X_Slices[validation_index]))
        valid_y = np.concatenate((Positive_y_Slices[validation_index], Negative_y_Slices[validation_index]))

        start = 0;

        for val in range(0, folds):
            if val != test_index and val != validation_index:
                start = val;
                break;

        train_X = np.concatenate((Positive_X_Slices[start], Negative_X_Slices[start]))
        train_y = np.concatenate((Positive_y_Slices[start], Negative_y_Slices[start]))

        for i in range(0, folds):
            if i != test_index and i != validation_index and i != start:
                tempX = np.concatenate((Positive_X_Slices[i], Negative_X_Slices[i]))
                tempy = np.concatenate((Positive_y_Slices[i], Negative_y_Slices[i]))

                train_X = np.concatenate((train_X, tempX))
                train_y = np.concatenate((train_y, tempy))

        test_X, test_y = shuffleData(test_X, test_y);
        valid_X, valid_y = shuffleData(valid_X, valid_y)
        train_X, train_y = shuffleData(train_X, train_y);

        model = getMode();

        early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience=30)
        model_check = ModelCheckpoint(filepath=OutputDir + "/model" + str(test_index + 1) + ".h5",
                                      monitor='val_binary_accuracy', save_best_only=True)
        reduct_L_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20)

        history = model.fit(train_X, train_y, batch_size=32, epochs=100, validation_data=(valid_X, valid_y),
                            callbacks=[model_check, early_stopping, reduct_L_rate]);

        trainning_result.append(calculateScore(train_X, train_y, model));
        validation_result.append(calculateScore(valid_X, valid_y, model));
        testing_result.append(calculateScore(test_X, test_y, model));


        K.clear_session()
        tf.reset_default_graph()

    temp_dict = (trainning_result, validation_result, testing_result)
    analyze(temp_dict, OutputDir);


