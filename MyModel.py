import numpy as np
np.random.seed(123)

import os
from collections import Counter

from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.neighbors import KDTree
import tensorflow as tf
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical


WORD2VECPATH    = "../data/class_vectors.npy"
DATAPATH        = "../data/zeroshot_data.pkl"
MODELPATH       = "../model/"


def build_model():
    model  = Sequential()
    model.add(tf.keras.applications.NASNetMobile(
    include_top=False,
    input_shape=(224,224,3),
    weights="imagenet"))
    model.add(Flatten())
    model.add(Dense(131072, activation='relu'))
    model.add(Dense(NUM_ATTR, activation='relu'))
    model.add(Dense(NUM_CLASS, activation='softmax', trainable=False))#, kernel_initializer=custom_kernel_init))
    return model


def train_model(model, train_ds, val_ds):
    adam = Adam(lr=5e-5)
    model.compile(loss      = 'categorical_crossentropy',
                  optimizer = adam,
                  metrics   = ['categorical_accuracy', 'top_k_categorical_accuracy'])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCH
    )

    print("model training is completed.")
    return history
    
def load_data():
    """read data, create datasets"""
    # READ DATA
    with gzip.GzipFile(DATAPATH, 'rb') as infile:
        data = cPickle.load(infile)


    # ONE-HOT-ENCODE DATA
    label_encoder   = LabelEncoder()
    label_encoder.fit(train_classes)

    training_data = [instance for instance in data if instance[0] in train_classes]
    zero_shot_data = [instance for instance in data if instance[0] not in train_classes]
    # SHUFFLE TRAINING DATA@
    np.random.shuffle(training_data)

    ### SPLIT DATA FOR TRAINING
    train_size  = 300
    train_data  = list()
    valid_data  = list()
    for class_label in train_classes:
        ct = 0
        for instance in training_data:
            if instance[0] == class_label:
                if ct < train_size:
                    train_data.append(instance)
                    ct+=1
                    continue
                valid_data.append(instance)

    # SHUFFLE TRAINING AND VALIDATION DATA
    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)

    train_data = [(instance[1], to_categorical(label_encoder.transform([instance[0]]), num_classes=15))for instance in train_data]
    valid_data = [(instance[1], to_categorical(label_encoder.transform([instance[0]]), num_classes=15)) for instance in valid_data]

    # FORM X_TRAIN AND Y_TRAIN
    x_train, y_train    = zip(*train_data)
    x_train, y_train    = np.squeeze(np.asarray(x_train)), np.squeeze(np.asarray(y_train))
    # L2 NORMALIZE X_TRAIN
    x_train = normalize(x_train, norm='l2')

    # FORM X_VALID AND Y_VALID
    x_valid, y_valid = zip(*valid_data)
    x_valid, y_valid = np.squeeze(np.asarray(x_valid)), np.squeeze(np.asarray(y_valid))
    # L2 NORMALIZE X_VALID
    x_valid = normalize(x_valid, norm='l2')


    # FORM X_ZSL AND Y_ZSL
    y_zsl, x_zsl = zip(*zero_shot_data)
    x_zsl, y_zsl = np.squeeze(np.asarray(x_zsl)), np.squeeze(np.asarray(y_zsl))
    # L2 NORMALIZE X_ZSL
    x_zsl = normalize(x_zsl, norm='l2')

    print("-> data loading is completed.")
    return (x_train, x_valid, x_zsl), (y_train, y_valid, y_zsl)

    

# def custom_kernel_init(shape):
#     class_vectors       = np.load(WORD2VECPATH)
#     training_vectors    = sorted([(label, vec) for (label, vec) in class_vectors if label in train_classes], key=lambda x: x[0])
#     classnames, vectors = zip(*training_vectors)
#     vectors             = np.asarray(vectors, dtype=np.float)
#     vectors             = vectors.T
#     return vectors



def main():
    
    global train_classes
    with open('data/DatasetA_train_20180813/label_list.txt', 'r') as infile:
        name_classes = [str.strip(line).split('\t') for line in infile]
      #Lets take 30 as Zsl classes
    global zsl_classes
    indexes = np.arange(0, len(name_classes))
    zsl_indexes = np.random.choice(indexes, size=30, replace=False)
    zsl_classes = np.array(name_classes)[zsl_indexes]
    train_classes= []
    for i,obj in enumerate(name_classes):
        if not obj in zsl_classes:
            train_classes.append(obj)


    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # SET HYPERPARAMETERS

    global NUM_CLASS, NUM_ATTR, EPOCH, BATCH_SIZE
    NUM_CLASS = 200
    NUM_ATTR = 300
    BATCH_SIZE = 128
    EPOCH = 65

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # TRAINING PHASE

    data_dir = 'data/ordered_data/training'
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    image_size=(224,224),
    seed=1234,
    batch_size=BATCH_SIZE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    image_size=(224,224),
    subset="validation",
    seed=1234,
    batch_size=BATCH_SIZE)

    #(x_train, x_valid, x_zsl), (y_train, y_valid, y_zsl) = load_data()
    model = build_model()
    for image, label in train_ds:
        print(image.shape)
        break;
    model.add()
    model.summary()
    train_model(model, train_ds, val_ds)
    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # CREATE AND SAVE ZSL MODEL

    inp         = model.input
    out         = model.layers[-2].output
    zsl_model   = Model(inp, out)
    print(zsl_model.summary())
    save_keras_model(zsl_model, model_path=MODELPATH)

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # EVALUATION OF ZERO-SHOT LEARNING PERFORMANCE
    #(x_train, x_valid, x_zsl), (y_train, y_valid, y_zsl) = load_data()

    zsl_dir = 'data/ordered_data/zeroshot'

    zsl_ds = tf.keras.preprocessing.image_dataset_from_directory(
    zsl_dir,
    seed=1234)

    zsl_model = load_keras_model(model_path=MODELPATH)

    with open('data/DatasetA_train_20180813/class_wordembeddings.txt', 'r') as infile:
        class_wordembeddings = [str.strip(line).split(' ') for line in infile]

    sorted_embedings = [embed for x in name_classes for embed in class_wordembeddings if embed[0] == x[1]]

    vectors = np.array(sorted_embedings)[:,1:]
    vectors = np.asarray(vectors, dtype=np.float)

    classnames = list(np.array(sorted_embedings)[:,0])
    # class_vectors       = sorted(np.load(WORD2VECPATH), key=lambda x: x[0])
    # classnames, vectors = zip(*class_vectors)
    # classnames          = list(classnames)
    # vectors             = np.asarray(vectors, dtype=np.float)




    tree        = KDTree(vectors)
    pred_zsl    = zsl_model.predict(zsl_ds)
    
    print("Able to predict :)")

    top5, top3, top1 = 0, 0, 0
    for i, pred in enumerate(pred_zsl):
        pred            = np.expand_dims(pred, axis=0)
        dist_5, index_5 = tree.query(pred, k=5)
        pred_labels     = [classnames[index] for index in index_5[0]]
        true_label      = y_zsl[i]
        if true_label in pred_labels:
            top5 += 1
        if true_label in pred_labels[:3]:
            top3 += 1
        if true_label in pred_labels[0]:
            top1 += 1

    print()
    print("ZERO SHOT LEARNING SCORE")
    print("-> Top-5 Accuracy: %.2f" % (top5 / float(len(x_zsl))))
    print("-> Top-3 Accuracy: %.2f" % (top3 / float(len(x_zsl))))
    print("-> Top-1 Accuracy: %.2f" % (top1 / float(len(x_zsl))))
    return


if __name__=='__main__':
    main()