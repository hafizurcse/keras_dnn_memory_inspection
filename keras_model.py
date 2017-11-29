# author: Hafizur Rahman
# email: hafizurcse02@gmail.complicate
# copy rights: can be viewed only, no part of the code can be copies or shared, all rights reserved

# This code builds a keras deep neural network model for categorical data and
# inspects the memory status at each stage
# It uses multi_gpu_model of the keras to bring data parallelism so it trains faster
# Note: initialize your setting params inside main function

from pandas import read_csv, get_dummies
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import from psutil import Process
from os import getpid
from gc import collect
# get the Process id
proc = Process(getpid())

# read the input data
def read_data(file_path):
    df = read_csv(file_path)
    df = df.applymap(str)
    # reduce the input size by half
    df_half = df[:int(len(df)/2)]
    del df
    collect()
    mem0 = proc.memory_info().rss
    print("read_data - mem0 = ", mem0)
    return df_half

# encode the target variable
def encode_target(items):
    le_target = LabelEncoder()
    le_target.fit(items)
    le_encoded_target = le_target.transform(items)
    encoded_target = np_utils.to_categorical(le_encoded_target)
    del le_encoded_target
    collect()
    mem0 = proc.memory_info().rss
    print("encode_target - mem0 = ", mem0)
    return encoded_target, le_target

# encode the input by avoiding zeros
def encode_input_features(df, skip):
    train_features = pd.get_dummies(df)
    inp_cols = df.columns
    temp_str = []
    for col in inp_cols:
        temp_str.append(col + '_' + skip)
    for col in temp_str:
        if col != 'item1_0':
            train_features[col] = 0
    del df, temp_str, col, inp_cols
    collect()
    mem0 = proc.memory_info().rss
    print("encode_input_features - mem0 = ", mem0)
    return train_features

# get encoded training features and their targets
def get_train_test(df, skip):
    # slect the columns we are interested in
    inp_cols = df.columns[1:len(cols) - 1]
    # encode the input
    train_features = encode_input_features(df[inp_cols].copy(), skip)
    # encode the target variable
    train_target, le_out = encode_target(df.recommendation)
    # get the target variable's number of categories
    output_dim = len(np.unique(np.array(df.recommendation)))
    # free some memory
    del df, cols
    collect()
    mem0 = proc.memory_info().rss
    print("get_train_test - mem0 = ", mem0)
    return train_features, train_target, le_out, output_dim

# shuffle the rows to bring randomness and classifier will discover the pattern
def shuffle_split(train_features, train_target, total_rows, ratio):
    # numpy permutation
    per_rows = np.random.permutation(total_rows)
    collect()
    mem1 = proc.memory_info().rss
    print("shuffle_split - mem0 = ", mem0)
    train_features_shuffled = train_features[per_rows, :]
    del train_features
    collect()
    mem1 = proc.memory_info().rss
    print("shuffle_split - mem1 = ", mem1)
    train_target_shuffled = train_target[per_rows, :]
    del train_target
    collect()
    mem2 = proc.memory_info().rss
    print("shuffle_split - mem2 = ", mem2)

    # create train and test dataset using split ratio
    split_index = int(len(per_rows)*ratio)
    test_feature = train_features_shuffled[0:split_index, :]
    test_target = train_target_shuffled[0:split_index, :]
    del per_rows, ratio
    collect()
    mem2 = proc.memory_info().rss
    print("shuffle_split - mem2 = ", mem2)
    return train_features_shuffled, train_target_shuffled, test_feature, test_target

# build the keras model
def build_model(nodes, activations, input_dim, num_of_gpus):
    with tf.device('/cpu:0'):
        model = Sequential()
        for i in range(len(nodes)):
            if i == 0:
                model.add(Dense(nodes[i], input_dim = input_dim, activation = activations[i]))
            else:
                model.add(Dense(nodes[i], activation = activations[i]))
    # Replicates `model` on 8 GPUs.
    # This assumes that your machine has 8 available GPUs.
    parallel_model = multi_gpu_model(model, gpus=num_of_gpus)
    return parallel_model

# train the keras model
def compile_train(model, train_features, train_target, test_features, test_target, model_path, patience, save_option):
    # Set callback functions to early stop training and save the best model so far
    checkpoint = [EarlyStopping(monitor = 'val_loss', patience = patience),
                 ModelCheckpoint(filepath = model_path, monitor='val_loss', save_best_only = save_option)]
    # compile the model
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    # train the model
    history = model.fit(train_features, train_target, epochs = 100, verbose = 2, callbacks = checkpoint, validation_data=(test_features, test_target))
    # get score and accuracy
    eval_scores = model.evaluate(test_features, test_target, verbose=0)
    print('Test loss:', eval_scores[0])
    print('Test accuracy:', eval_scores[1])
    return model

if __name__ == '__main__':
    # initialize directory
    input_path = 'data/all_items_half_million.csv'
    ratio = 0.8
    skip = '0'

    # model params
    nodes = [200, 200, 250, 300]
    num_of_gpus = 4
    activations = ['relu', 'relu', 'sigmoid', 'sigmoid', 'softmax']
    save_option = True
    patience = 3
    model_path = 'saved_models/best_model.hdf5'

    # read data
    df = read_data(input_path)
    total_rows = len(df)

    # get train and test data
    train_features, train_target, le_out, output_dimension = get_train_test(df, skip)
    del df
    collect()
    mem0 = proc.memory_info().rss
    print("main - mem0 = ", mem0)
    encoded_cols = train_features.columns
    # shuffle and split the input features and target variable
    train_features, train_target, test_features, test_target = shuffle_split(train_features.values, train_target, total_rows, ratio)
    # update nodes of the keras NN model
    nodes.append(output_dimension)
    # input dimension of the keras NN model
    input_dim = train_features.shape[1]
    collect()
    mem1 = proc.memory_info().rss
    print("main - mem1 = ", mem1)
    # build the model
    model = build_model(nodes, activations, input_dim, num_of_gpus)
    # compile, train and save
    model = compile_train(model, train_features, train_target, test_features, test_target, model_path, patience, save_option)
