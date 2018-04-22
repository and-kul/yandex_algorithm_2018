from typing import Union, List

import keras
import numpy as np
import pandas as pd
from keras import Model
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, LSTM, Dense, Dropout, Lambda, GRU, Bidirectional, Reshape
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from_word_to_vector = dict()

with open("fasttext_train_and_public.txt", "r", encoding="utf-8") as file:
    for line in file:
        words = line.split()
        from_word_to_vector[words[0]] = np.array([float(x) for x in words[1:]])

dataset = pd.read_csv("data/train.tsv", names=["context_id", "context_2", "context_1", "context_0",
                                               "reply_id", "reply", "label", "confidence"], header=None, sep="\t",
                      quoting=3)


def convert_sentence_to_list_of_vectors(sentence: Union[str, float]):
    if type(sentence) == str:
        return [from_word_to_vector[word] for word in sentence.split()]
    else:
        return []


label_to_number = {
    "bad": -1,
    "neutral": 0,
    "good": 1
}


def convert_label_to_number(label: str):
    return label_to_number[label]


dataset[["context_2", "context_1", "context_0", "reply"]] = \
    dataset[["context_2", "context_1", "context_0", "reply"]].applymap(convert_sentence_to_list_of_vectors)
dataset[["label"]] = dataset[["label"]].applymap(convert_label_to_number)


def convert_sequence_of_vectors_to_padded_2D_array(sequence: List[np.ndarray], max_len: int):
    padding_size = max_len - len(sequence)
    padding = [np.zeros((300,)) for _ in range(padding_size)]
    return np.array(padding + sequence)


def get_matrices(batch):
    max_sentence_len = np.max([len(x) for x in batch.context_2] +
                              [len(x) for x in batch.context_1] +
                              [len(x) for x in batch.context_0] +
                              [len(x) for x in batch.reply])

    X = np.zeros((batch.shape[0], 4, max_sentence_len, 300), dtype=np.float32)
    # (train_data_batch.shape[0], 4, 128)
    y = batch.label.values
    weights = batch.confidence.values

    for i in range(batch.shape[0]):
        X[i, 0, :, :] = convert_sequence_of_vectors_to_padded_2D_array(batch.iloc[i].context_2, max_sentence_len)
        X[i, 1, :, :] = convert_sequence_of_vectors_to_padded_2D_array(batch.iloc[i].context_1, max_sentence_len)
        X[i, 2, :, :] = convert_sequence_of_vectors_to_padded_2D_array(batch.iloc[i].context_0, max_sentence_len)
        X[i, 3, :, :] = convert_sequence_of_vectors_to_padded_2D_array(batch.iloc[i].reply, max_sentence_len)

    return X, y, weights


def get_test_matrices(batch):
    max_sentence_len = np.max([len(x) for x in batch.context_2] +
                              [len(x) for x in batch.context_1] +
                              [len(x) for x in batch.context_0] +
                              [len(x) for x in batch.reply])

    X = np.zeros((batch.shape[0], 4, max_sentence_len, 300), dtype=np.float32)

    for i in range(batch.shape[0]):
        X[i, 0, :, :] = convert_sequence_of_vectors_to_padded_2D_array(batch.iloc[i].context_2, max_sentence_len)
        X[i, 1, :, :] = convert_sequence_of_vectors_to_padded_2D_array(batch.iloc[i].context_1, max_sentence_len)
        X[i, 2, :, :] = convert_sequence_of_vectors_to_padded_2D_array(batch.iloc[i].context_0, max_sentence_len)
        X[i, 3, :, :] = convert_sequence_of_vectors_to_padded_2D_array(batch.iloc[i].reply, max_sentence_len)

    return X


train_part, validation_part = train_test_split(dataset, test_size=0.1, stratify=dataset.label, random_state=0)

batch_size = 64


def generate_train_batch():
    while True:
        for i in range(train_part.shape[0] // batch_size):
            train_data_batch = train_part.iloc[i * batch_size: (i + 1) * batch_size]
            yield get_matrices(train_data_batch)


def generate_validation_batch():
    while True:
        for i in range(validation_part.shape[0] // batch_size):
            validation_data_batch = validation_part.iloc[i * batch_size: (i + 1) * batch_size]
            yield get_matrices(validation_data_batch)


def my_loss(y_true, y_pred):
    where_drop_to_0 = tf.logical_or(
        tf.logical_and(
            tf.equal(y_true, -1),
            tf.less(y_pred, -1)
        ),
        tf.logical_and(
            tf.equal(y_true, 1),
            tf.greater(y_pred, 1)
        )
    )

    res = tf.where(where_drop_to_0,
                   tf.zeros_like(y_true),
                   tf.square(y_pred - y_true))

    return tf.reduce_mean(res, axis=-1)


inp = Input(shape=(4, None, 300))

get_context_2 = Lambda(lambda batch: batch[:, 0, :, :])(inp)
get_context_1 = Lambda(lambda batch: batch[:, 1, :, :])(inp)
get_context_0 = Lambda(lambda batch: batch[:, 2, :, :])(inp)
get_reply = Lambda(lambda batch: batch[:, 3, :, :])(inp)





shared_bidirectional_gru = Bidirectional(GRU(100), merge_mode="concat")

encoded_context_2 = shared_bidirectional_gru(get_context_2)
encoded_context_1 = shared_bidirectional_gru(get_context_1)
encoded_context_0 = shared_bidirectional_gru(get_context_0)
encoded_reply = shared_bidirectional_gru(get_reply)


concatenated_context = keras.layers.concatenate([encoded_context_2, encoded_context_1, encoded_context_0])
reshaped_context = Reshape((3, 200))(concatenated_context)

vertical_gru = GRU(200)(reshaped_context)

drop_after_vertical_gru = Dropout(0.4)(vertical_gru)
dense_after_vertical_gru = Dense(200)(drop_after_vertical_gru)

drop_after_encoded_reply = Dropout(0.4)(encoded_reply)
dense_after_encoded_reply = Dense(200)(encoded_reply)

stacked = keras.layers.concatenate([dense_after_vertical_gru, dense_after_encoded_reply])


drop0 = Dropout(0.4)(stacked)

dense1 = Dense(400, activation="relu")(drop0)

drop1 = Dropout(0.4)(dense1)

# dense2 = Dense(100, activation="relu")(drop1)
# drop2 = Dropout(0.4)(dense2)

output_layer = Dense(1)(drop1)

model = Model(input=inp, output=output_layer)
model.compile(loss=my_loss, optimizer=Adam(clipnorm=1.))

model.fit_generator(generate_train_batch(),
                    steps_per_epoch=train_part.shape[0] // batch_size,
                    epochs=2,
                    verbose=True,
                    validation_data=generate_validation_batch(),
                    validation_steps=validation_part.shape[0] // batch_size,
                    callbacks=[
                        ModelCheckpoint("weights.hdf5", monitor="val_loss", save_best_only=True)
                    ]
                    )

#####

public_test = pd.read_csv("data/public.tsv", names=["context_id", "context_2", "context_1", "context_0",
                                                    "reply_id", "reply"], header=None, sep="\t",
                          quoting=3)

public_test[["context_2", "context_1", "context_0", "reply"]] = \
    public_test[["context_2", "context_1", "context_0", "reply"]].applymap(convert_sentence_to_list_of_vectors)

X_test = get_test_matrices(public_test)

y_predicted = model.predict(X_test)


public_test["score"] = y_predicted

public_test = public_test[["context_id", "reply_id", "score"]]


public_test = public_test.groupby("context_id", sort=True) \
    .apply(lambda g: g.sort_values(["score"], ascending=False))[["reply_id", "score"]] \
    .reset_index()




public_test.to_csv("submission.txt", sep="\t", columns=["context_id", "reply_id"], header=False, index=False)
