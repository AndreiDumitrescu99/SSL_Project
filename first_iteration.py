from typing import Any
import tensorflow as tf
import numpy as np
from data_reader import read_dataset_alyt, read_dataset_tcc, read_dataset_convabuse
from preprocess import get_sentence_embeds_for_dataset
from utils import plot_loss, plot_accuracy, translate_prediction
from sklearn.metrics import accuracy_score, f1_score

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def bilstm_attention_train(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, save_name: str, total_outputs: int = 2) -> Any:

    inputs = tf.keras.Input(shape=(None, 300))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences= True))(inputs)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences= True))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Attention Part
    attention = tf.keras.layers.Dense(300, activation='tanh')(x)
    attention = tf.exp(attention)
    attention = attention / tf.cast(tf.math.reduce_sum(attention, axis=1, keepdims=True) + 1e-5, tf.float32)
    x = x * attention
    x = tf.math.reduce_sum(x, axis=1)

    x = tf.keras.layers.Dense(100)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(total_outputs, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_name, monitor='val_loss',verbose=1, save_best_only=True, mode='auto')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics='accuracy', run_eagerly=False)

    x_train = np.asarray(x_train).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    x_valid = np.asarray(x_valid).astype('float32')
    y_valid = np.asarray(y_valid).astype('float32')

    history = model.fit(
        x = x_train,
        y = y_train,
        batch_size = 128,
        epochs = 50,
        verbose=1,
        validation_data=(x_valid, y_valid),
        callbacks=[checkpoint]
    )
    
    return history

def bilstm_attention_test(model_path: str, x_test: np.ndarray, y_test: np.ndarray):

    loaded_model = tf.keras.models.load_model(model_path)
    x_pred_nn = loaded_model.predict(np.array(x_test), verbose=1)

    prediction = translate_prediction(x_pred_nn)

    print("accuracy", accuracy_score(y_test, prediction))
    print("macro_F1", f1_score(y_test, prediction, average='macro'))

def run_first_iteration_experiments(just_eval: bool = False) -> None:

    train_set, test_set = read_dataset_alyt('./datasets/ALYT_data.csv')

    x_train, y_train = get_sentence_embeds_for_dataset(train_set, targets=3, average=False)
    x_test, y_test = get_sentence_embeds_for_dataset(test_set, targets=3, average=False)

    if just_eval == False:
        percentage = 0.2
        history = bilstm_attention_train(
            np.array(x_train[:int(len(train_set) * (1 - percentage))]),
            np.array(y_train[:int(len(train_set) * (1 - percentage))]),
            np.array(x_train[int(len(train_set) * (1 - percentage)) + 1:]),
            np.array(y_train[int(len(train_set) * (1 - percentage)) + 1:]),
            './models/bilstm_attention_alyt',
            3
        )

        plot_loss(history, './plots/bilstm_attention_alyt_loss')
        plot_accuracy(history, './plots/bilstm_attention_alyt_accuracy')

    bilstm_attention_test('./models/bilstm_attention_alyt', x_test, y_test)

    train_set, test_set = read_dataset_tcc('./datasets/Dataset40CombinedV2.csv')

    x_train, y_train = get_sentence_embeds_for_dataset(train_set, targets=2, average=False)
    x_test, y_test = get_sentence_embeds_for_dataset(test_set, targets=2, average=False)

    if just_eval == False:
        history = bilstm_attention_train(
            np.array(x_train[:int(len(train_set) * (1 - percentage))]),
            np.array(y_train[:int(len(train_set) * (1 - percentage))]),
            np.array(x_train[int(len(train_set) * (1 - percentage)) + 1:]),
            np.array(y_train[int(len(train_set) * (1 - percentage)) + 1:]),
            './models/bilstm_attention_combined',
            2
        )

        plot_loss(history, './plots/bilstm_attention_combined_loss')
        plot_accuracy(history, './plots/bilstm_attention_combined_accuracy')

    bilstm_attention_test('./models/bilstm_attention_combined', x_test, y_test)

    train_set, validation_set, test_set = read_dataset_convabuse('./datasets/ConvAbuseEMNLPtrain.csv', './datasets/ConvAbuseEMNLPvalid.csv', './datasets/ConvAbuseEMNLPtest.csv')

    x_train, y_train = get_sentence_embeds_for_dataset(train_set, targets=2, average=False)
    x_valid, y_valid = get_sentence_embeds_for_dataset(validation_set, targets=2, average=False)
    x_test, y_test = get_sentence_embeds_for_dataset(test_set, targets=2, average=False)

    if just_eval == False:

        history = bilstm_attention_train(
            np.array(x_train),
            np.array(y_train),
            np.array(x_valid),
            np.array(y_valid),
            './models/bilstm_attention_convabuse',
            2
        )

        plot_loss(history, './plots/bilstm_attention_convabuse_loss')
        plot_accuracy(history, './plots/bilstm_attention_convabuse_accuracy')

    bilstm_attention_test('./models/bilstm_attention_convabuse', x_test, y_test)


if __name__ == "__main__":

    run_first_iteration_experiments(True)