from cProfile import run
from typing import Any
from nbformat import read
from data_reader import read_dataset_alyt, read_dataset_tcc, read_dataset_convabuse
from preprocess import get_sentence_embeds_for_dataset
from sklearn.metrics import accuracy_score, f1_score
from utils import plot_loss, plot_accuracy, translate_prediction
from sklearn import svm
import numpy as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def svm_baseline_train(x_train: np.ndarray, y_train: np.ndarray, verbose: int = 0) -> svm.SVC:

    clf_tune = svm.SVC(random_state=0, verbose=verbose)
    clf_tune.fit(np.array(x_train), np.array(y_train))

    return clf_tune

def svm_baseline_test(clf: svm.SVC, x_test: np.ndarray, y_test: np.ndarray) -> None:

    pred_clf = clf.predict(x_test)

    print("accuracy", accuracy_score(y_test, pred_clf))
    print("macro_F1", f1_score(y_test, pred_clf, average='macro'))

def nn_baseline_train(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, save_name: str, total_outputs: int = 2) -> Any:

    print(x_train)
    print(y_train)
    print(x_valid)
    print(y_valid)

    inputs = tf.keras.Input(shape=(300,))
    x = tf.keras.layers.Dense(150, activation='relu')(inputs)
    x = tf.keras.layers.Dense(75, activation='relu')(x)
    x = tf.keras.layers.Dense(50, activation='relu')(x)
    x = tf.keras.layers.Dense(25, activation='relu')(x)
    x = tf.keras.layers.Dense(total_outputs, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_name, monitor='val_loss',verbose=1, save_best_only=True, mode='auto')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss= 'categorical_crossentropy', metrics='accuracy', run_eagerly=False)

    x_train = np.asarray(x_train).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    x_valid = np.asarray(x_valid).astype('float32')
    y_valid = np.asarray(y_valid).astype('float32')

    history = model.fit(
        x = x_train,
        y = y_train,
        batch_size = 64,
        epochs = 100,
        verbose=1,
        validation_data=(x_valid, y_valid),
        callbacks=[checkpoint]
    )
    
    return history

def nn_baseline_eval(model_path: str, x_test: np.ndarray, y_test: np.ndarray):

    loaded_model = tf.keras.models.load_model(model_path)
    x_pred_nn = loaded_model.predict(np.array(x_test), verbose=1)

    prediction = translate_prediction(x_pred_nn)

    print("accuracy", accuracy_score(y_test, prediction))
    print("macro_F1", f1_score(y_test, prediction, average='macro'))

def run_svm_experiments() -> None:

    train_set, test_set = read_dataset_alyt('./datasets/ALYT_data.csv')
    print("~~~ ALYT Dataset ~~~")
    print(len(train_set), len(test_set))

    x_train, y_train = get_sentence_embeds_for_dataset(train_set)
    x_test, y_test = get_sentence_embeds_for_dataset(test_set)
    svm_baseline_test(svm_baseline_train(x_train, y_train), x_test, y_test)

    print("~~~ Combined Dataset ~~~")
    train_set, test_set = read_dataset_tcc('./datasets/Dataset40CombinedV2.csv')
    print(len(train_set), len(test_set))

    x_train, y_train = get_sentence_embeds_for_dataset(train_set)
    x_test, y_test = get_sentence_embeds_for_dataset(test_set)
    svm_baseline_test(svm_baseline_train(x_train, y_train), x_test, y_test)


    print("~~~ Conv Abuse Dataset ~~~")
    train_set, validation_set, test_set = read_dataset_convabuse('./datasets/ConvAbuseEMNLPtrain.csv', './datasets/ConvAbuseEMNLPvalid.csv', './datasets/ConvAbuseEMNLPtest.csv')
    print(len(train_set), len(validation_set), len(test_set))

    x_train, y_train = get_sentence_embeds_for_dataset(train_set)
    x_test, y_test = get_sentence_embeds_for_dataset(test_set)
    svm_baseline_test(svm_baseline_train(x_train, y_train), x_test, y_test)

def run_nn_experiments(just_eval: bool = False) -> None:

    percentage = 0.2

    train_set, test_set = read_dataset_alyt('./datasets/ALYT_data.csv')
    print("~~~ ALYT Dataset ~~~")
    print(len(train_set), len(test_set))

    x_train, y_train = get_sentence_embeds_for_dataset(train_set, True, 3)
    x_test, y_test = get_sentence_embeds_for_dataset(test_set, True, 3)

    if just_eval == False:
        history = nn_baseline_train(
            np.array(x_train[:int(len(train_set) * (1 - percentage))]),
            np.array(y_train[:int(len(train_set) * (1 - percentage))]),
            np.array(x_train[int(len(train_set) * (1 - percentage)) + 1:]),
            np.array(y_train[int(len(train_set) * (1 - percentage)) + 1:]),
            './models/nn_baseline_alyt',
            3
        )

        plot_loss(history, './plots/nn_baseline_alyt_loss')
        plot_accuracy(history, './plots/nn_baseline_alyt_accuracy')

    nn_baseline_eval('./models/nn_baseline_alyt', x_test, y_test)

    train_set, test_set = read_dataset_tcc('./datasets/Dataset40CombinedV2.csv')
    print("~~~ Combined Dataset ~~~")
    print(len(train_set), len(test_set))

    x_train, y_train = get_sentence_embeds_for_dataset(train_set, True, 2)
    x_test, y_test = get_sentence_embeds_for_dataset(test_set, True, 2)

    if just_eval == False:
        history = nn_baseline_train(
            np.array(x_train[:int(len(train_set) * (1 - percentage))]),
            np.array(y_train[:int(len(train_set) * (1 - percentage))]),
            np.array(x_train[int(len(train_set) * (1 - percentage)) + 1:]),
            np.array(y_train[int(len(train_set) * (1 - percentage)) + 1:]),
            './models/nn_baseline_combined',
            2
        )

        plot_loss(history, './plots/nn_baseline_combined_loss')
        plot_accuracy(history, './plots/nn_baseline_combined_accuracy')

    nn_baseline_eval('./models/nn_baseline_combined', x_test, y_test)

    train_set, validation_set, test_set = read_dataset_convabuse('./datasets/ConvAbuseEMNLPtrain.csv', './datasets/ConvAbuseEMNLPvalid.csv', './datasets/ConvAbuseEMNLPtest.csv')

    x_train, y_train = get_sentence_embeds_for_dataset(train_set, True, 2)
    x_valid, y_valid = get_sentence_embeds_for_dataset(validation_set, True, 2)
    x_test, y_test = get_sentence_embeds_for_dataset(test_set, True, 2)

    if just_eval == False:
        history = nn_baseline_train(
            np.array(x_train),
            np.array(y_train),
            np.array(x_valid),
            np.array(y_valid),
            './models/nn_baseline_convabuse',
            2
        )

        plot_loss(history, './plots/nn_baseline_convabuse_loss')
        plot_accuracy(history, './plots/nn_baseline_convabuse_accuracy')

    nn_baseline_eval('./models/nn_baseline_convabuse', x_test, y_test)


if __name__ == "__main__":

    run_nn_experiments(True)