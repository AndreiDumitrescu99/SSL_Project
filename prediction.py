from tkinter import BASELINE
import tensorflow as tf
import numpy as np
from preprocess import get_sentence_embeds_for_dataset
from data_reader import read_dataset_convabuse, read_dataset_tcc, read_dataset_alyt

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

DATASET = 'ALYT'
MODEL_PATH = './models/bilstm_attention_alyt'
TARGETS = 2
BASELINE = False

if __name__ == "__main__":

    test_set = []

    if DATASET == 'ConvAbuse':
        _, _, test_set = read_dataset_convabuse('./datasets/ConvAbuseEMNLPtrain.csv', './datasets/ConvAbuseEMNLPvalid.csv', './datasets/ConvAbuseEMNLPtest.csv')
    if DATASET == 'TCC':
        _, test_set = read_dataset_tcc('./datasets/Dataset40CombinedV2.csv')
    if DATASET == 'ALYT':
        train_set, test_set = read_dataset_alyt('./datasets/ALYT_data.csv')
        TARGETS = 3

    x_test = []
    y_test = []

    if BASELINE == True:
        x_test, y_test = get_sentence_embeds_for_dataset(test_set, nn=True, targets=TARGETS, average=True)
    else:
        x_test, y_test = get_sentence_embeds_for_dataset(test_set, targets=TARGETS, average=False)

    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    predictions = loaded_model.predict(np.array(x_test), verbose=1)

    cnt = 0
    sem_wrong = 0
    sem_correct = 0
    for p in predictions:
        outcome = np.argmax(p)
        real_outcome = np.argmax(y_test[cnt])
        
        if outcome != real_outcome and sem_wrong == 0:
            print("~~~~~ Wrong Sample ~~~~~")
            print("Text:")
            print(test_set[cnt][0])
            print("Real Outcome: ", real_outcome, " Our outcome: ", outcome)
            sem_wrong = 1

        if outcome == real_outcome and sem_correct == 0:
            print("~~~~~ Correct Sample ~~~~~")
            print("Text:")
            print(test_set[cnt][0])
            print("Real Outcome: ", real_outcome, " Our outcome: ", outcome)
            sem_correct = 1
        
        if sem_correct == 1 and sem_wrong == 1:
            break

        cnt = cnt + 1