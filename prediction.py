from tkinter import BASELINE
from pytz import VERSION
import tensorflow as tf
import torch as th
import numpy as np
from preprocess import get_sentence_embeds_for_dataset
from data_reader import read_dataset_convabuse, read_dataset_tcc, read_dataset_alyt, get_dataset_bert
from second_iteration import Net, Net2


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

DATASET = 'ConvAbuse'
MODEL_PATH = './models_torch/convabuse_combined_hatebert2.pt'
TARGETS = 2
VERSION = 'second_iteration' # This should be one of the following: 'baseline', 'first_iteration', 'second_iteration'.

"""THIS ARE ONLY FOR PYTORCH EXPERIMENTS"""
INPUTS = 768
MODEL_VERSION = 2

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

    if VERSION == 'baseline':
        x_test, y_test = get_sentence_embeds_for_dataset(test_set, nn=True, targets=TARGETS, average=True)
    else:
        if VERSION == 'first_iteration':
            x_test, y_test = get_sentence_embeds_for_dataset(test_set, targets=TARGETS, average=False)
        else:
            x_test, y_test, max_len = get_dataset_bert(test_set, TARGETS)


    if VERSION == 'second_iteration':
        if MODEL_VERSION == 2:
            net = Net2(INPUTS, TARGETS)
        else:
            net = Net(INPUTS, TARGETS)

        net.load_state_dict(th.load(MODEL_PATH))
        net.to(device)
    else:
        loaded_model = tf.keras.models.load_model(MODEL_PATH)
        predictions = loaded_model.predict(np.array(x_test), verbose=1)

    cnt = 0
    sem_wrong = 0
    sem_correct = 0

    if VERSION == 'second_iteration':
        for i, x in enumerate(x_test):
            x = th.tensor([x]).to(device)

            y_out = net.forward(x).cpu().detach().numpy()

            y_out = np.argmax(y_out)

            if np.argmax(y_test[i]) != y_out and sem_wrong == 0:
                print("~~~~~ Wrong Sample ~~~~~")
                print("Text:")
                print(test_set[cnt][0])
                print("Real Outcome: ", np.argmax(y_test[i]), " Our outcome: ", y_out)
                sem_wrong = 1
            
            if np.argmax(y_test[i]) == y_out and sem_correct == 0:
                print("~~~~~ Correct Sample ~~~~~")
                print("Text:")
                print(test_set[cnt][0])
                print("Real Outcome: ", np.argmax(y_test[i]), " Our outcome: ", y_out)
                sem_correct = 1
            
            if sem_correct == 1 and sem_wrong == 1:
                break

            cnt = cnt + 1
    else: 
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