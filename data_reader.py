import csv
import math
from typing import List, Tuple
import numpy as np  
import tensorflow as tf
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import BertweetTokenizer, AutoTokenizer, AutoModel
import random
from preprocess import map
# nltk.download('punkt')

SEED = 13

def read_dataset_alyt(path_to_dataset: str, percentage_split: float = 0.2) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    
    random.seed(SEED)

    data_1_label = []
    data_0_label = []
    data_minus1_label = []

    input_file = open(path_to_dataset, 'r', encoding = "utf8")
    csv_reader = csv.reader(input_file, delimiter=",")
    header = next(csv_reader)

    for row in csv_reader:
        comment_id, comment, label, label_normalised, video_id, video_topic, video_type = row

        if label_normalised == '0':
          data_0_label.append((comment, int(label_normalised)))
        if label_normalised == '1':
          data_1_label.append((comment, int(label_normalised)))
        if label_normalised == '-1':
          data_minus1_label.append((comment, int(label_normalised)))

    test_set = []
    train_set = []

    random.shuffle(data_0_label)
    random.shuffle(data_1_label)
    random.shuffle(data_minus1_label)

    test_set = data_0_label[:math.floor(len(data_0_label) * percentage_split)]
    test_set = test_set + data_1_label[:math.floor(len(data_1_label) * percentage_split)]
    test_set = test_set + data_minus1_label[:math.floor(len(data_minus1_label) * percentage_split)]

    train_set = data_0_label[math.floor(len(data_0_label) * percentage_split) + 1:]
    train_set = train_set + data_1_label[math.floor(len(data_1_label) * percentage_split) + 1:]
    train_set = train_set + data_minus1_label[math.floor(len(data_minus1_label) * percentage_split) + 1:]

    random.shuffle(train_set)
    random.shuffle(test_set)

    input_file.close()

    return train_set, test_set

def read_dataset_tcc(path_to_dataset: str, percentage_split: float = 0.2) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:

    random.seed(SEED)

    test_set = []
    train_set = []
    data = []

    input_file = open(path_to_dataset, 'r', encoding="utf8")
    csv_reader = csv.reader(input_file, delimiter=",")
    header = next(csv_reader)

    mapLab = {
        "['offensive']": 1,
        "['none']": 0
    }

    labels_list = {
        "['offensive']": [],
        "['none']": []
    }

    for row in csv_reader:
        id, file_platform, file_language, file_name, text, labels = row
        if file_language == "en" and labels != "['idk/skip']" and labels != "['relation']" and labels != "[]":
            labels_list[labels].append((text, mapLab[labels]))

    test_set = labels_list["['none']"][:math.floor(len(labels_list["['none']"]) * percentage_split)]
    test_set = test_set + labels_list["['offensive']"][:math.floor(len(labels_list["['offensive']"]) * percentage_split)]

    train_set = labels_list["['none']"][math.floor(len(labels_list["['none']"]) * percentage_split) + 1:]
    train_set = train_set + labels_list["['offensive']"][math.floor(len(labels_list["['offensive']"]) * percentage_split) + 1:]

    random.shuffle(train_set)
    random.shuffle(test_set)

    input_file.close()

    return train_set, test_set

def read_dataset_convabuse(train_dataset: str, validation_dataset: str, test_dataset: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:

	test_set = []
	train_set = []
	validation_set = []

	train_file = open(train_dataset, 'r', encoding="utf8")
	csv_reader = csv.reader(train_file, delimiter=",")
	header = next(csv_reader)

	cnt = 0
	for row in csv_reader:
		train_set.append((row[2] + ' ' + row[3] + ' ' + row[4] + ' ' + row[5], int(row[-1])));
		cnt = cnt + int(row[-1])

	train_file.close()

	test_file = open(test_dataset, 'r', encoding="utf8")
	csv_reader = csv.reader(test_file, delimiter=",")
	header = next(csv_reader)

	cnt = 0
	for row in csv_reader:
		test_set.append((row[2] + ' ' + row[3] + ' ' + row[4] + ' ' + row[5], int(row[-1])));
		cnt = cnt + int(row[-1])

	test_file.close()
	validation_file = open(validation_dataset, 'r', encoding="utf8")
	csv_reader = csv.reader(validation_file, delimiter=",")
	header = next(csv_reader)

	cnt = 0
	for row in csv_reader:
		validation_set.append((row[2] + ' ' + row[3] + ' ' + row[4] + ' ' + row[5], int(row[-1])));
		cnt = cnt + int(row[-1])
	
	validation_file.close()

	return train_set, validation_set, test_set

def get_dataset_bert(myset: List[Tuple[str, int]], testing: bool = False, target: int = 2, max_len: int = 100):

    tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")

    dataset = []
    xs = []
    ys = []
    ids = []
    stds = []

    for x, y in myset:

        embeds = tokenizer.encode(x, max_length = max_len, padding = 'max_length', truncation=True)
        xs.append(embeds)
        ys.append(map(y, target))

    return xs, ys, max_len

