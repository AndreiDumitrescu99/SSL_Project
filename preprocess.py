from typing import List, Tuple
import gensim.downloader as api
import nltk
import numpy as np

punctuation_list = ["'", ".", "!", "?", "-", "’", ",", "—", "''", "``", ":", "(", ")", "@", "/"]
word2vec = api.load('word2vec-google-news-300')
WORD_EMBED_SIZE = 300

def find_word_embeding_sentence(sentence: str, average = True) -> List[float]:

    vecs = []
    tk = nltk.TweetTokenizer() 
    # tokenize the sentence into words
    for j in tk.tokenize(sentence):
        if j not in punctuation_list and j.find("http") == -1:
            try:
                vecs.append(word2vec[j.lower()])
            except KeyError:
                continue

    if average == False:
        return vecs

    if len(vecs) != 0:
        sum = vecs[0]
        for j in range(1, len(vecs)):
            sum = sum + vecs[j]
        
        return sum / len(vecs)

    return []

def map(target: int, targets: int) -> List[int]:

    if targets == 3:
        if target == -1:
            return [1, 0, 0]
        if target == 0:
            return [0, 1, 0]
        if target == 1:
            return [0, 0, 1]
    else:
        if target == 0:
            return [1, 0]
        if target == 1:
            return [0, 1]

def get_sentence_embeds_for_dataset(dataset: List[Tuple[str, int]], nn: bool = False, targets: int = 2, average: bool = True) -> Tuple[np.ndarray, np.ndarray]:

    x_train = []
    y_train = []

    max_len = 0
    cnt = 0

    if average == False:

        for text, target in dataset:
            emb = find_word_embeding_sentence(text, average)
            if len(emb) > max_len and len(emb) <= 120:
                max_len = len(emb)
            if len(emb) > 120:
                cnt = cnt + 1

    for text, target in dataset:
        emb = find_word_embeding_sentence(text, average)

        if emb != []:

            if average == False:
                if len(emb) <= max_len:
                    for _ in range(max_len - len(emb)):
                        emb.append(np.zeros(WORD_EMBED_SIZE, dtype="float32"))

                    assert np.array(emb).shape[0] == max_len
                    x_train.append(np.array(emb, dtype="float32"))
                    y_train.append(map(target, targets))
            else:
                x_train.append(np.array(emb, dtype="float32"))

                if nn == True:
                    y_train.append(map(target, targets))
                else:
                    y_train.append(target)

    print(len(x_train), len(y_train))
    return np.array(x_train), np.array(y_train)