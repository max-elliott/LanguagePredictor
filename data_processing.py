import torch
import os
from utils import *
import random


def _shuffle_data(x, labels):

    indices = list(range(len(x)))
    random.shuffle(indices)

    shuffled_x = []
    shuffled_labels = []

    for i in indices:
        shuffled_x.append(x[i])
        shuffled_labels.append(labels[i])

    return shuffled_x, shuffled_labels


def process_data(data_dir, word_length=10, shuffle=True):

    input_files = [file for file in os.listdir(data_dir)]

    corpus = []
    labels = []
    label_dictionary = {}

    for label_idx, language_file in enumerate(input_files):
        print(f'Processing file {language_file}. Label = {label_idx}')

        label_dictionary[label_idx] = language_file.replace('.csv', '')
        file = os.path.join(data_dir, language_file)
        with open(file, 'r') as f:
            language = os.path.basename(file).replace('.csv', '')
            print(language)
            for word in f:
                corpus.append(word.lower())
                labels.append(label_idx)

    print(len(corpus))
    print(len(labels))
    print(input_files)
    d = generate_index_dict(corpus)

    index_corpus = word2index(corpus, d)

    # print(index_corpus)
    # example_word = index_corpus[0]
    # print(index2onehot(example_word, 5))
    # print(index_corpus[:3])
    # print(indexCorpus2onehotCorpus(index_corpus[:3]))

    onehot_corpus, vector_length, word_length = indexCorpus2onehotCorpus(index_corpus, word_length=word_length)

    onehot_corpus, labels = _shuffle_data(onehot_corpus, labels)

    print('Processed data')

    return onehot_corpus, labels, vector_length, word_length, label_dictionary

if __name__ == '__main__':
    pass
