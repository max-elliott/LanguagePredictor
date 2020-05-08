import os
import torch


def generate_index_dict(corpus):

    char2int_dict = {}

    current_index = 1
    for word in corpus:
        for c in word:
            if c not in char2int_dict.keys() and not c.isspace() and not '\n' in c:
                char2int_dict[c] = current_index
                current_index += 1
    char2int_dict['<pad>'] = 0

    return char2int_dict


def word2index(corpus, dictionary):

    output_corpus = []

    for word in corpus:
        word_as_index = []
        for c in word:
            if '\n' != c:
                index = dictionary[c]
                if index != 0:
                    word_as_index.append(index)
        output_corpus.append(word_as_index)
    return output_corpus


def index2onehot(index_word, vector_length, word_length=-1):

    word_length = len(index_word) if word_length < 0 else word_length
    onehot_word = torch.zeros(word_length, vector_length)

    for i, idx in enumerate(index_word):
        if i >= word_length:
            break
        onehot_word[i, idx] = 1

    return onehot_word

def indexCorpus2onehotCorpus(index_corpus, vector_length=-1, word_length=-1):

    if vector_length < 0:
        for word in index_corpus:
            for idx in word:
                if idx > vector_length:
                    vector_length = idx
        vector_length += 1

    if word_length < 0:
        word_length = max([len(word) for word in index_corpus])

    print(f"vector_length: {vector_length}")
    print(f"vector_length: {word_length}")

    onehot_words = []

    for index_word in index_corpus:
        if len(index_word) > 2:
            onehot_words.append(index2onehot(index_word, vector_length, word_length))

    return onehot_words, vector_length, word_length
