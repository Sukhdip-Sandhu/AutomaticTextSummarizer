"""
The following resources were used when creating the program:
https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/
https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70
"""

import pickle
import re
from os import path

import networkx as nx
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

SUMMARY_POINTS = 5


# open file and return content as a string
def open_file_to_summarize():
    with open('text_to_summarize.txt', encoding="utf8") as file:
        data = file.read().replace('\n', ' ')
    return data


# break string into list of string for each sentence
def tokenize_sentences(raw_text):
    sentences = [sent_tokenize(raw_text)]
    return sentences[0]


# clean the input data
def clean_text(sentences):
    clean_sentences = sentences.copy()
    for i in range(len(clean_sentences)):
        sentence = clean_sentences[i]
        sentence = re.sub("[^A-Za-z]+", ' ', sentence)
        sentence = sentence.lower()
        clean_sentences[i] = sentence
    return clean_sentences


# function to remove stopwords
def remove_stopwords(clean_sentences):
    stop_words = set(stopwords.words('english'))
    for i in range(len(clean_sentences)):
        tokens_without_sw = []
        for word in clean_sentences[i].split():
            if word not in stop_words:
                tokens_without_sw.append(word)
        filtered_sentence = " ".join(tokens_without_sw)
        clean_sentences[i] = filtered_sentence
    return clean_sentences


# create word vectors from pre trained GloVe dataset
def create_word_vectors():
    if not path.exists("data_300.pkl"):
        word_vectors = {}
        a_file = open("glove.6B.300d.txt", encoding='utf-8')
        for line in a_file:
            values = line.split()
            word_vectors[values[0]] = np.asarray(values[1:], dtype='float32')
        a_file.close()
        a_file = open("glove.6B.300d.pkl", "wb")
        pickle.dump(word_vectors, a_file)
        a_file.close()
    else:
        a_file = open("glove.6B.300d.pkl", "rb")
        word_vectors = pickle.load(a_file)
        a_file.close()
    return word_vectors


def alg(word_vectors, sentence_with_stopwords_removed, sentences):
    sentence_vectors = get_sentence_vectors(word_vectors, sentence_with_stopwords_removed)
    similarity_matrix = create_similarity_matrix(sentence_vectors, sentences)
    ranked_sentences = text_rank(similarity_matrix, sentences)
    ranked_sentences = ranked_sentences[::-1]
    return ranked_sentences


def get_sentence_vectors(word_vectors, sentence_with_stopwords_removed):
    sentence_vectors = []
    for i in sentence_with_stopwords_removed:
        if len(i) != 0:
            denominator = (len(i.split()) + 0.001)
            summation = 0
            for word in i.split():
                if word in word_vectors:
                    summation += word_vectors.get(word)
            v = summation / denominator
        else:
            v = np.zeros((300,))
        sentence_vectors.append(v)
    return sentence_vectors


def create_similarity_matrix(sentence_vectors, sentences):
    similarity_matrix = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = \
                    cosine_similarity(sentence_vectors[i].reshape(1, 300), sentence_vectors[j].reshape(1, 300))[0, 0]
    return similarity_matrix


def text_rank(similarity_matrix, sentences):
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    scores_list = []
    for value in scores.items():
        scores_list.append(value[1])

    scores_list, sentences = zip(*sorted(zip(scores_list, sentences)))
    ranked_sentences = sentences
    return ranked_sentences


def print_to_file(ranked_sentences, num_sentences):
    a_file = open("summarized_text.txt", "w")
    for i in range(num_sentences):
        a_file.write("• " + ranked_sentences[i] + "\n")
        print("• " + ranked_sentences[i] + "\n")
    a_file.close()


if __name__ == '__main__':
    raw_text = open_file_to_summarize()
    sentences = tokenize_sentences(raw_text)
    clean_sentences = clean_text(sentences)
    sentence_with_stopwords_removed = remove_stopwords(clean_sentences)
    word_vectors = create_word_vectors()
    ranked_sentences = alg(word_vectors, sentence_with_stopwords_removed, sentences)
    print_to_file(ranked_sentences, SUMMARY_POINTS)
