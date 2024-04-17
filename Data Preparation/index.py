import re
import nltk
nltk.download('punkt')

import emoji
import numpy as np
from nltk.tokenize import word_tokenize
from utils2 import get_dict

# Corpus Defn
corpus = 'Who ❤️ "word embeddings" in 2020? I do!!!'

# replace all interrupting punctuations w/ periods
print(f"Corpus:{corpus}")
data = re.sub(r"[,?!;-]+", '.', corpus)
print(f"Cleaned corpus: {data}")

# Split corpus into individual tokens
data = nltk.word_tokenize(data)
print(data)

# remove numbers, other punctuations(excepet '.') & lowercase all
data = [ch.lower() for ch in data
        if ch.isalpha()
        or ch == '.'
        or emoji.is_emoji(ch)
        ]
print(f"Clean corpus v1.2: {data}")

# make previous steps into a fucntion
def tokenize(corpus):
    data = re.sub(r"[,?!-;]+", '.', corpus)
    data = nltk.word_tokenize(data)
    data = [ch.lower() for ch in data
        if ch.isalpha()
        or ch == '.'
        or emoji.is_emoji(ch)
        ]
    return data
new_corpus = "Lewis Hamilton was robbed in the 2021 Abu Dhabi GP!!!"
words = tokenize(new_corpus)

# SLIDING Window of Words
def get_windows(words, C):
    i = C
    while i < len(words)-C:
        center_word= words[i]
        context_word = words[(i-C):i] + words[(i+1):(i+C+1)]
        yield context_word, center_word
        i+=1

for x,y in get_windows(tokenize(new_corpus), 2):
    print(f"{x}\t{y}")

# Transforing words into vectors for the training set
word2Ind, Ind2word = get_dict(words)
V = len(word2Ind)
def word_to_1hot_vector(word, word2Ind, V):
    one_hot_vector = np.zeros(V)
    one_hot_vector[word2Ind[word]]=1
    return one_hot_vector
context_words = ["lewis", "hamilton", "robbed", "in"]
context_wordVectors = [word_to_1hot_vector(w, word2Ind, V) for w in context_words]
print(context_wordVectors)
print("\n\n")

# Vector mean
def context_words_to_vector(context_words, word2Ind, V):
    context_wordVectors=[word_to_1hot_vector(w, word2Ind, V) for w in context_words]
    context_wordVectors = np.mean(context_wordVectors, axis=0)
    return context_wordVectors
print(context_words_to_vector(context_words, word2Ind, V))

# Building the training set
for context_words, center_word in get_windows(words, 2):
    print(f"Context words: {context_words} -> {context_words_to_vector(context_words,word2Ind, V)}")
    print(f"Center word: {center_word} -> {word_to_1hot_vector(center_word, word2Ind, V)}")
    print("\n\n")


# GENERATOR function
def get_training_example(words, C, word2Ind, V):
    for context_words, center_word in get_windows(words, C):
        yield context_words_to_vector(context_words, word2Ind, V), word_to_1hot_vector(center_word, word2Ind, V)
    
for context_words_vector, center_word_vector in get_training_example(words, 2, word2Ind, V):
    print(f'Context words vector:  {context_words_vector}')
    print(f'Center word vector:  {center_word_vector}')
    print()
