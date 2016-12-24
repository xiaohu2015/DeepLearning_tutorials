"""
The data used in RNN language model.
"""
import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime


def get_data(fileName='/data/reddit-comments-2015-08.csv', vocabulary_size = 8000, unknown_token = "UNKNOWN_TOKEN",
             sentence_start_token="SENTENCE_START", sentence_end_token = "SENTENCE_END"):
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading CSV file...")
    with open(sys.path[0]+fileName, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.__next__()
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
    print("\nExample sentence: '%s'" % sentences[0])
    print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])
    # get the training set
    X_train = []
    y_train = []
    for sen in tokenized_sentences:
        X_train.append(list([word_to_index[w] for w in sen[:-1]]))
        y_train.append(list([word_to_index[w] for w in sen[1:]]))

    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    x_example, y_example = X_train[17], y_train[17]
    print("x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example))
    print("\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example))
    return (X_train, y_train)

if __name__ == "__main__":
    X_train, y_train = get_data()
    print(type(X_train[0]))