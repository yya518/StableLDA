import operator
import io

from gensim.parsing.preprocessing import strip_punctuation, strip_numeric, remove_stopwords, stem, strip_short
from gensim.parsing.preprocessing import preprocess_string
from gensim.corpora import Dictionary

'''
given a textual corpus, create vocabulary and convert textual corpus into a cleaned one
'''

class Dataset:
    '''
    self.id2word: gensim.corpora.dictionary
    self.text: list of list of words, sequential representation of document, used for obtaining word embedding
    '''
    def __init__(self, filepath, num_words):
        docs = []

        with io.open(filepath, 'r', encoding='utf-8') as f: # this may encounter utf-8 encoding error
            docs = [line for line in f.read().splitlines()]
        print(len(docs))

        # preprocessing and tokenization
        CUSTOMER_FILTERS = [lambda x: x.lower(), strip_punctuation, strip_numeric, remove_stopwords, lambda x: strip_short(x, 3)]
        preprocess_func = lambda x: preprocess_string(x, CUSTOMER_FILTERS)
        docs = list(map(preprocess_func, docs))  # list of list of words, since gensim dictionary requires this format

        # generate dictionary
        self.id2word = Dictionary(docs)
        self.id2word.filter_extremes(no_below=3, no_above=0.25, keep_n=num_words)
        self.id2word.compactify()
        print('vocabulary size:', len(self.id2word))

        # generate sequence
        seq_func = lambda x: [w for w in x if w in self.id2word.token2id]
        self.text = list(map(seq_func, docs))
        self.text = [doc for doc in self.text if len(doc) > 0] # remove empty doc
        print('corpus size:', len(self.text))

    def save_data(self, bow_file, vocab_file):
        with io.open(bow_file, 'w', encoding='utf-8') as f:
            for doc in self.text:
                f.write(' '.join(doc) + '\n')
        vocab = [i[0] for i in sorted(self.id2word.token2id.items(), key=operator.itemgetter(1))]
        with io.open(vocab_file, 'w', encoding='utf-8') as f:
            for v in vocab:
                f.write(v + '\n')

