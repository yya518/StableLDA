from gensim.models import Word2Vec
from gensim.models import FastText
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import defaultdict, Counter
import numpy as np
import random
import operator
from scipy.stats import binom, binom_test
import os
import sys
import io


class StableLDA():

    def __init__(self, num_topics, num_words, alpha, beta, eta, rand_seed, output_dir, embed_method='cbow'):
        print('--------running Stable LDA model----------')
        self.num_topics = num_topics
        self.num_words = num_words
        self.num_cluster = self.num_topics + 10
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.rand_seed = rand_seed
        self.embed_method = embed_method
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_data(self, bow_file, vocab_file):
        print('--------- loading data ----------------')
        self.text = []
        with io.open(bow_file, 'r', encoding='utf-8') as f:
            self.text = [line.split() for line in f.read().splitlines()]

        self.vocab = []
        with io.open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = [line for line in f.read().splitlines()]

        self.vocab2id = {}
        for idx, v in enumerate(self.vocab):
            self.vocab2id[v] = idx

        self.num_words = len(self.vocab)  # update number of words, sometimes after preprocessing, vocab may shrink

        self.bow = []
        for doc in self.text:
            self.bow.append([self.vocab2id[w] for w in doc])

        self.bow_file = bow_file
        self.vocab_file = vocab_file
        self.cluster_file = self.output_dir + 'cluster.dat'
        self.sample_file = self.output_dir + 'z.dat'

    def train(self, bow_file, vocab_file, epochs):
        self.load_data(bow_file, vocab_file)
        self.init_word_cluster(self.embed_method)
        self.initialize()
        self.save_intermediate()
        self.inference(epochs)

    def save_intermediate(self):
        with open(self.sample_file, 'w') as f:  # save intermediate samples
            for sample in self.zsamples:
                f.write(' '.join([str(z) for z in sample]) + '\n')
        with io.open(self.cluster_file, 'w', encoding='utf-8') as f:  # save word topical clusters
            for cluster in self.topical_clusters:
                f.write(','.join(cluster) + '\n')
                
    def init_word_cluster(self, embed_method):
        self.w2v_model = None
        if embed_method == 'cbow':
            self.w2v_model = Word2Vec(sentences=self.text, size=100, window=3, min_count=5, workers=1, iter=10, sg=0)
        elif embed_method == 'skip-gram':
            self.w2v_model = Word2Vec(sentences=self.text, size=100, window=3, min_count=5, workers=1, iter=10, sg=1)
        elif embed_method == 'fasttext':
            self.w2v_model = FastText(sentences=self.text, size=100, window=3, min_count=5, workers=1, iter=10)

        vocabs = self.w2v_model.wv.vocab.keys()
        embeddings = [self.w2v_model.wv[w] for w in vocabs]
        self.embeddings = embeddings
        self.embeddings = [i / np.linalg.norm(i) for i in embeddings]
        ## kmeans clustering has randomness, and we want to eliminate randomness caused by kmeans
        self.kmeans = KMeans(n_clusters=self.num_cluster, algorithm='full', n_init=10, init='k-means++',random_state=0).fit(self.embeddings)
        #self.kmeans = AgglomerativeClustering(n_clusters=self.num_cluster).fit(self.embeddings)


    def initialize(self):
        '''
        this method uses the topical clusters to guide stable lda, builiding constraints for dirichlet forest prior
        '''
        self.topical_clusters = []
        vocabs = list(self.w2v_model.wv.vocab.keys())
        for cluster_id in range(self.num_cluster):
            temp = []
            for idx, label in enumerate(self.kmeans.labels_):  # for each word
                if cluster_id == label:
                    temp.append(vocabs[idx])
            self.topical_clusters.append(temp) #topical_clusters is a list of list of words.
            
        self.doc_topic = []
        self.word_topic = defaultdict(lambda: defaultdict(int))
        self.doc_assignments = []
        self.topic_word = defaultdict(lambda: defaultdict(float))
        self.zsamples = []

        # for each document, find most significant topic, first, for each topic, count the occurance of words in that topic
        self.word2part = defaultdict(int)
        for idx, words in enumerate(self.topical_clusters):
            prob = max(min(10 * self.eta / len(self.vocab) * 0.9, 0.9),0.1)
            for w in words:
                if random.random() < prob:
                    self.word2part[self.vocab2id[w]] = idx
        topic_pr = Counter()
        for doc in self.bow:
            for w in doc:  # for each word in doc, find the partition index
                if w in self.word2part:
                    topic_pr[self.word2part[w]] += 1
        num_words = np.sum(list(topic_pr.values()))

        for k, v in topic_pr.items():
            topic_pr[k] = float(v) / num_words

        # if topic only used in very few documents, treat it as invalid_topics.
        invalid_topics = set(
            [i[0] for i in sorted(topic_pr.items(), key=operator.itemgetter(1), reverse=True)[self.num_topics:]])

        max_doc_size = np.max([len(doc) for doc in self.bow])

        # since the number of topical cluster is greater than the number of topics, we need to ``ignore those less frequent''
        # topical clusters.
        # more frequent topics are examined over the corpus. For each word in a doc, if the probability of sampling
        # topic t is greater than independently sampling from topic_pr, it would be considered as a frequent topic.
        # we use binom_test() for this purpose. Once we obtain
        # the frequent topics, we move the mass from less frequent topics to the frequent topics.
        doc_valid_topics = []
        for doc in self.bow: # for each document, examine frequent topics
            topic_usage = defaultdict(int)
            smallest_pvalue = 1.0001
            smallest_pvalue_ok = 1.0001
            valid_topic = -1
            valid_topic_ok = -1

            for w in doc:
                if w in self.word2part:
                    topic_usage[self.word2part[w]] += 1

            for k, v in topic_usage.items():
                binom_pvalue = binom_test(v, len(doc), topic_pr[k], alternative='greater')

                if binom_pvalue < smallest_pvalue_ok and k not in invalid_topics:
                    valid_topic_ok = k
                    smallest_pvalue_ok = binom_pvalue

                if binom_pvalue < smallest_pvalue:
                    valid_topic = k
                    smallest_pvalue = binom_pvalue

            if valid_topic_ok != -1:
                doc_valid_topics.append(valid_topic_ok)
            else:
                doc_valid_topics.append(valid_topic)

        valid_topic_counter = Counter()
        for idx in doc_valid_topics:
            valid_topic_counter[idx] += 1

        # make sure valid_topic only contains T topics
        # it's okay that valid_topic contains -1 topic, which indicates a non-frequent topic

        if len(set(valid_topic_counter)) < self.num_topics:
            print('error: valid topic is less than the pre-defined topics')
            sys.exit()

        useless_topics = []
        if len(set(valid_topic_counter)) > self.num_topics:
            useless_count = len(set(valid_topic_counter)) - self.num_topics + 1

            sorted_counter = valid_topic_counter.most_common()
            sorted_counter.reverse()
            useless_topics = [k[0] for k in sorted_counter[:useless_count]]

        for idx, t in enumerate(doc_valid_topics):
            if t in useless_topics:
                doc_valid_topics[idx] = -1
        valid_topic_counter = Counter()
        for idx in doc_valid_topics:
            valid_topic_counter[idx] += 1

        valid_topics = set()
        for idx, doc in enumerate(self.bow):
            valid_topics.add(doc_valid_topics[idx])

        new_hard_mems = {}
        for k, v in self.word2part.items():
            if v in valid_topics:
                new_hard_mems[k] = v

        self.word2part = new_hard_mems

        for idx, valid_topic in enumerate(doc_valid_topics):
            if valid_topic == -1:
                # this is to assign -1 topic to the maximum topic possible (which is num_cluster +1)
                valid_topic = self.num_cluster + 1
                doc_valid_topics[idx] = valid_topic

        topic_counts = Counter()
        for topic in doc_valid_topics:
            topic_counts[topic] += 1

        for idx, doc in enumerate(self.bow):
            topic_dist = defaultdict(float)
            for w in doc:
                if w in self.word2part:
                    topic_num = self.word2part[w]
                    topic_dist[topic_num] += float(1) / len(doc)
                    if w not in self.word_topic:
                        self.word_topic[w] = defaultdict(int)
                    self.word_topic[w][topic_num] += 1

            valid_topic = doc_valid_topics[idx]
            for w in doc:
                if w not in self.word2part:
                    topic_dist[valid_topic] += float(1) / len(doc)
                    if w not in self.word_topic:
                        self.word_topic[w] = defaultdict(int)
                    self.word_topic[w][valid_topic] += 1
            self.doc_topic.append(topic_dist)

        topic_counts = Counter()
        for word, v in self.word_topic.items():
            for topic, count in v.items():
                topic_counts[topic] += 1

        pt = defaultdict(float)  # k:topic v:proportion in the corpus
        for word, topic_count in self.word_topic.items():
            for topic, count in topic_count.items():
                pt[topic] += count

        for idx, doc in enumerate(self.bow):
            self.doc_assignments.append(defaultdict(int))

        for idx, doc in enumerate(self.bow):
            valid_topic = doc_valid_topics[idx]
            for w in doc:  # topic num is either the valid topic or the topic the word belongs to originally
                topic_num = valid_topic
                if w in self.word2part:
                    topic_num = self.word2part[w]

                self.doc_assignments[idx][w] = topic_num

        pt = defaultdict(float)

        for word, topic_count in self.word_topic.items():
            for topic, count in topic_count.items():
                pt[topic] += count

        # calcuclate topic-word distribution
        for word, v in self.word_topic.items():  # k is vocab index, v is a dictionary where k is topic, and v is the count
            for topic, count in v.items():
                if topic not in self.topic_word:
                    self.topic_word[topic] = defaultdict(float)
                self.topic_word[topic][word] += count

        # normalize
        for topic, v in self.topic_word.items():  # k is topic, v is a dictionary where k is word, and v is count
            num_words = np.sum(list(v.values()))
            for word, count in v.items():
                self.topic_word[topic][word] = float(count) / num_words
            pt[topic] = num_words

        num_words = np.sum(list(pt.values()))
        for topic, count in pt.items():
            pt[topic] = float(count) / num_words

        # make topic names consecutive, sorting topic names so that they start from zero and there are no gaps
        old_to_new_labels = defaultdict(int)
        for topic, v in self.topic_word.items():
            old_to_new_labels[topic] = len(old_to_new_labels)

        for idx, doc in enumerate(self.bow):
            new_topic_distr = defaultdict()
            for topic, prob in self.doc_topic[idx].items():
                new_topic_distr[old_to_new_labels[topic]] = prob
            self.doc_topic[idx] = new_topic_distr

        new_topic_word = defaultdict(lambda: defaultdict(float))
        for topic, v in self.topic_word.items():
            new_topic = old_to_new_labels[topic]
            new_topic_word[new_topic] = v
        self.topic_word = new_topic_word

        for idx, doc in enumerate(self.doc_assignments):
            for w, topic in doc.items():
                self.doc_assignments[idx][w] = old_to_new_labels[topic]

        new_pt = defaultdict(float)
        for topic, mass in pt.items():
            new_pt[old_to_new_labels[topic]] = mass
        pt = new_pt

        for idx, doc in enumerate(self.bow):
            samples = []
            for w in doc:
                topic = self.doc_assignments[idx][w]
                samples.append(topic)
            self.zsamples.append(samples)


    def inference(self, epochs):
        # make sure argument values are correctly setup before passing to C++ main function
        #cmd = 'train'    # windows
        cmd = './train'  # linux
        cmd += ' -f {}'.format(self.bow_file)
        cmd += ' -v {}'.format(self.vocab_file)
        cmd += ' -c {}'.format(self.cluster_file)
        cmd += ' -z {}'.format(self.sample_file)
        cmd += ' -t {}'.format(self.num_topics)
        cmd += ' -w {}'.format(self.num_words)
        cmd += ' -a {}'.format(self.alpha)
        cmd += ' -b {}'.format(self.beta)
        cmd += ' -e {}'.format(self.eta)
        cmd += ' -n {}'.format(epochs)
        cmd += ' -r {}'.format(self.rand_seed)
        cmd += ' -o {}'.format(self.output_dir)

        print(cmd)
        os.system(cmd)