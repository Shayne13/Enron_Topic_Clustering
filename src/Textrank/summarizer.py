from math import log10 as _log10
import math
import itertools
from pagerank_weighted import pagerank_weighted_scipy as _pagerank
from textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from commons import build_graph as _build_graph
from commons import remove_unreachable_nodes as _remove_unreachable_nodes
import numpy as np

from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
# from nltk import word_tokenize

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']

def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']

def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None

def _set_lex_graph_edge_weights(graph):
    for su1 in graph.nodes():
        for su2 in graph.nodes():

            edge = (su1, su2)
            if su1 != su2 and not graph.has_edge(edge):
                similarity = _get_lex_similarity(su1, su2)
                if similarity != 0:
                    graph.add_edge(edge, similarity)

def _set_graph_edge_weights(graph):
    for su1 in graph.nodes():
        for su2 in graph.nodes():

            edge = (su1, su2)
            if su1 != su2 and not graph.has_edge(edge):
                similarity = _get_similarity(su1, su2)
                if similarity != 0:
                    graph.add_edge(edge, similarity)

def _get_lex_similarity(su1, su2):

    # tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    # v1 = tfidf.fit_transform(su1.text)
    # v2 = tfidf.fit_transform(su2.text)
    d1 = {}
    for w in tokenize(su1.text):
        if w not in d1:
            d1[w] = 0.0
        d1[w] += 1.0
    d2 = {}
    for w in tokenize(su2.text):
        if w not in d2:
            d2[w] = 0.0
        d2[w] += 1.0
    v1 = []
    v2 = []
    for w in d1.keys():
        if w in d2.keys():
            v1.append(d1[w])
            v2.append(d2[w])

    if v1 and v2:
        return 1.0 - cosine_distance(v1, v2)
    else:
        return 0.0


def cosine_distance(u, v):
    """
    Returns the cosine of the angle between vectors v and u. This is equal to
    u.v / |u||v|.
    """
    return np.dot(u, v) / (math.sqrt(np.dot(u, u)) * math.sqrt(np.dot(v, v)))

def tokenize(text):
    tokens = word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

def _get_similarity(su1, su2):

    words1 = [ w for w in su1.processed if w[1] not in ['NNP', 'NNPS'] ]
    words2 = [ w for w in su2.processed if w[1] not in ['NNP', 'NNPS'] ]
    properNouns1 = [ w for w in su1.processed if w[1] in ['NNP', 'NNPS'] ]
    properNouns2 = [ w for w in su2.processed if w[1] in ['NNP', 'NNPS'] ]

    log_s1 = _log10(len(su1.processed)+1)
    log_s2 = _log10(len(su2.processed)+1)
    if log_s1 + log_s2 == 0:
        return 0

    eo = entail_overlap(words1, words2)
    no = noun_overlap(properNouns1, properNouns2)
    x = (eo + no) / (log_s1 + log_s2)

    # if su1.index <= 3 and su2.index <= 5:
    #     print "HELLO!!!!!!!"
    #     print su1.text
    #     print su2.text
    #     print _count_common_words(su1.text.split(), su2.text.split())
    #     print log_s1, log_s2
    #     print eo, no
    return x


def entail_overlap(words1, words2):
    overlap = 0
    for (w1, w2) in itertools.product(words1, words2):
        w1Lemmas = [ l for ss in wn.synsets(w1[0], penn_to_wn(w1[1])) for l in ss.lemmas() ]
        w2Lemmas = [ l for ss in wn.synsets(w2[0], penn_to_wn(w2[1])) for l in ss.lemmas() ]
        for l1 in w1Lemmas:
            if l1 in w2Lemmas:
                overlap += 1
                break
    return overlap

def noun_overlap(pn1, pn2):
    overlap = 0
    for (w1, w2) in itertools.product(pn1, pn2):
        if w1[0] == w2[0]:
            overlap += 1
    return overlap


def _count_common_words(words_sentence_one, words_sentence_two):
    return len(set(words_sentence_one) & set(words_sentence_two))


def _format_results(extracted_sentences, split, score):
    if score:
        return [(sentence.text, sentence.score) for sentence in extracted_sentences]
    if split:
        return [sentence.text for sentence in extracted_sentences]
    return "\n".join([sentence.text for sentence in extracted_sentences])


def _add_scores_to_sentences(sentences, scores):
    for sentence in sentences:
        # Adds the score to the object if it has one.
        if sentence.token in scores:
            sentence.score = scores[sentence.token]
        else:
            sentence.score = 0


def _get_sentences_with_word_count(sentences, words):
    """ Given a list of sentences, returns a list of sentences with a
    total word count similar to the word count provided.
    """
    word_count = 0
    selected_sentences = []
    # Loops until the word count is reached.
    for sentence in sentences:
        words_in_sentence = len(sentence.text.split())

        # Checks if the inclusion of the sentence gives a better approximation
        # to the word parameter.
        if abs(words - word_count - words_in_sentence) > abs(words - word_count):
            return selected_sentences

        selected_sentences.append(sentence)
        word_count += words_in_sentence

    return selected_sentences


def _extract_most_important_sentences(sentences, ratio, words):
    sentences.sort(key=lambda s: s.score, reverse=True)

    # If no "words" option is selected, the number of sentences is
    # reduced by the provided ratio.
    if words is None:
        length = len(sentences) * ratio
        return sentences[:int(length)]

    # Else, the ratio is ignored.
    else:
        return _get_sentences_with_word_count(sentences, words)


def summarize(text, ratio=0.2, words=None, language="english", split=False, scores=False):
    # Gets a list of processed sentences.
    sentences = _clean_text_by_sentences(text, language)

    # Creates the graph and calculates the similarity coefficient for every pair of nodes.
    graph = _build_graph([sentence.token for sentence in sentences])
    _set_graph_edge_weights(graph)

    # Remove all nodes with all edges weights equal to zero.
    _remove_unreachable_nodes(graph)

    # Ranks the tokens using the PageRank algorithm. Returns dict of sentence -> score
    pagerank_scores = _pagerank(graph)

    # Adds the summa scores to the sentence objects.
    _add_scores_to_sentences(sentences, pagerank_scores)

    # Extracts the most important sentences with the selected criterion.
    extracted_sentences = _extract_most_important_sentences(sentences, ratio, words)

    # Sorts the extracted sentences by apparition order in the original text.
    extracted_sentences.sort(key=lambda s: s.index)

    return _format_results(extracted_sentences, split, scores)


def get_graph(text, language="english"):
    sentences = _clean_text_by_sentences(text, language)

    graph = _build_graph([sentence.token for sentence in sentences])
    _set_graph_edge_weights(graph)

    return graph

