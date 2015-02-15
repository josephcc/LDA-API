# coding: utf-8
import pdb 

import math
import string
import cPickle as pickle
from operator import *
from collections import Counter

from ScriptingBridge import *
from Foundation import *

import nltk
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
from nltk.tokenize.punkt import PunktWordTokenizer

import numpy as np
from scipy.spatial.distance import cosine

from gensim.models.ldamodel import LdaModel
lda = LdaModel.load('wikipedia.lda')
doc2bow = lda.id2word.doc2bow
def tfidf2bow(tfidf):
    bow = []
    for word, prob in tfidf.items():
        if lda.id2word.token2id.has_key(word):
            bow.append((lda.id2word.token2id[word], prob))
    return bow

from html2text import html2text
from readability.readability import Document

CACHE = {}

def clean_html(source):
    if len(source.strip()) == 0:
        return ''
    return html2text(Document(source).summary())
#    return nltk.clean_html(source)

def get_tokens(text):
    text = text.replace(u'’', "'")
    tokens = PunktWordTokenizer().tokenize(text)
    tokens = map(lambda x: x.strip(string.punctuation + string.digits), tokens)
    tokens = map(lambda x: x.lower(), tokens)
    tokens = filter(lambda x: '=' not in x, tokens)
    tokens = filter(lambda x: '.' not in x, tokens)
    tokens = filter(lambda x: 'e.g' not in x, tokens)
    tokens = filter(lambda x: 'i.e' not in x, tokens)
    tokens = filter(lambda x: 'php' not in x, tokens)
    tokens = filter(lambda x: 'html' not in x, tokens)
    tokens = filter(lambda x: 'gif' not in x, tokens)
    tokens = filter(lambda x: 'http' not in x, tokens)
    tokens = filter(lambda x: 'jquery' not in x, tokens)
    tokens = filter(lambda x: 'function' not in x, tokens)
    tokens = filter(lambda x: "'" not in x, tokens)
    tokens = filter(lambda x: '"' not in x, tokens)
    tokens = filter(lambda x: '/' not in x, tokens)
    tokens = filter(lambda x: len(x) > 2, tokens)
    tokens = filter(lambda x: len(x) < 20, tokens)
    tokens = filter(lambda x: x not in stopwords, tokens)
    return tokens

def get_windows():
    safari = SBApplication.applicationWithBundleIdentifier_('com.apple.Safari')
    windows = safari.windows()
    for wid, window in enumerate(windows):
        if len(window.tabs()) == 0:
            continue
        yield wid, window

def get_tabs(window):
    for tab in window.tabs():
        url = tab.URL()
        title = tab.name()
        source = tab.source()
        text = clean_html(source)
        tokens = get_tokens(text)
        yield url, title, source, text, tokens

__Web1T_TF__ = None
def get_Web1T_TF():
    global __Web1T_TF__
    if __Web1T_TF__ == None:
        __Web1T_TF__ = pickle.load(open('web1t_tf.pickle'))
    return __Web1T_TF__


def get_IDF():
    numberOfTabs = 0
    DF = Counter()
    for wid, window in get_windows():
        for url, title, source, text, tokens in get_tabs(window):
            numberOfTabs += 1
            uniqueTokens = set(tokens)
            DF.update(uniqueTokens)
    IDF = {}
    for token, df in DF.items():
        IDF[token] = math.log( float(numberOfTabs) / df )
    return IDF

def get_IDF_for_sources(sources):
    numberOfTabs = len(sources)
    DF = Counter()
    for tokens in sources:
        uniqueTokens = set(tokens)
        DF.update(uniqueTokens)
    IDF = {}
    for token, df in DF.items():
        IDF[token] = math.log( float(numberOfTabs) / df )
    return IDF

def get_TFIDF(IDF, isTF=False):
    tabnames = 0
    topic_vectors = []
    titles = []
    for wid, window in get_windows():
        print 'Window %d' % wid
        for url, title, source, text, tokens in get_tabs(window):
            TF = Counter(tokens)
            TFIDF = {}
            for token, tf in TF.items():
                if not IDF.has_key(token):
                    continue
                if isTF:
                    TFIDF[token] = tf / IDF[token]
                else:
                    TFIDF[token] = tf * IDF[token]
            keywords = sorted(TFIDF.items(), key=itemgetter(1), reverse=True)
            print '\tTAB%d "%s" - %s' % (tabnames, title, url[:30])
            tabnames += 1
            titles.append(title)
            print '\t  tfidf: ' + ', '.join(map(itemgetter(0), keywords[:16]))
            bow = doc2bow(tokens)
            tfidfbow = tfidf2bow(TFIDF)

            topics = lda[bow]
            topic_vector = map(itemgetter(1), lda.__getitem__(bow, eps=-1))
            topic_vectors.append(np.array(topic_vector))
            topics.sort(key=itemgetter(1), reverse=True)
            print '\t  lda:'
            for tid, prob in topics[:4]:
                print '\t        %.3f*topic%d:' % (prob, tid),
                print lda.print_topic(tid, topn=8)

            '''
            topics = lda[tfidfbow]
            topics.sort(key=itemgetter(1), reverse=True)
            print '\t  lda(tfidf):'
            for tid, prob in topics[:4]:
                print '\t        %.3f*topic%d:' % (prob, tid),
                print lda.print_topic(tid, topn=5)
            '''

            print

    print
    print 'Cosine simiarity matrix:'
    print '-' * 33

    print '\t',
    for i, v1 in enumerate(topic_vectors):
        print str(i).rjust(4), '\t',
    print
    for i, v1 in enumerate(topic_vectors):
        print str(i), '\t',
        for j, v2 in enumerate(topic_vectors):
            print '%.2f' % abs(1 - cosine(v1, v2)), '\t',
        print

    print
    for i, title in enumerate(titles):
        print i, title
        o
def get_LDA_for_tokens(tokens):
    bow = doc2bow(tokens)
    topics = lda[bow]
    topics.sort(key=itemgetter(1), reverse=True)
    topic_vector = map(itemgetter(1), lda.__getitem__(bow, eps=-1))
    out_lda = ''
    print '\t  lda:'
    for tid, prob in topics[:4]:
        print '\t        %.3f*topic%d:' % (prob, tid),
        print lda.print_topic(tid, topn=8)

        out_lda += ('%.3f*topic%d:' % (prob, tid))
        out_lda += lda.print_topic(tid, topn=5)
        out_lda += '\n'
    return out_lda, topic_vector

def get_TFIDF_for_tokens(tokens, IDF, isTF=False):
    tabnames = 0
    topic_vectors = []
    titles = []

    TF = Counter(tokens)
    TFIDF = {}
    for token, tf in TF.items():
        if not IDF.has_key(token):
            continue
        if isTF:
            TFIDF[token] = tf / IDF[token]
        else:
            TFIDF[token] = tf * IDF[token]
    keywords = sorted(TFIDF.items(), key=itemgetter(1), reverse=True)[:16]
    keywords = ['%.2f*%s' % (p, w) for w, p in keywords]
    print '\t  tfidf: ' + ', '.join(keywords)
    out_tfidf =  ', '.join(keywords)


    return out_tfidf

def run():
    IDF = get_IDF()
    get_TFIDF(IDF)

def run2():
    Web1T_TF = get_Web1T_TF()
    get_TFIDF(Web1T_TF, isTF=True)

import json
import pdb
from flask import Flask, request
import cPickle as pickle
app = Flask(__name__)
app.debug = True

@app.route('/', methods=['POST'])
def _index():
    groups = json.loads(request.data)
    # query -> url -> [0] -> html
    groups = process(groups)
    return json.dumps(groups)

@app.route('/searchInfo', methods=['POST'])
def _searchInfo():
    data = json.loads(request.form['data'])
    htmls = data['htmls']
    tokenss = [get_tokens(clean_html(html)) for html in htmls]


    IDF = get_IDF_for_sources(tokenss)
    topics, vector = get_LDA_for_tokens(reduce(add, tokenss))

    tfidfs = []
    for tokens in tokenss:
        tfidf = get_TFIDF_for_tokens(tokens, IDF)
        tfidfs.append(tfidf)

    return json.dumps({'tfidfs': tfidfs, 'lda': topics, 'lda_vector': vector})

def process(groups):
    for query in groups.keys():
        if len(query) < 2:
            continue
        alldocs = []
        for url in groups[query].keys():
            group = groups[query][url]
            if CACHE.has_key(url):
                groups[query][url][0]['tokens'] = CACHE[url]
            elif len(group) > 0 and group[0].has_key('html'):
                groups[query][url][0]['tokens'] = get_tokens(clean_html(group[0]['html']))
                CACHE[url] = groups[query][url][0]['tokens'] 
            alldocs.append(groups[query][url][0]['tokens'])
        IDF = get_IDF_for_sources(alldocs)

        for url in groups[query].keys():
            if len(groups[query][url]) > 0 and groups[query][url][0].has_key('html'):
                source = groups[query][url][0]['tokens']
                tfidf = get_TFIDF_for_tokens(source, IDF)
                source = groups[query][url][0]['tfidf'] = tfidf

        topics, vector = get_LDA_for_tokens(reduce(add, alldocs))
        groups[query]['LDA Topics'] = [{'tfidf': topics}]

        for url in groups[query].keys():
            if len(groups[query][url]) > 0 and groups[query][url][0].has_key('html'):
                del groups[query][url][0]['tokens']
                del groups[query][url][0]['html']
    return groups

if __name__ == '__main__':
    app.run()
    #run()
    #print '-' * 33
    #run2()
    #groups = pickle.load(open('data.pickle'))
    #groups = process(groups)

