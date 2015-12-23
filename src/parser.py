#!/usr/bin/python
# Module: parser
# To call from the command line, run `python src/parser <raw_data_rootpath>`, where
# <raw_data_rootpath> is the root path of the raw data such that all email account
# folders are directly inside. EG: 'python src/parser Data/Raw'

import sys, os, re, operator, string
import numpy as np
import nltk
from collections import defaultdict, Counter
from Textrank import textrank
from Textrank.Units import EmailUnit, SentenceUnit
from util.Timer import Timer
from util import pickler
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# Loading Sentence Detector and Stop Words List
sentenceDetector = nltk.data.load('tokenizers/punkt/english.pickle')
stopwords = stopwords.words('english')

# Returns a list of all emails in the Enron Email Dataset:
def parse_all_email_accounts(rootpath):
    allEmails = []
    accountpaths = [path for path in listdir_nohidden(rootpath)]
    timer = Timer('to parse %d email accounts' % len(accountpaths))
    for i, accountpath in enumerate(accountpaths):
        allEmails += parseEmailAccount(accountpath)
        print 'X'
        if ((i+1) % 10) == 0:
            timer.markEvent('Parsed %d of %d accounts' % (i+1, len(accountpaths)))

    timer.finish()
    return allEmails

# Takes in the root path to an email account and returns all EmailUnits in that account.
def parseEmailAccount(rootpath):
    emails = filter(None, [parseEmail(path, rootpath[rootpath.rfind('/') + 1:]) for path in listdir_allsubpaths(rootpath)])
    return emails

# Yields all non-hidden files/subdirectories (one level down) of a directory.
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield '{0}/{1}'.format(path, f)

# Yields all non-hidden leaf files any level down within a directory.
def listdir_allsubpaths(path):
    for root, dirs, files in os.walk(path):
        for filename in files:
            if not filename.startswith('.'):
                yield os.path.join(root, filename)
        	

# Takes in an emails file path and the account owner and returns the EmailUnit.
def parseEmail(path, owner):
    
    with open(path, 'r') as f:
        contents = [l for l in f]
        
    # Find the line numbers of the corresponding fields:
    senderIndex = [i for i, l in enumerate(contents) if re.search('(From:)', l)][0]
    recipientIndex = [i for i, l in enumerate(contents) if re.search('(X-To:)', l)][0]
    subjectIndex = [i for i, l in enumerate(contents) if re.search('(Subject:)', l)][0]
    textIndex = [i for i, l in enumerate(contents) if re.search('(X-FileName:)', l)][0] + 2
    
    # Extract Email metadata:
    sender = re.search('(From:\s)([^\n]+)', contents[senderIndex]).group(2).strip()
    recipient = re.search('(To:\s)([^\n]+)', contents[recipientIndex]).group(2).strip()
    subject = re.search('(Subject:\s)([^\n]+)', contents[subjectIndex]).group(2).strip()
    
    # Lines that match these regexes should be removed:
    removeRegex = compileRegex(['----------------------\sForwarded\sby',
                               '-----Original\sMessage-----',
                               'From:\s',
                               'Sent:\s',
                               'To:\s',
                               'cc:\s',
                               'bcc:\s',
                               '[IMAGE]',
                               'Subject:\s',
                               ])
    
    # clean and process the body of the email
    body = contents[textIndex:]
    body.append(subject)
    cleaned = filter(lambda l: re.search(removeRegex, l) == None, body)
    text = " ".join(cleaned).replace("\n", "").strip()
    
    # Extract TextRank keywords for the email
    sentences = sentenceDetector.tokenize(text)
    sentenceUnits = [ SentenceUnit(s) for s in sentences ]
    keyWords = textrank.textrank_keyword(sentenceUnits)
    
    # Remove stopwords for processed text (needs to be done after TextRank)
    words = word_tokenize(text)
    filtered = filter_words(words)
    processed = " ".join([ w for w in filtered if w not in stopwords ])

    return EmailUnit(owner, sender, recipient, subject, text, processed, keyWords)


# Takes a list of regexs and returns one regex that
# will capture any of the regexs in the list (OR).
def compileRegex(regex_list):
    return "({0})".format("|".join(regex_list))

# Returns a list of the human names in a body of text:
def listNames(text):
    names = set()
    for sent in sentenceDetector.tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'node'):
                if chunk.node == 'PERSON':
                    names.add(' '.join(c[0] for c in chunk.leaves()))
    return names

def apply_filters(sentence, filters):
    for f in filters:
        sentence = f(sentence)
    return sentence

def filter_words(sentences):
    filters = [lambda x: x.lower(), strip_numeric, strip_punctuation, lambda x: str(x)]

    apply_filters_to_token = lambda token: apply_filters(token, filters)
    return map(apply_filters_to_token, sentences)

# Taken from gensim
RE_PUNCT = re.compile('([%s])+' % re.escape(string.punctuation), re.UNICODE)
def strip_punctuation(s):
    s = to_unicode(s)
    return RE_PUNCT.sub(" ", s)


# Taken from gensim
RE_NUMERIC = re.compile(r"[0-9]+", re.UNICODE)
def strip_numeric(s):
    s = to_unicode(s)
    return RE_NUMERIC.sub("", s)

# Taken from gensim
def to_unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


################################################################################
# Module command-line behavior #
################################################################################

if __name__ == '__main__':

    timer = Timer('processing raw email data.')
    rawDataRootPath = str(sys.argv[1])
    emails = parse_all_email_accounts(rawDataRootPath)
    # emails = parseEmailAccount(rawDataRootPath)

    timer.markEvent('Emails fully parsed.')
    vectorizer = TfidfVectorizer(min_df=1)
    X_tfidf = vectorizer.fit_transform([email.processed for email in emails])
    featureNames = vectorizer.get_feature_names()
    timer.markEvent('TF-IDF matrix constucted.')

    saveLocation = 'Data/'
    pickler.save(emails, saveLocation + 'emails.txt')
    pickler.save(featureNames, saveLocation + 'feature_labels.txt')
    pickler.save(X_tfidf, saveLocation + 'x_tfidf.matrix')
    timer.finish()
