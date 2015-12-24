#!/usr/bin/python
# Module: topicmodels
# To call from the command line, run `python src/topicmodels <dr_model> <cluster_model>`.
# <dr_model> can be one of 'mnf', 'lda', 'lsa' or 'all'.
# <cluster_model> can be one of 'ms', 'sp', 'af', 'ag' or 'all'.

import sys, os, re, operator, string
import numpy as np
from util.Timer import Timer
from util import pickler
from collections import defaultdict, Counter
from Textrank.Units import EmailUnit, SentenceUnit

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import spectral_clustering, AffinityPropagation, MeanShift, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# --------------------------------------------
# Alter these accordingly:

num_clusters = 60
num_dr_components = 100

emails_source_file = 'Data/small_emails.txt'
feature_labels_source_file = 'Data/small_feature_labels.txt'
x_tfidf_source_file = 'Data/small_X_tfidf.matrix'

# --------------------------------------------

def dr_stage(abbrev, X_tfidf, featureNames):
	dr_models = {}

	if abbrev == 'mnf':

		name, X_mnf = mnf_model(X_tfidf, featureNames)
		dr_models[name] = X_mnf

	elif abbrev == 'lda':

		name, X_lda = lda_model(X_tfidf, featureNames)
		dr_models[name] = X_lda

	elif abbrev == 'lsa':

		name, X_lsa = lsa_model(X_tfidf, featureNames)
		dr_models[name] = X_lsa

	elif abbrev == 'all':

		name1, X_mnf = mnf_model(X_tfidf, featureNames)
		name2, X_lda = lda_model(X_tfidf, featureNames)
		name3, X_lsa = lsa_model(X_tfidf, featureNames)

		dr_models[name1] = X_mnf
		dr_models[name2] = X_lda
		dr_models[name3] = X_lsa

	else:
		print '%s is not a valid dimensionality reduction model'
		sys.exit()

	return dr_models


def clustering_stage(abbrev, dr_name, X, emails):
	if abbrev == 'ms':

		name = dr_name + '  -->  Mean Shift'
		labels, cluster_centers, numClusters = clusterEmails('ms', X)
		print_cluster_info(name, labels, numClusters, emails)

	elif abbrev == 'sp':

		name = dr_name + '  -->  Spectral Clustering'
		labels, cluster_centers, numClusters = clusterEmails('sp', X)
		print_cluster_info(name, labels, numClusters, emails)

	elif abbrev == 'af':

		name = dr_name + '  -->  Affinity Propagation'
		labels, cluster_centers, numClusters = clusterEmails('af', X)
		print_cluster_info(name, labels, numClusters, emails)

	elif abbrev == 'ag':

		name = dr_name + '  -->  Agglomerative Clustering'
		labels, cluster_centers, numClusters = clusterEmails('ag', X)
		print_cluster_info(name, labels, numClusters, emails)

	elif abbrev == 'all':

		name1 = dr_name + '  -->  Mean Shift'
		name2 = dr_name + '  -->  Spectral Clustering'
		name3 = dr_name + '  -->  Affinity Propagation'
		name4 = dr_name + '  -->  Agglomerative Clustering'

		labels1, cluster_centers1, numClusters1 = clusterEmails('ms', X)
		labels2, cluster_centers2, numClusters2 = clusterEmails('sp', X)
		labels3, cluster_centers3, numClusters3 = clusterEmails('af', X)
		labels4, cluster_centers4, numClusters4 = clusterEmails('ag', X)

		print_cluster_info(name1, labels1, numClusters1, emails)
		print_cluster_info(name2, labels2, numClusters2, emails)
		print_cluster_info(name3, labels3, numClusters3, emails)
		print_cluster_info(name4, labels4, numClusters4, emails)

	else:
		print '%s is not a valid cluster model'
		rsys.exit()


def mnf_model(X_tfidf, featureNames):
	name ='Non-Negative Matrix Factorization'
	timing = Timer(name)
	MNF_topics = dimensionalityReduction('mnf', X_tfidf, num_dr_components)
	print_top_words(name, MNF_topics, featureNames, 10)
	x_mnf = MNF_topics.transform(X_tfidf)
	timing.finish()
	return name, x_mnf

def lda_model(X_tfidf, featureNames):
	name = 'Latent Dirichlet Allocation'
	timing = Timer(name)
	LDA_topics = dimensionalityReduction('lda', X_tfidf, num_dr_components)
	print_top_words(name, LDA_topics, featureNames, 10)
	x_lda = LDA_topics.transform(X_tfidf)
	timing.finish()
	return name, x_lda

def lsa_model(X_tfidf, featureNames):
	name = 'Latent Semantic Analysis'
	timing = Timer(name)
	LSA_topics = dimensionalityReduction('lsa', X_tfidf, num_dr_components)
	print_top_words(name, LSA_topics, featureNames, 10)
	x_lsa = LSA_topics.transform(X_tfidf)
	timing.finish()
	return name, x_lsa

# Pass in 'mnf', 'lda' or 'lsa' as the modelType, the features X,
# the feature names and the number of desired components.
# For example: topicModel('mnf', X_tfidf, X_featureNames, 100)
def dimensionalityReduction(modelType, X, nComponents):
    
    if modelType == 'mnf':
        return NMF(n_components=nComponents, init='random', random_state=0).fit(X)
    elif modelType == 'lda':
        return LatentDirichletAllocation(n_topics=nComponents).fit(X)
    elif modelType == 'lsa':
        return TruncatedSVD(n_components=nComponents).fit(X)
        
    print 'Invalid dimensionality reduction model'
    return None


# Pass in 'ms', 'spectral' or 'affinity' as the modelType, the features X
# and it will return the models prescribed labels, cluster centers and
# number of clusters
def clusterEmails(modelType, X, nClusters=100):
    
    if modelType == 'ms':
        ms = MeanShift().fit(X)
        return ms.labels_, ms.cluster_centers_, len(ms.cluster_centers_)
    elif modelType == 'sp':
        symmetricMat = cosine_similarity(X[0:X.shape[0]], X)
        labels = spectral_clustering(symmetricMat, n_clusters=num_clusters)
        return labels, None, nClusters
    elif modelType == 'af':
        af = AffinityPropagation().fit(X)
        return  af.labels_, af.cluster_centers_, len(af.cluster_centers_)
    elif modelType == 'ag':
    	ag = AgglomerativeClustering(n_clusters=num_clusters).fit(X)
    	return ag.labels_, ag.cluster_centers_, len(ag.cluster_centers_)

    print 'Invalid cluster model argument'
    return None

# Print out the top words for the topic model:
def print_top_words(modelName, model, feature_names, n_top_words):
	numComponents = model.components_.shape[0]
	print '*************************************************'
	print 'TOPIC MODEL: ' + modelName + ' with %d components' % numComponents
	print '*************************************************'
	for topic_idx, topic in enumerate(model.components_):
		print(("Topic %d:   " % topic_idx) + " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
		print
	print

# Prints out the cluster model distribution and keywords:
def print_cluster_info(specs, labels, numClusters, emails):
    print '*********************************************'
    print 'TOPIC CLUSTER: ' + specs + ":  - generated %d clusters -" % numClusters
    print '*********************************************'
    distribution = getClusterDistribution(labels)
    print 'Cluster Distribution: '
    print sorted(distribution.values(), reverse=True)
    for cluster in range(numClusters):
    	print getClusterKeywords(emails, labels, cluster, distribution)
    	print
    print

# Extracts the most frequent Textrank keywords from the emails with the given
# cluster label. These words essentially give meaning or a human understandable
# "label" to the cluster. 
def getClusterKeywords(emails, labels, cluster, distribution):
    keywordFrequency = Counter()
    indexes = [ i for i, l in enumerate(labels) if l == cluster ]
    for index in indexes:
        keywordFrequency.update(emails[index].keywords)
    return ("Cluster %d: (%d emails)  \t" % (cluster, distribution[cluster])) + " ".join([ str(k[0]) for k in keywordFrequency.most_common(10) ])

# Returns a mapping from cluster index to number of emails in that cluster.
def getClusterDistribution(labels):
	distribution = defaultdict(int)
	for l in labels:
		distribution[l] += 1
	return distribution

################################################################################
# Module command-line behavior #
################################################################################

if __name__ == '__main__':

	dr_model = str(sys.argv[1]).lower()
	cluster_model = str(sys.argv[2]).lower()

	timer = Timer('Topic Modelling')

	emails = pickler.load(emails_source_file)
	featureNames = pickler.load(feature_labels_source_file)
	X_tfidf = pickler.load(x_tfidf_source_file)
	print X_tfidf.shape

	timer.markEvent('Loaded email feature data')

	X_dr = dr_stage(dr_model, X_tfidf, featureNames)

	timer.markEvent('Finished dimensionality reduction step')
	print 'Starting Topic Clustering'

	for dr_name in X_dr.keys():
		clustering_stage(cluster_model, dr_name, X_dr[dr_name], emails)

	timer.finish()