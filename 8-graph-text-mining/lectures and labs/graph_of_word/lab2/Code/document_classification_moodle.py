import pandas as pd
from sklearn import svm
from sklearn import metrics
from library import *

##################################
# data loading and preprocessing #
##################################

train = pd.read_csv("webkb-train-stemmed.txt", header=None, delimiter="\t")
print train.shape

test = pd.read_csv("webkb-test-stemmed.txt", header=None, delimiter="\t")
print test.shape

# inspect head of data frames
print "first five rows of training data:"
print train.ix[:5,:]

print "first five rows of testing data:"
print test.ix[:5,:]

# get index of empty (nan) and less than four words documents
index_remove = [i for i in range(len(train.ix[:,1])) if (train.ix[i,1]!=train.ix[i,1]) or ((train.ix[i,1]==train.ix[i,1])and(len(train.ix[i,1].split(" "))<4))]

# remove those documents
print "removing", len(index_remove), "documents from training set"
train = train.drop(train.index[index_remove])
print train.shape

# repeat above steps for testing set
index_remove = [i for i in range(len(test.ix[:,1])) if (test.ix[i,1]!=test.ix[i,1]) or ((test.ix[i,1]==test.ix[i,1])and(len(test.ix[i,1].split(" "))<4))]
print "removing", len(index_remove), "documents from testing set"
test = test.drop(test.index[index_remove])
print test.shape

labels = train.ix[:,0]
unique_labels = list(set(labels))

truth = test.ix[:,0]
unique_truth = list(set(truth))

print "number of observations per class:"
for label in unique_labels:
    print label, ":", len([temp for temp in labels if temp==label])

print "storing terms from training documents as list of lists"
terms_by_doc = [document.split(" ") for document in train.ix[:,1]]
n_terms_per_doc = [len(terms) for terms in terms_by_doc]

print "storing terms from testing documents as list of lists"
terms_by_doc_test = [document.split(" ") for document in test.ix[:,1]]

print "min, max and average number of terms per document:", min(n_terms_per_doc), max(n_terms_per_doc), sum(n_terms_per_doc)/len(n_terms_per_doc)

# store all terms in list
all_terms = [terms for sublist in terms_by_doc for terms in sublist]

# compute average number of terms
avg_len = sum(n_terms_per_doc)/len(n_terms_per_doc)

# unique terms
all_unique_terms = list(set(all_terms))

# store IDF values in dictionary
n_doc = len(labels)

idf = dict(zip(all_unique_terms,[0]*len(all_unique_terms)))
counter = 0

for element in idf.keys():
    # number of documents in which each term appears
    df = sum([element in terms for terms in terms_by_doc])
    # idf
    idf[element] = math.log10(float(n_doc+1)/df)
    
    counter+=1
    if counter % 200 == 0:
        print counter, "terms have been processed"
        
############
# training #
############

print "creating a graph-of-words for each training document \n"

# hint: use the terms_to_graph function (found in library.py) with a window of size 3
# loop over all documents (list of lists of terms) and store the graphs in a list called all_graphs

###################
#                 #
# YOUR CODE HERE  #
#                 #
###################

# sanity checks (should return True)
print len(terms_by_doc)==len(all_graphs)
print len(set(terms_by_doc[0]))==len(all_graphs[0].vs)

print "computing vector representations of each training document"

b = 0.003
features_degree = []
features_w_degree = []
features_closeness = []
features_w_closeness = []
features_tfidf = []

len_all = len(all_unique_terms)

counter = 0

idf_keys = idf.keys()

for i in xrange(len(all_graphs)):
    
    graph = all_graphs[i]
    terms_in_doc = terms_by_doc[i]
    doc_len = len(terms_in_doc)
    
    # returns node (1) name, (2) degree, (3) weighted degree, (4) closeness, (5) weighted closeness
    metrics = compute_node_centrality(graph)
    
    feature_row_degree = [0]*len_all
    feature_row_w_degree = [0]*len_all
    feature_row_closeness = [0]*len_all
    feature_row_w_closeness = [0]*len_all
    feature_row_tfidf = [0]*len_all
    
    for term in list(set(terms_in_doc)):
        index = all_unique_terms.index(term)
        idf_term = idf[term]
        denominator = (1-b+(b*(float(doc_len)/avg_len)))
        metrics_term = [tuple[1:5] for tuple in metrics if tuple[0]==term][0]
        
        # store TW-IDF values
        feature_row_degree[index] = (float(metrics_term[0])/denominator) * idf_term
        feature_row_w_degree[index] = (float(metrics_term[1])/denominator) * idf_term
        feature_row_closeness[index] = (float(metrics_term[2])/denominator) * idf_term
        feature_row_w_closeness[index] = (float(metrics_term[3])/denominator) * idf_term
        
        # number of occurences of word in document
        tf = terms_in_doc.count(term)
        
        # store TF-IDF value
        feature_row_tfidf[index] = ((1+math.log1p(1+math.log1p(tf)))/(1-0.2+(0.2*(float(doc_len)/avg_len)))) * idf_term
    
    features_degree.append(feature_row_degree)
    features_w_degree.append(feature_row_w_degree)
    features_closeness.append(feature_row_closeness)
    features_w_closeness.append(feature_row_w_closeness)
    features_tfidf.append(feature_row_tfidf)

    counter += 1
    if counter % 100 == 0:
        print counter, "documents have been processed"

# convert list of lists into array
# documents as rows, unique words as columns (i.e., document-term matrix)
training_set_tfidf = numpy.array(features_tfidf)
training_set_degree = numpy.array(features_degree)
training_set_w_degree = numpy.array(features_w_degree)
training_set_closeness = numpy.array(features_closeness)
training_set_w_closeness = numpy.array(features_w_closeness)

# convert labels into integers then into column array
labels = list(labels)

labels_int = [0] * len(labels)
for j in range(len(unique_labels)):
    index_temp = [i for i in range(len(labels)) if labels[i]==unique_labels[j]]
    for element in index_temp:
        labels_int[element] = j

# check that coding went smoothly
zip(labels_int,labels)[:20]

labels_array = numpy.array(labels_int)


print "initializing classifiers"

# hint: use the svm.LinearSVC() function (sklearn) to initialize a SVM with linear Kernel for each training set
# you can name the classifiers "classifier_nameoftechnique", e.g., classifier_tfidf, classifier_degree, etc.

###################
#                 #
# YOUR CODE HERE  #
#                 #
###################


print "training classifiers"

# hint: use the .fit() attribute of the classifier. You need to pass both the features (term document matrices) and labels
# e.g., classifier_tfidf.fit(training_set_tfidf, labels_array)

###################
#                 #
# YOUR CODE HERE  #
#                 #
###################


###########
# testing #
###########

print "creating a graph-of-words for each testing document \n"

# hint: use the terms_to_graph function (found in library.py) with a window of size 3
# loop over all testing documents (list of lists of terms) and store the graphs in a list called all_graphs_test

###################
#                 #
# YOUR CODE HERE  #
#                 #
###################

# sanity checks (should return True)
print len(terms_by_doc_test)==len(all_graphs_test)
print len(set(terms_by_doc_test[0]))==len(all_graphs_test[0].vs)

print "computing vector representations of each testing document"
# each testing document is represented in the training space only

features_degree_test = []
features_w_degree_test = []
features_closeness_test = []
features_w_closeness_test = []
features_tfidf_test = []

counter = 0

for i in xrange(len(all_graphs_test)):
    
    graph = all_graphs_test[i]
	# retain only the terms originally present in the training test
    terms_in_doc = [term for term in terms_by_doc_test[i] if term in all_unique_terms]
    doc_len = len(terms_in_doc)
    
    # returns node (1) name, (2) degree, (3) weighted degree, (4) closeness, (5) weighted closeness
    metrics = compute_node_centrality(graph)
    
    feature_row_degree_test = [0]*len_all
    feature_row_w_degree_test = [0]*len_all
    feature_row_closeness_test = [0]*len_all
    feature_row_w_closeness_test = [0]*len_all
    feature_row_tfidf_test = [0]*len_all
    
    for term in list(set(terms_in_doc)):
        index = all_unique_terms.index(term)
        idf_term = idf[term]
        denominator = (1-b+(b*(float(doc_len)/avg_len)))
        metrics_term = [tuple[1:5] for tuple in metrics if tuple[0]==term][0]
        
        # store TW-IDF values
        feature_row_degree_test[index] = (float(metrics_term[0])/denominator) * idf_term
        feature_row_w_degree_test[index] = (float(metrics_term[1])/denominator) * idf_term
        feature_row_closeness_test[index] = (float(metrics_term[2])/denominator) * idf_term
        feature_row_w_closeness_test[index] = (float(metrics_term[3])/denominator) * idf_term
        
        # number of occurences of word in document
        tf = terms_in_doc.count(term)
        
        # store TF-IDF value
        feature_row_tfidf_test[index] = ((1+math.log1p(1+math.log1p(tf)))/(1-0.2+(0.2*(float(doc_len)/avg_len)))) * idf_term
    
    features_degree_test.append(feature_row_degree_test)
    features_w_degree_test.append(feature_row_w_degree_test)
    features_closeness_test.append(feature_row_closeness_test)
    features_w_closeness_test.append(feature_row_w_closeness_test)
    features_tfidf_test.append(feature_row_tfidf_test)

    counter += 1
    if counter % 100 == 0:
        print counter, "documents have been processed"


# convert list of lists into array
# documents as rows, unique words as columns (i.e., document-term matrix)

testing_set_degree = numpy.array(features_degree_test)
testing_set_w_degree = numpy.array(features_w_degree_test)
testing_set_closeness = numpy.array(features_closeness_test)
testing_set_w_closeness = numpy.array(features_w_closeness_test)
testing_set_tfidf = numpy.array(features_tfidf_test)

# convert truth into integers then into column array
truth = list(truth)

truth_int = [0] * len(truth)
for j in range(len(unique_truth)):
    index_temp = [i for i in range(len(truth)) if truth[i]==unique_truth[j]]
    for element in index_temp:
        truth_int[element] = j

# check that coding went smoothly
zip(truth_int,truth)[:20]

truth_array = numpy.array(truth_int)

print "issuing predictions"

# hint: use the .predict() attribute of the classifier. You need to pass only the features of new observations
# e.g., predictions_tfidf = classifier_tfidf.predict(testing_set_tfidf)
# type(predictions_tfidf)
# predictions_tfidf.shape

###################
#                 #
# YOUR CODE HERE  #
#                 #
###################

print "computing accuracy"

# hint: use the metrics.accuracy_score of sklearn. You need to pass the true and predicted labels 
# as first and second argument respectively
# e.g., print "accuracy TF-IDF:", metrics.accuracy_score(truth_array,predictions_tfidf)

###################
#                 #
# YOUR CODE HERE  #
#                 #
###################