import os
import re
import nltk
import string
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
# sys.path.append(os.path.abspath("./Code"))
from library import *

nltk.download('punkt') # for tokenization
nltk.download('maxent_treebank_pos_tagger') # for POS tagging
nltk.download('stopwords')

stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()
punct = string.punctuation.replace("-", "")

##################################
# read and pre-process abstracts #
##################################
print "reading and pre-processing abstracts\n"

# path = os.getcwd() + "\\Hulth2003testing\\abstracts"
# os.chdir(path)
nltk.download('maxent_treebank_pos_tagger') # for POS tagging

root_dir = os.getcwd()
os.chdir("../Data")
path = os.getcwd() + "/Hulth2003testing/abstracts"  
os.chdir(path)

abstracts = []
counter = 0

for filename in sorted(os.listdir(path)):
  f = open(filename, 'r')
  content = f.read()
  # remove formatting
  content =  re.sub("\s+", " ", content)
  # convert to lower case
  content = content.lower()
  # remove punctuation (preserving intra-word dashes)
  content = "".join(letter for letter in content if letter not in punct)
  # remove dashes attached to words but that are not intra-word
  content = re.sub("[^[:alnum:]['-]", " ", content)
  content = re.sub("[^[:alnum:][-']", " ", content)
  # remove extra white space
  content = re.sub(" +"," ", content)
  # remove leading and trailing white space
  content = content.strip()
  # tokenize
  tokens = content.split(" ")
  # remove stopwords
  tokens_keep = [token for token in tokens if token not in stpwds]
  # POS-tag 
  tagged_tokens = nltk.pos_tag(tokens_keep)
  # keep only nouns and adjectives    
  tokens_keep = [pair[0] for pair in tagged_tokens if (pair[1] in ["NN","NNS","NNP","NNPS","JJ","JJS","JJR"])]
  # apply Porter stemmer
  tokens_keep = [stemmer.stem(token) for token in tokens_keep]
  # store list of tokens
  abstracts.append(tokens_keep)

  counter += 1
  if counter % 100 == 0:
  	print counter, 'abstracts have been processed'
 
#################################
# read and pre-process keywords #
#################################
print "reading and pre-processing humman assigned keywords\n"   
   
path = os.getcwd() + "\\Hulth2003testing\\uncontr"
os.chdir(path)

golden_keywords = []

for filename in sorted(os.listdir(path)):
   f = open(filename, 'r')
   content = f.read()
   # remove formatting
   content =  re.sub("\s+", " ", content)
   # convert to lower case
   content = content.lower()
   # turn string into list of keywords, preserving intra-word dashes 
   # but breaking n-grams into unigrams to easily compute precision and recall
   content = content.split(";")
   content = [keyword.split(" ") for keyword in content]
   # flatten list
   content = [keyword for sublist in content for keyword in sublist]
   # remove empty elements
   content = [keyword for keyword in content if len(keyword)>0]
   # remove stopwords (rare but can happen due to n-gram breaking)
   content = [keyword for keyword in content if keyword not in stpwds]
   # apply Porter's stemmer
   content = [stemmer.stem(keyword) for keyword in content]
   # remove leading and trailing whitespace
   content = [keyword.strip() for keyword in content]
   # remove duplicates (can happen due to n-gram breaking)
   content = list(set(content))
   # save final list
   golden_keywords.append(content)
   
# ################################
# # main core keyword extraction #
# ################################
   
print "creating a gow for each abstract \n"
all_graphs = []
window = 3
for abstract in abstracts:
    all_graphs.append(terms_to_graph(abstract, window))

print "extracting main cores from each graph \n"
mcs_weighted = []
mcs_unweighted = []
counter = 0

# hint: use the core_dec function (found in library.py)
# this function returns the main core of the graph (subgraph, igraph obect):
# e.g., core_dec(graph, weighted = True)["main_core"]
# the names of the vertices of this subgraph can be obtained as a list via:
# core_dec(graph, weighted = True)["main_core"].vs["name"]

# loop over "all_graphs" and store the main cores in the "mcs_weighted" and "mcs_unweighted" lists
# you will end up with two lists of lists

###################
#                 #
# YOUR CODE HERE  #
#                 #
###################

# on average, the weighted main cores are much smaller
# this corroborates the fact that weighted is better for precision but less for recall
print "average size of weighted main cores:", sum([len(main_core) for main_core in mcs_weighted])/len([len(main_core) for main_core in mcs_weighted])
print "average size of unweighted main cores:",sum([len(main_core) for main_core in mcs_unweighted])/len([len(main_core) for main_core in mcs_unweighted])

# #############
# # baselines #
# #############

# TF_IDF
print "baseline tf-idf \n"

corpus = [str(" ".join(abstract)) for abstract in abstracts]

# (optional) find out the number of unique words in abstracts
temp = [list(set(abstract)) for abstract in abstracts]
temptemp = [item for sublist in temp for item in sublist]
print len(set(temptemp))

vectorizer = TfidfVectorizer()

# transforms the input data into bow feature vectors with tf-idf weights
features = vectorizer.fit_transform(corpus)

colnames = vectorizer.get_feature_names()

features_dense_list = features.todense().tolist()

tf_idf = []

for element in features_dense_list:
    
    # bow feature vector as list of tuples
    row = zip(colnames,element)
    
    # keep only non zero values
    # (the final length of the list should match exactly the number of unique terms in the document)
    nonzero = [tuple for tuple in row if tuple[1]!=0]
    
    # rank in decreasing order
    nonzero = sorted(nonzero, key=operator.itemgetter(1), reverse=True)
    
    # retain top 33% words as keywords
    numb_to_retain = int(round(len(nonzero)/3))
    
    keywords = [tuple[0] for tuple in nonzero[:numb_to_retain]]
    
    tf_idf.append(keywords)

# PageRank
print "baseline PageRank"

PageRank = []
counter = 0

for graph in all_graphs:
    
    # compute PageRank scores
	
	# hint: use .pagerank attribute of igraph objects
	# store result as "pr_scores"
	
	###################
	#                 #
	# YOUR CODE HERE  #
	#                 #
	###################
	
	# preserve name one-to-one mapping before sorting
	pr_scores = zip(graph.vs["name"],pr_scores)
	
    # rank in decreasing order
    pr_scores = sorted(pr_scores, key=operator.itemgetter(1), reverse=True)
    
    # retain top 33% words as keywords
    numb_to_retain = int(round(len(pr_scores)/3))
    
    keywords = [tuple[0] for tuple in pr_scores[:numb_to_retain]]
    
    PageRank.append(keywords)    
    
    counter+=1
    if counter % 100 == 0:
        print counter, "graphs have been processed"

# ##############
# # evaluation #
# ##############
accuracy_mcs_weighted = []
accuracy_mcs_unweighted = []
accuracy_PageRank = []
accuracy_tf_idf = []

for i in xrange(len(golden_keywords)):
    truth = golden_keywords[i]
    
    accuracy_mcs_weighted.append(accuracy_metrics(candidate = mcs_weighted[i], truth = truth))
    accuracy_mcs_unweighted.append(accuracy_metrics(candidate = mcs_unweighted[i], truth = truth))
    accuracy_PageRank.append(accuracy_metrics(candidate = PageRank[i], truth = truth))
    accuracy_tf_idf.append(accuracy_metrics(candidate = tf_idf[i], truth = truth))

# macro-averaged results (collection level)
print "macro-averaged precision:", sum([tuple[0] for tuple in accuracy_mcs_weighted])/len(golden_keywords)
print "macro-averaged recall:", sum([tuple[1] for tuple in accuracy_mcs_weighted])/len(golden_keywords)
print "macro-averaged F-1 score:", sum([tuple[2] for tuple in accuracy_mcs_weighted])/len(golden_keywords)

sum([tuple[0] for tuple in accuracy_mcs_unweighted])/len(golden_keywords)
sum([tuple[1] for tuple in accuracy_mcs_unweighted])/len(golden_keywords)
sum([tuple[2] for tuple in accuracy_mcs_unweighted])/len(golden_keywords)

sum([tuple[0] for tuple in accuracy_PageRank])/len(golden_keywords)
sum([tuple[1] for tuple in accuracy_PageRank])/len(golden_keywords)
sum([tuple[2] for tuple in accuracy_PageRank])/len(golden_keywords)

sum([tuple[0] for tuple in accuracy_tf_idf])/len(golden_keywords)
sum([tuple[1] for tuple in accuracy_tf_idf])/len(golden_keywords)
sum([tuple[2] for tuple in accuracy_tf_idf])/len(golden_keywords)