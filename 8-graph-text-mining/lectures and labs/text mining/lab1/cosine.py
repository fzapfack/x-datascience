# -*- coding: utf-8 -*-

import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# here we define our document collection which contains 5 documents
# this is an array of strings
documents = ["Euler is the father of graph theory",
             "Graph theory studies the properties of graphs",
             "Bioinformatics studies the application of efficient algorithms in biological problems",
             "DNA sequences are very complex biological structures",
             "Genes are parts of a DNA sequence"]
                          
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# get the unique terms of the collection and display them
terms = tfidf_vectorizer.get_feature_names()
print
print
print "The unique terms of the collection are: "
print terms
print

# print matrix dimensionality
print "The dimensionality of the tfidf matrix is: "
print tfidf_matrix.shape
print

# print matrix contents
print tfidf_matrix.toarray()
print
print

# define the doc-doc similarity matrix based on the cosine distance
print "This is the doc-doc similarity matrix :"
ddsim_matrix = cosine_similarity(tfidf_matrix[:], tfidf_matrix)
print ddsim_matrix
print


# display the first line of the similarity matrix
# these are the similarity values between the first document with the rest of the documents
print "The first row of the doc-doc similarity matrix: "
print ddsim_matrix[:1]
print

cosine_1_2 = 0.42284413
angle_in_radians = math.acos(cosine_1_2)
angle_in_degrees = math.degrees(angle_in_radians)
print "The cosine of the angle between doc1 and doc2 is : \t" + str(cosine_1_2)
print "The angle (in radians) between doc1 and doc2 is  : \t"  + str(angle_in_radians)
print "The angle (in degrees) between doc1 and doc2 is  : \t"  + str(angle_in_degrees)
