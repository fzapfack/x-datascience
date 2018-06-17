# -*- coding: utf-8 -*-

import math
from sklearn.feature_extraction.text import TfidfVectorizer

documents = ["Euler is the father of graph theory",
             "Graph theory studies the properties of graphs",
             "Bioinformatics studies the application of efficient algorithms in biological problems",
             "DNA sequences are very complex structures",
             "Genes are parts of a DNA sequence",
             "Run to the hills, run for your lives",
             "The lonenliness of the long distance runner",
             "Heaven can wait til another day",
             "Road runner and coyote is my favorite cartoon",
             "Heaven can can Heaven can"] # the last document is our query

def my_cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
tfidf_matrix = tfidf_matrix.toarray()

# define that the query is the last document in the collection
query_vector = tfidf_matrix[9,:]
print query_vector
print

print "Similarity among the query and the documents: "
for x in range(0,9):
    print my_cosine_similarity(query_vector, tfidf_matrix[x,:])

