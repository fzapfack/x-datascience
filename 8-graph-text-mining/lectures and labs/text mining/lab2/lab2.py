
# Import python modules
import numpy as np
import csv
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import *
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix


# TODO for later versions of the lab
# Change csv reader - problem with utf-8 on windows


#%% 
# Part 1: Load data from .csv file
############
with open('movie_reviews.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',',quotechar='"')
        
    # Initialize lists for data and class labels
    data =[]
    labels = []
    # For each row of the csv file
    for row in reader:
        # skip missing data
        if row[0] and row[1]:
            #data.append(row[0])
            data.append(row[0].decode('utf-8'))
            y_label = -1 if row[1]=='negative' else 1
            labels.append(y_label)

#%%
# Part 2: data preprocessing
############
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

# For each document in the dataset, do the preprocessing
for doc_id, text in enumerate(data):
    
    # Remove punctuation and lowercase
    punctuation = set(string.punctuation)    
    doc = ''.join([w for w in text.lower() if w not in punctuation])
        
    # Stopword removal
    doc = [w for w in doc.split() if w not in stopwords]  
        
    # Stemming
    stemmer = PorterStemmer()
    doc = [stemmer.stem(w) for w in doc] 
        
    # Covenrt list of words to one string
    doc = ' '.join(w for w in doc)
    data[doc_id] = doc       


#%%
# Part 3: create the TF-IDF matrix
#############
m = TfidfVectorizer()
tfidf_matrix = m.fit_transform(data)
tfidf_matrix = tfidf_matrix.toarray() # convert to numpy array
print "Size of TF-IDF matrix: ", tfidf_matrix.shape    
print "Sparsity of the TF_IDF matrix: ", \
float(np.count_nonzero(tfidf_matrix)) / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]) 


#%%
# Part 4: Classification
#############
# Split the data into random train and test subsets. Here we use 40% of the data for testing
data_train, data_test, labels_train, labels_test = cross_validation.train_test_split(\
        tfidf_matrix, labels, test_size=0.4, random_state=42)

## Model learning and prediction
## TODO: test different learning algorithms
#clf = LogisticRegression()
clf = BernoulliNB().fit(data_train,labels_train)
#clf = RandomForestClassifier(n_estimators=100)
y_score = clf.fit(data_train, labels_train).predict_proba(data_test)
labels_predicted = clf.predict(data_test)   
       
# Evaluation of the prediction
print classification_report(labels_test, labels_predicted)
print "The accuracy score is {:.2%}".format(accuracy_score(labels_test, labels_predicted))
print confusion_matrix(labels_test, labels_predicted)
    
# Compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(labels_test, y_score[:, 1])
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc
    
# Plot ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show() 

#%%
# Analyse de l impact d'une reduction de dimension