import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

#             inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
#                 if inBody:
                    lines.append(line)
#                 elif line == '\n':
#                     inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

data = data.append(dataFrameFromDirectory('C:/Users/Admin/Desktop/plmail/enron1/spam', 'spam'))
data = data.append(dataFrameFromDirectory('C:/Users/Admin/Desktop/plmail/enron1/ham', 'ham'))
#print(data)





#xldl
import re
import nltk
import gensim 
import lxml
import xml.etree.ElementTree
from nltk.corpus import stopwords
# nltk.download('stopswords')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
# STOPWORDS = set(stopwords.words('english'))
STOPWORDS=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", 
           "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
data = data.reset_index(drop=True)
def clean_text(text):
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub(' ', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', ' ')
    text = re.sub(r'\W+', ' ', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text
def remove_tags(text):
    return ''.join(xml.etree.ElementTree.fromstring(text).itertext())
import re

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    cleantext = gensim.utils.simple_preprocess(cleantext) # xoa cac ki tu dac biet 
    return cleantext
data.message=data.message.astype(str)

data['message'] = data['message'].str.replace(',',' ')
# data['message'] = data['message'].apply(cleanhtml)# chuyển str thành list mất
data['message'] = data['message'].apply(clean_text)
# data['message'] = data['message'].apply(remove_tags)
data['message'] = data['message'].str.replace('\d+', ' ')

from nltk.tokenize import word_tokenize
data['message']= [word_tokenize(entry) for entry in data['message']]




# chia train test 
from sklearn import model_selection, naive_bayes, svm
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(data['message'],data['class'],test_size=0.2)
Test_X.to_pickle('Test_X') 
Train_X.to_pickle('Train_X')
Test_Y.to_pickle('Test_Y')
Test_X.to_pickle('Test_X')


# giai doan hoc
demspam=0
for i in Train_Y:
    if (i=='spam'):
        demspam+=1

Pspam=float(demspam)/len(Train_Y)
Pham=1-Pspam
Pspam

sotuspam=0
sotuham=0
from collections import defaultdict
count_words = defaultdict(lambda : [0, 0])
for ts, is_spam in zip(Train_X, Train_Y):
    for words in ts:
        count_words[words][0 if is_spam=='spam' else 1] += 1
        if is_spam=='spam':
            sotuspam+=1
        else:
            sotuham+=1

word_prob=[(w,(spam + 1) / (len(count_words) + sotuspam), (non_spams + 1) / (len(count_words)  + sotuham)) for w, (spam, non_spams) in count_words.items() ]





# giai doan test
import math
# message=Train_X[1032]
def spam_probability(word_probability, message):
    log_prob_spam = log_prob_ham = 0.0
    for word, spam_prob, notspam_prb in word_probability:
        #if the word is in the message update the counters accordingly
        if word in message:
#             print(word)
            log_prob_spam += math.log(spam_prob)
            log_prob_ham += math.log(notspam_prb)
    log_prob_spam += math.log(Pspam)
    log_prob_ham += math.log(Pham)
    if (log_prob_spam>log_prob_ham): 
        return 'spam'
    else:
        return 'ham'

predictions = [ (spam_probability(word_prob,message),is_spam) for message, is_spam in zip(Test_X, Test_Y)]

acc=[(1 if (doan==that) else 0) for doan, that in predictions ]

accuracy=numpy.mean(acc)
print (accuracy)