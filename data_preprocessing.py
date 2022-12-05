# libraries & methods import
import re  
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from nltk import word_tokenize
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords


# formation of a list with stopwords
f_stopword = open("stopwords.txt")
stopword = f_stopword.read().splitlines()

# preparation of training data
## form a list of file names stored in a folder
folder_path = "/data/train"
file_pattern = r".*\.txt"
ptb = PlaintextCorpusReader(folder_path, file_pattern)
list_file = ptb.fileids()
j = 0
corpus = []
index = ["" for x in range(len(list_file))]
label = np.zeros((len(list_file), 1))
new_string = []
for fn in list_file:
    index[j] = fn
    # get the label
    file_number = fn.split('.')[0].split('_')[2]
    label[j] = int(file_number)
    j = j + 1
    # list of paths to files with training data and opening for reading
    paths_list = [line.strip() for line in open("/data/train" + fn, 'r')]
    for i in range(len(paths_list )):
        # lowercase
        paths_list[i] = paths_list[i].lower()
        # clearing a string of all characters except letters, numbers, and underscores
        paths_list[i] = re.sub(r'\W', ' ', paths_list[i])
        # stripping a string of one or more spaces, or one or more spaces at the end of a string
        paths_list[i] = re.sub(r'(^\s+|\s+$)', '', paths_list[i]) 
        # clearing a string from characters from a list
        paths_list[i] = re.sub(r'\[!@#$%^&*]/g', '', paths_list[i])
        # clearing a string of numbers
        paths_list[i] = re.sub(r'\d', '', paths_list[i])
        # clearing a string of dots at the beginning or end of a string
        paths_list[i] = re.sub(r'(^| ).( |$)', '', paths_list[i]) 
    
    sent_str_ = ""
    for i in paths_list:
        sent_str_ += str(i) + "-"
    sent_str_ = sent_str_[:-1]
    # splitting text into words
    sent_str_ = word_tokenize(sent_str_)
    # removing stopwords
    sent_str_without_sw = [word for word in sent_str_ if not word in stopwords.words()]
    new_string = " ".join(sent_str_without_sw)
    # saving the cleared text to a list
    corpus.append(new_string) 
vectorizer = CountVectorizer()   
transformer = TfidfTransformer() 
# converting a collection of documents to a matrix of TF-IDF features
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
# converting a set of texts into a matrix of tokens in the text
word = vectorizer.get_feature_names() # bag-of-words 
# tf-idf weight
weight = tfidf.toarray()  
labels_list = []
labels_list_numbers = []
for i in index:
    labels_list.append(i.split('_')[0])
for i in labels_list:
    if i == "CURRENT SMOKER":
        labels_list_numbers.append(0)
    elif i == "NON-SMOKER":
        labels_list_numbers.append(1)
    elif i == "PAST SMOKER":
        labels_list_numbers.append(2)
    else:
        labels_list_numbers.append(3) 
df_matrix_labels = pd.DataFrame(weight)
df_matrix_labels["outcome"] = labels_list_numbers 
# saving tf-idf matrix and labels in csv-files
df_matrix_labels.to_csv("tfidf_matrix_label.csv", index=False)
