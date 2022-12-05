SYSTEM REQUIREMENTS:
Python3= 3.9.15 \\
Keras=2.9.0\\
Gensim =4.1.2\\
sklearn=1.1.3\\
numpy=1.22.3\\
pandas=1.5.1\\
nltk=3.7

Access to download the original dataset can be requested by the website of Harvard Medical School:
URL https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

Google News Vectors need to be downloaded from below source and keep the download under data folder:
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g


GOALS
1. Classify patients into 4 categories(“CURRENT SMOKER, NON-SMOKER, PAST-SMOKER and UNKNOWN) according to smoking status.
2. Build baseline and CNN model
    – Naive Bayes classifier;
    - XGBoost;
    – Gradient boosting;
    – CNN.
3. Compare performance in extracting Smoking status from Patients Clinical notes:


FILE DESCRIPTION
1. data folder  – Initial data divided into test and training
                  Each individual file stores a textual medical report about 
                  a particular patient.There are 40 progress notes under 
2. data_preprocessing.py – contains a python script that allows you to get tfidf_matrix_label.csv
                           file, in which the reports are translated into a language suitable for
                           training the model, namely, the frequency of occurrence of medical terms 
                           in patient reports is calculated.
3. stopwords.txt – contains a list of stop words for their subsequent exclusion from medical reports.
4. BaselineModel_TFIDF.ipynb – python notebook that contains the results of the baseline models.
5. CNN_Word2vec_V1.ipynb  - python notebook that contains the result of CNN model with retrained word2vec.
6. tfidf_matrix_label   - A file in which the reports are translated into a language suitable for
                           training the model, namely, the frequency of occurrence of medical terms 
                           in patient reports is calculated.


INSTRUCTION TO RUN THE CODE
Google News Vectors need to be downloaded from below source and keep the download under data folder:
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g
tfidf_matrix_label.csv has already been generated using data_preprocessing.py. This data can directly be used to train our baseline model.



