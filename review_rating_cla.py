import string
from collections import *
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#
# readcsv is a starting point - it returns the rows from a standard csv file...
#
def readcsv( csv_file_name ):
    """ readcsv takes as
         + input:  csv_file_name, the name of a csv file
        and returns
         + output: a list of lists, each inner list is one row of the csv
           all data items are strings; empty cells are empty strings
    """
    try:
        csvfile = open( csv_file_name, newline='' )  # open for reading
        csvrows = csv.reader( csvfile )              # creates a csvrows object

        all_rows = []                               # we need to read the csv file
        for row in csvrows:                         # into our own Python data structure
            all_rows.append( row )                  # adds only the word to our list

        del csvrows                                  # acknowledge csvrows is gone!
        csvfile.close()                              # and close the file
        return all_rows                              # return the list of lists

    except FileNotFoundError as e:
        print("File not found: ", e)
        return []


#
# write_to_csv shows how to write that format from a list of rows...
#  + try   write_to_csv( [['a', 1 ], ['b', 2]], "smallfile.csv" )
#
def write_to_csv( list_of_rows, filename ):
    """ write_to_csv shows how to write that format from a list of rows...
#  + try   write_to_csv( [['a', 1 ], ['b', 2]], "smallfile.csv" )
    """
    try:
        csvfile = open( filename, "w", newline='' )
        filewriter = csv.writer( csvfile, delimiter=",")
        for row in list_of_rows:
            filewriter.writerow( row )
        csvfile.close()

    except:
        print("File", filename, "could not be opened for writing...")

# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
file = 'reviews_mid.csv'

if file == 'reviews_mid.csv':
    df = pd.read_csv(file, header=None)
    df.columns = ["review_id",	"business_id", "user_id", "stars", "date", "text", "useful", "funny", "cool"]
if file == 'reviews_1000.csv':
    df = pd.read_csv('reviews_1000.csv', header=0)

df.info()

#let's drop columns with too few values or that won't be meaningful
col = ['stars', 'text']
df = df[col]
df = df[pd.notnull(df['text'])]
df.columns = ['stars', 'text']

#dictionaries for future use
#category_to_id = dict(category_id_df.values)
#id_to_category = dict(category_id_df[['category_id', 'Product']].values)

print("+++ End of pandas +++\n")

#display some data info
fig = plt.figure(figsize=(8,6))
df.groupby('stars').text.count().plot.bar(ylim=0)
#plt.show()

"""
bag of words. Turning text data into managebale forms 
"""

from sklearn.feature_extraction.text import TfidfVectorizer

#standard setting 
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.text).toarray() #fit model
labels = df.stars
features.shape

#now we can use sklearn.feature_selection.chi2 to find the words that are most correlated with the stars  
from sklearn.feature_selection import chi2

N = 10
STARS = [1,2,3,4,5]

for star in sorted(STARS):
  features_chi2 = chi2(features, labels == star)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# 'rating star - {}':".format(star))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))


"""
Time to train our classifier 
"""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#X is our feature, y is our target result which is stars 
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['stars'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train) #vectorizing text data 
tfidf_transformer = TfidfTransformer() 
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train) #classifying transformed text data to target value 


















