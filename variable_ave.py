import requests
import string
import json
from collections import *

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

def write_to_csv( list_of_rows, filename ):
    """ write_to_csv shows how to write that format from a list of rows...
#  + try   write_to_csv( [['a', 1 ], ['b', 2]], "smallfile.csv" )
    """
    try:
        #fname = filename + "_bus.csv"
        csvfile = open( filename, "w", newline='' )
        filewriter = csv.writer( csvfile, delimiter=",")
        for row in list_of_rows:
            filewriter.writerow( row )
        csvfile.close()

    except:
        print("File", filename, "could not be opened for writing...")

def business_ave():
    """takes in the business_id dictionary and the review dataset and return a 1-d array of business features + reviews.
        This would be modified in the future to generate ML sklearn ready excel files  
    """
    #businessesArray = [] #initialize 
    stars_sum = 0
    review_count_sum = 0
    count = 0
    state_breakdown = {}
    with open('../yelp_dataset/yelp_academic_dataset_business.json') as f:
        for line in f:
            # 1. load in the line (each line is a json object)
            Jline = json.loads(line)

            # 2. sum up numeric values  
            stars_sum += Jline["stars"]
            review_count_sum += Jline["review_count"]

            J_state = Jline["state"]
            if J_state not in state_breakdown: #check if the state_name already exists as a key in the state dict
                state_breakdown[J_state] = [0,0,0] #each state has its own dictionary of businesses
            state_breakdown[J_state][0] += Jline["stars"]#each business has the feature array has its key's value
            state_breakdown[J_state][1] += Jline["review_count"]
            state_breakdown[J_state][2] += 1
            # 3. increase count 
            count += 1 
    
    stars_ave = stars_sum/count
    review_count_ave = review_count_sum/count
    print("There are in total: "+ str(count) +" businesses in the dataset")
    print()
    print("average stars for businesses:" + str(stars_ave))
    print()
    print("average review count for businesses:" + str(review_count_ave))
    return [count, stars_ave, review_count_ave, state_breakdown]


def review_ave():
    """takes in the business_id dictionary and the review dataset and return a 1-d array of business features + reviews.
        This would be modified in the future to generate ML sklearn ready excel files  
    """
    #businessesArray = [] #initialize 
    stars_sum = 0
    count = 0
    state_breakdown = {}
    with open('../yelp_dataset/yelp_academic_dataset_review.json') as f:
        for line in f:
            # 1. load in the line (each line is a json object)
            Jline = json.loads(line)

            # 2. sum up numeric values  
            stars_sum += Jline["stars"]

            # 3. increase count 
            count += 1 
    
    stars_ave = stars_sum/count
    print("There are in total: " +str(count)+ " reviews in the dataset")
    print()
    print("average stars for reviews:" + str(stars_ave))
    return [count,stars_ave]

def dic_to_2darrays(dic):
    """return a 2d array converted from a dictionary 
        Each list within is a row in the csv, in sequential order 
    """
    list_of_rows = []
    for key in dic:
        #list_of_rows += [[key,dic.get(key)]]
        list_of_rows.append([key] + dic.get(key))#[[key,dic.get(key)]]
    return list_of_rows


def main():
    """ top-level function for testing problem 1
    """
    bus_ave = business_ave()
    #write_to_csv(dic_to_2darrays(bus_ave[3]), "state_average.csv")
    