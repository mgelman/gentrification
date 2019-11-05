import requests
import string
import json
from collections import *
import csv

def write_to_csv( list_of_rows, filename ):
    """ write_to_csv shows how to write that format from a list of rows...
#  + try   write_to_csv( [['a', 1 ], ['b', 2]], "smallfile.csv" )
    """
    try:
        fname = filename + "_bus.csv"
        csvfile = open( fname, "w", newline='' )
        filewriter = csv.writer( csvfile, delimiter=",")
        for row in list_of_rows:
            filewriter.writerow( row )
        csvfile.close()

    except:
        print("File", filename, "could not be opened for writing...")


def dic_to_2darrays(dic):
    """return a 2d array converted from a dictionary 
        Each list within is a row in the csv, in sequential order 
    """
    list_of_rows = []
    for key in dic:
        #list_of_rows += [[key,dic.get(key)]]
        list_of_rows.append([key] + dic.get(key))#[[key,dic.get(key)]]
    return list_of_rows

def convert2array(sb):
    """ take in a 2-d dictionary, such as "state_breakdown", as input and output 
        a new dictionary that has the second level dictionary now converted to a 2-d array
    """
    ready = {}
    for key in sb:
        ready[key] = dic_to_2darrays(sb[key]) # each key is a row in the 2d array
    return ready

def parse():
    """takes in the business dataset and return a 2-d dictionary classifying all businesses by states 
    """
    state_breakdown = {} #initialize return dictionary
    with open('yelp_dataset/yelp_academic_dataset_business.json') as f:
        for line in f:
            # 1. load in the line (each line is a json object)
            Jline = json.loads(line) 
            J_bus_id = Jline["business_id"] # extract business id from line, making it the modal id

            # 2. extract other features 
            J_name = Jline["name"]
            J_state = Jline["state"]
            J_review_count = Jline["review_count"]
            J_neighborhood = Jline["neighborhood"]
            J_city = Jline["city"]
            J_stars = Jline["stars"]
            #J_first_category = Jline["categories"][0] ##need to parse list of strings

            # 3. put these features in an array
            feature_array = [J_name, J_neighborhood, J_city, J_state, J_review_count, J_stars]#J_first_category, J_stars]

            # 4. populating state_breakdown array
            if J_state not in state_breakdown: #check if the state_name already exists as a key in the state dict
                state_breakdown[J_state] = {} #each state has its own dictionary of businesses
            state_breakdown[J_state][J_bus_id] = feature_array #each business has the feature array has its key's value
    
    return state_breakdown

def convert2csv(r):
    """ take in output from convert2array, and feed each key's 2-d array to write_to_csv
    """
    for state in r:
        write_to_csv(r[state], state)    

def main():
    """ top-level function for testing problem 1
    """
    ready = convert2array(parse())
    convert2csv(ready)


# if __name__ == "__main__":
#     main()