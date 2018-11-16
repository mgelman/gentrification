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
        csvfile = open( filename, "w", newline='' )
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

def parse():
    """takes in the business dataset and return a 1-d dictionary of all business 
    """
    id_breakdown = {} #initialize return dictionary. Making it a dictionary because we need to use it to map review to business using business id
    with open('yelp_dataset/yelp_academic_dataset_business.json') as f:
        for line in f:
            # 1. load in the line (each line is a json object)
            Jline = json.loads(line)
            J_bus_id = Jline["business_id"] # extract business id from line, making it the modal id
            
            # 2. extract other features from each business object
            J_name = Jline["name"]
            J_state = Jline["state"]
            J_review_count = Jline["review_count"]
            J_neighborhood = Jline["neighborhood"]
            J_city = Jline["city"]
            J_stars = Jline["stars"]
            #J_first_category = Jline["categories"][0] ##need to parse list of strings

            # 3. put these features in an array
            feature_array = [J_name, J_neighborhood, J_city, J_state, J_review_count, J_stars]
            
            # 4. populating state_breakdown array
            id_breakdown[J_bus_id] = feature_array
    
    return id_breakdown

     
def parseReviews(id_breakdown):
    """takes in the business_id dictionary and the review dataset and return a 1-d array of business features + reviews.
        This would be modified in the future to generate ML sklearn ready excel files  
    """
    reviewsArray = [] #initialize 
    with open('yelp_dataset/yelp_academic_dataset_review.json') as f:
        for line in f:
            # 1. load in the line (each line is a json object)
            Jline = json.loads(line)
            J_bus_id = Jline["business_id"] # extract business id from line, making it the modal id
            
            # 2. extract other features from each review object 
            J_stars = Jline["stars"]
            J_text = Jline["text"]

            # 3. put these features in an array
            feature_array = [J_stars, J_text]

            # 4. populating state_breakdown array
            reviewsArray.append([J_bus_id]+id_breakdown[J_bus_id] + feature_array) 
    return reviewsArray
            

def main():
    x=parse()
    write_to_csv(parseReviews(x), "reviews_new.csv")


# if __name__ == "__main__":
#     main()