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

     
def parseBusiness():
    """takes in the business_id dictionary and the review dataset and return a 1-d array of business features + reviews.
        This would be modified in the future to generate ML sklearn ready excel files  
    """
    businessesArray = [] #initialize 
    count = 1
    with open('../yelp_dataset/yelp_academic_dataset_business.json') as f:
        for line in f:
            # 1. load in the line (each line is a json object)
            Jline = json.loads(line)
            J_bus_id = Jline["business_id"] # extract business id from line, making it the modal id

            # 2. extract other features from each review object 
            J_name = Jline["name"]
            J_nei = Jline["neighborhood"]
            J_add = Jline["address"]
            J_city = Jline["city"]
            J_state = Jline["state"]
            J_pos = Jline["postal_code"]
            J_lat = Jline["latitude"]
            J_long = Jline["longitude"]
            J_stars = Jline["stars"]
            J_rev = Jline["review_count"]
            J_is_open = Jline["is_open"]
            J_att = Jline["attributes"]
            J_cat = Jline["categories"]
            J_hours = Jline["hours"]

            # 3. put these features in an array, representing a row 
            row = [J_bus_id, J_name, J_nei, J_add, J_city, J_state, J_pos, J_lat, J_long, J_stars, J_rev, J_is_open, J_att, J_cat, J_hours]

            # 4. populating state_breakdown array
            businessesArray.append(row) 
    return businessesArray


def parseReview():
    """takes in the business_id dictionary and the review dataset and return a 1-d array of business features + reviews.
        This would be modified in the future to generate ML sklearn ready excel files  
    """
    reviewArray = [] #initialize 
    with open('../yelp_dataset/yelp_academic_dataset_review.json') as f:
        for line in f:
            # 1. load in the line (each line is a json object)
            Jline = json.loads(line)
            J_rev_id = Jline["review_id"] # extract business id from line, making it the modal id
            J_bus_id = Jline["business_id"]
            J_user_id = Jline["user_id"]

            # 2. extract other features from each review object 
            J_stars = Jline["stars"]
            J_date = Jline["date"]
            J_text = Jline["text"]
            J_useful = Jline["useful"]
            J_funny = Jline["funny"]
            J_cool = Jline["cool"]

            # 3. put these features in an array, representing a row 
            row = [J_rev_id, J_bus_id, J_user_id, J_stars, J_date, J_text, J_useful, J_funny, J_cool]

            # 4. populating state_breakdown array
            reviewArray.append(row) 
    return reviewArray   


def main():
    #review_csv = parseReview()
    business_csv = parseBusiness()
    #write_to_csv(review_csv, "reviews_full.csv")
    write_to_csv(business_csv, "business_full.csv")


# if __name__ == "__main__":
#     main()