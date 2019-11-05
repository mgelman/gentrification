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
    file = 'test.json'
    #with open('yelp_dataset/yelp_academic_dataset_business.json') as f:=
    with open(file) as f:
        for line in f:
            # 1. load in the line (each line is a json object)
            Jline = json.loads(line)
            J_bus_id = Jline["business_id"] # extract business id from line, making it the modal id
            
            # 2. extract other features from each business object
            J_name = Jline["name"]
            J_state = Jline["state"]
            J_review_count = Jline["review_count"]
            J_city = Jline["city"]
            J_stars = Jline["stars"]
            J_zipcode = Jline["postal_code"]
            #J_first_category = Jline["categories"][0] ##need to parse list of strings

            # 3. put these features in an array
            feature_array = [J_zipcode, J_name ,J_city, J_state, J_review_count, J_stars]
            
            # 4. populating state_breakdown array
            id_breakdown[J_bus_id] = feature_array
    
    return id_breakdown

     
def parseReviews(id_breakdown):
    """takes in the business_id dictionary and the review dataset and return a 1-d array of business features + reviews.
        This would be modified in the future to generate ML sklearn ready excel files  
    """
    reviews_business_merge = [] #initialize 
    reviews_business_merge.append(["business_id", "stars", "review_text", "time", "postal_code", "business_name", "state", "review_count", "city", "stars"])
    file = 'testReviews.json'
    #with open('yelp_dataset/yelp_academic_dataset_review.json') as f:
    with open(file) as f:
        for line in f:
            # 1. load in the line (each line is a json object)
            Jline = json.loads(line)
            J_bus_id = Jline["business_id"] # extract business id from line, making it the modal id
            
            # 2. extract other features from each review object 
            J_review_stars = Jline["stars"]
            J_review_text = Jline["text"]
            J_review_time = Jline["date"][:-3] #only getting the year and date 

            # 3. put these features in an array
            feature_array = [J_review_stars, J_review_text]

            # 3.5 adding the variables row
            variable_array = ["business_id", ]

            # 4. populating state_breakdown array
            reviews_business_merge.append([J_bus_id]+ feature_array+id_breakdown[J_bus_id]) 
    return reviews_business_merge
            



def main():
    x=parse()
    write_to_csv(parseReviews(x), "reviews_new.csv")

LA_zip = [90895,91001,91006,91007,
91011,
91010,
91016,
91020,
91017,
93510,
91023,
91024,
91030,
91040,
91043,
91042,
91101,
91103,
91105,
93534,
91104,
93532,
91107,
93536,
91106,
93535,
91108,
93543,
93544,
91123,
93551,
93550,
93553,
93552,
91182,
93563,
93590,
91189,
91202,
91201,
93591,
91204,
91203,
91206,
91205,
91208,
91207,
91210,
91214,
91302,
91301,
91304,
91303,
91306,
91307,
91310,
92397,
91311,
91316,
91321,
91325,
91324,
91326,
91331,
91330,
91335,
91340,
91343,
91342,
91345,
91344,
91350,
91346,
91352,
91351,
91354,
91356,
91355,
91357,
91361,
91364,
91367,
91365,
91381,
91383,
91384,
91387,
91390,
91402,
91401,
91403,
91406,
91405,
91411,
91423,
91436,
91495,
91501,
91502,
91505,
91504,
91506,
91602,
91601,
91604,
91606,
91605,
91608,
91607,
91614,
91706,
91702,
91711,
91722,
91724,
91723,
91732,
91731,
91733,
91735,
91740,
91741,
91745,
91744,
91747,
91746,
91748,
90002,
91750,
90001,
91755,
90004,
91754,
90003,
90006,
90005,
90008,
91759,
90007,
90010,
90012,
91765,
90011,
90014,
91767,
91766,
90013,
90016,
91768,
90015,
90018,
90017,
91770,
91773,
90020,
91772,
90019,
91776,
90022,
90021,
91775,
91780,
90024,
91778,
90023,
90026,
90025,
90028,
90027,
91790,
91789,
90029,
91792,
90032,
91791,
90031,
90034,
91793,
90033,
90036,
90035,
90038,
91801,
90037,
90040,
91803,
90039,
90042,
90041,
90044,
90043,
90046,
90045,
90048,
90047,
90049,
90052,
90056,
90058,
90057,
90060,
90059,
90062,
90061,
90064,
90063,
90066,
90065,
90068,
90067,
90069,
90071,
90074,
90077,
91008,
90084,
90089,
90095,
90094,
90096,
90201,
90189,
90211,
90210,
90212,
90221,
90220,
90222,
90230,
90232,
90241,
90240,
90245,
90242,
90248,
90247,
90250,
90249,
90254,
90260,
90255,
90262,
90264,
90263,
90266,
90265,
90270,
90274,
90272,
90277,
90275,
90280,
90278,
90291,
90290,
90293,
90292,
90295,
90301,
90296,
90303,
90302,
90305,
90304,
90402,
90401,
90404,
90403,
90406,
93243,
90405,
90501,
90503,
90502,
90505,
90504,
90508,
90601,
90603,
90602,
90605,
90604,
90606,
90631,
90639,
90638,
90650,
90640,
90660,
90670,
90702,
90701,
90704,
90703,
90706,
90710,
90713,
90712,
90715,
90717,
90716,
90731,
90723,
90733,
90732,
90745,
90744,
90747,
90746,
90755,
90803,
90802,
90805,
90804,
90807,
90806,
90808,
90813,
90810,
90815,
90814,
90840]

# if __name__ == "__main__":
#     main()