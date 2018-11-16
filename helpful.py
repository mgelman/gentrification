
import requests
import string
import json
from collections import *

"""
Examples to run for problem 1:

Web scraping, the basic command (Thanks, Prof. Medero!)
"""

def google_api(Sources, Dests):
    """ Inputs: Sources is a list of starting places
                Dests is a list of ending places

        This function uses Google's distances API to find the distances
             from all Sources to all Dests. 
        It saves the result to distances.json

        Problem: right now it only works with the FIRST of each list!
    """
    print("Start of google_api")

    url="http://maps.googleapis.com/maps/api/distancematrix/json"

    if len(Sources) < 1 or len(Dests) < 1:
        print("Sources and Dests need to be lists of >= 1 city each!")
        return

    start = Sources
    end = Dests
    my_mode="driving"  # walking, biking

    inputs={"origins":start,"destinations":end,"mode":my_mode}

    result = requests.get(url,params=inputs)
    print(result)
    data = result.json()
    print("data is", data)

    #
    # save this json data to the file named distances.json
    #
    filename_to_save = "distances.json"
    f = open( filename_to_save, "w" )     # opens the file for writing
    string_data = json.dumps( data, indent=2 )  # this writes it to a string
    f.write(string_data)                        # then, writes that string to a file
    f.close()                                   # and closes the file
    print("\nFile", filename_to_save, "written.")
    # no need to return anything, since we're better off reading it from file later...
    return


#
# example of handling json data via Python's json dictionary
#
def json_process():
    """ This function reads the json data from "distances.json"

        and create distance and time chart of the destinations and origins inputed 
    """
    filename_to_read = "distances.json"
    f = open( filename_to_read, "r" )
    string_data = f.read()
    JD = json.loads( string_data )  # JD == "json dictionary"

    #creates variables that represent destinations and origins
    destinations = JD['destination_addresses']
    origins = JD['origin_addresses']

    #creating distance chart:
    print("the following is a distance chart")
    print("\n")
    for i in range(len(origins)):
        print("From ", origins[i])
        each_origin = JD['rows'][i]

        for j in range(len(destinations)):
            to_destination = each_origin['elements'][j]
            distance_as_string = to_destination['distance']['text']
       
            print(". . . to", destinations[j], ":", distance_as_string)
    
    print("\n")
    print("the following is a travel time chart")
    print("\n")

    for i in range(len(origins)):
        print("From ", origins[i])
        each_origin = JD['rows'][i]

        for j in range(len(destinations)):
            to_destination = each_origin['elements'][j]
            distance_as_string = to_destination['duration']['text']
       
            print(". . . to", destinations[j], ":", distance_as_string)

    
    return JD


import csv

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


def dic_to_2darrays(dic):
    """return a 2d array converted from a dictionary 
    """
    list_of_rows = []
    for key in dic:
        list_of_rows += [[key,dic.get(key)]]
    return list_of_rows


#
# a main function for lab problem 1 (the multicity distance problem)
#
def main():
    """ top-level function for testing problem 1
    """

    Dests = ['Seattle,WA','Miami,FL','Boston,MA']  # starts
    Sources = ['Claremont,CA','Seattle,WA','Philadelphia,PA'] # ends
    #join the sources and destinations together 
    Sources = '|'.join(Sources)
    Dests = '|'.join(Dests)
    #print (Sources) 
    #print(Dests)
    if 1:  # do we want to run the API call?
        google_api(Sources, Dests)  # get file
    json_process()



if __name__ == "__main__":
    main()

