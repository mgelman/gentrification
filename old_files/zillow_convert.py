import requests
import string
import json
from collections import *
import csv
import numpy as np
import pandas as pd

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
        csvfile = open( csv_file_name, newline='' , encoding='ISO-8859-1')  # open for reading
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


"""
Managing Data
"""

def filter_housing_category(price):
    return int(price / 25)

def create_new_df(file = "Zip_MedianValuePerSqft_AllHomes.csv"):

    data_frame = pd.read_csv(file, encoding='latin-1')
    columns = data_frame.columns

    for col in columns:
        if col == "2007-01": break
        if col == "RegionID":
            continue
        else:
            data_frame = data_frame.drop(columns=[col])

    return data_frame

def create_new_csv(data_frame):
    
    new_csv = [["zip_code", "time", "housing_price", "price_category"]]
    columns = data_frame.columns
    for index, row in data_frame.head().iterrows():
        for col in columns:
            if col != "RegionID":
                zip_code = row['RegionID']
                time = col
                housing_price = row[time]
                new_csv_row = [zip_code, time, housing_price, filter_housing_category(housing_price)]
                new_csv.append(new_csv_row)
    return new_csv


#
# a main function for lab problem 1 (the multicity distance problem)
#
def main():
    """ top-level function for testing problem 1
    """
    new_csv = create_new_csv(create_new_df())
    write_to_csv(new_csv, "zillow.csv")



# if __name__ == "__main__":
#     main()

