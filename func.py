import pandas as pd
import os
import csv
import json

def cols(path, file):
    df = pd.read_csv(os.path.join(path, file))
    return tuple(df.columns.values)

# path = input("Enter File Path: ")
# file = input("Enter File Name: ")
# print(cols(path,file))

def make_json(csvFilePath, jsonFilePath):

    data = {}
    df = pd.read_csv(csvFilePath)
    x=str(df.iloc[0,0])
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)

        for rows in csvReader:

            key = rows['Year']
            data[key] = rows

    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))

def convert(csvFilePath, jsonFilePath):
    df = pd.read_csv(csvFilePath,sep = ",", header = "infer", index_col = False)
    df.to_json(jsonFilePath,orient = "records",force_ascii = True)

# csvFilePath = input("Enter csv file path: ")
# jsonFilePath = input("Enter json file path: ")
#
# convert(csvFilePath, jsonFilePath)

def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getmtime)

print(newest('D:/aryabhatta/training' + '/Regression/' + 'sahith' + '/data/'))

