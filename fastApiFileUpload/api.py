import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

@app.post("/convert/")
async def csvtojson(csvFilePath: UploadFile, jsonFilePath:str):
    return await convert(csvFilePath, jsonFilePath)


def convert(csvFilePath, jsonFilePath):
    df = pd.read_csv(csvFilePath,sep = ",", header = "infer", index_col = False)
    result = df.to_json(jsonFilePath,orient = "records",force_ascii = True)
    return result

def cols(file):
    df = pd.read_csv(file)
    return list(df.columns.values)

def newfile():
    list_of_files = glob.glob('/path/to/folder/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return(latest_file)

def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getmtime)

def bar(x,y):
    data = pd.read_csv(newest('D:/aryabhatta/training' + '/Regression/' + 'sahith' + '/data/'))
    df = pd.DataFrame(data)

    X = list(df.loc[:,x])
    plt.xlabel(x)

    Y = list(df.loc[:,y])
    plt.ylabel(y)

    plt.bar(X, Y, color='g')
    plt.show()

