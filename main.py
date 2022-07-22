import json
import shutil
import datetime
import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request, Response
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import os
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return {"message": "Hello World"}


# @app.get("/hello/{name}")
# async def say_hello(name: str):
#     return {"message": f"Hello {name}"}


@app.post("/addcsv", tags=["datas"], summary="Add datas", description="Add / Insert data")
async def def_post_add_data(request: Request, file: UploadFile = File(...), project_name: str = Form(...)):
    return await add_data(project_name, file, request)


async def add_data(projectName: str, file: UploadFile, request):
    ts = datetime.datetime.now().strftime("%m%d_%H%M%S")
    path = Path('D:/aryabhatta/training' + '/Regression/' + projectName + '/data/' + ts + '_' + file.filename)
    await save_upload_file(file, path)
    return {"filename": file.filename}

@app.post("/uploadfile/{ts}")
async def create_upload_file(projectName: str, file: UploadFile, request,ts:str):
    #ts = datetime.datetime.now().strftime("%m%d_%H%M%S")
    path = Path('D:/aryabhatta/training' + '/Regression/' + projectName + '/data/' + ts )
    await save_upload_file(file, path)
    df = pd.read_csv(path)
    return df.to_json(orient = "records",force_ascii = True)

@app.post("/convert/")
async def csvtojson(csvFilePath: UploadFile, jsonFilePath: UploadFile):
    return await convert(csvFilePath, jsonFilePath)

@app.post("/index/")
async def index(first:int,last:int):
    return 0;

@app.get("/slice/{firstrow}/lastrow/{lastrow}/name/{name}")
async def slice(firstrow:str,lastrow:str,name:str):
    df = pd.read_csv('D:/aryabhatta/training' + '/Regression/' + 'sahith' + '/data/'+name)
    #lastrow=df.shape[0]
    #print(df[firstrow:lastrow])
    firstrow=int(firstrow)
    lastrow=int(lastrow)
    return df[firstrow:lastrow].to_json(orient = "records",force_ascii = True)

@app.get("/stats/{column}/types/{types}/{name}")
async def stats(column:str,types:str,name:str):
    df = pd.read_csv('D:/aryabhatta/training' + '/Regression/' + 'sahith' + '/data/'+name)
    #X = list(df.loc[:, column])
    #print(X)
    #plt.xlabel("Years")
    if types == 'mean':
        res = df[column].mean()
    elif types == 'mode':
        res = df[column].mode()
    elif types == 'median':
        res = df[column].median()
    df[column] = df[column].fillna(res)
    #print(df[fill])
    return df[column].to_json(orient ="records",force_ascii = True)


@app.get("/train_test/{name}")
async def train_test(name:str):
    df = pd.read_csv('D:/aryabhatta/training' + '/Regression/' + 'sahith' + '/data/'+name)
    # col=[x,y]
    # df = df.loc[:, col]
    # features = [x]
    # X = df.loc[:, features]
    # y = df.loc[:, [x]]
    train_data,test_data = train_test_split(df, random_state=1, train_size=.75)
    x=train_data.to_json(orient = "records",force_ascii = True)
    y = test_data.to_json(orient="records", force_ascii=True)
    arr=[x,y]
    return json.dumps(arr)

def convert(csvFilePath, jsonFilePath):
    df = pd.read_csv(csvFilePath,sep = ",", header = "infer", index_col = False)
    df.to_json(jsonFilePath,orient = "records",force_ascii = True)

@app.get("/lastrow/{name}")
async def lastrow(name:str):
    df = pd.read_csv('D:/aryabhatta/training' + '/Regression/' + 'sahith' + '/data/'+name)
    return df.shape[0]

@app.get("/columns/{name}")
async def columns(name:str):
    df = pd.read_csv('D:/aryabhatta/training' + '/Regression/' + 'sahith' + '/data/'+name)
    return list(df.columns.values)

def cols(file):
    df = pd.read_csv(file)
    return list(df.columns.values)

def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getmtime)

@app.get("/bar/{x}/y/{y}/{name}", responses={200: {"content" : {"image/jpg" : {"example" : "picture of a vector image."}}}},response_class=FileResponse)
async def bar(x:str,y:str,name:str):
    barchart(x,y,name)
    return FileResponse("output.jpg", media_type="image/jpg")


def barchart(x,y,name):
    data = pd.read_csv('D:/aryabhatta/training' + '/Regression/' + 'sahith' + '/data/'+name)
    df = pd.DataFrame(data)
    plt.clf()
    X = list(df.loc[:, x])
    plt.xlabel(x)
    Y = list(df.loc[:, y])
    plt.ylabel(y)

    plt.bar(X, Y, color='g')
    plt.savefig("output.jpg")
    #plt.show()

@app.get("/scatter/{x}/y/{y}/{name}", responses={200: {"content" : {"image/jpg": {"example" : "picture of a vector image."}}}},response_class=FileResponse)
async def scatter(x:str,y:str,name:str):
    scatterplot(x,y,name)
    return FileResponse("scatter.jpg", media_type="image/jpg")

def scatterplot(a,b,name):
    data = pd.read_csv('D:/aryabhatta/training' + '/Regression/' + 'sahith' + '/data/'+name)
    df = pd.DataFrame(data)
    plt.clf()
    A = df.loc[:,a]
    plt.xlabel(a)

    B = df.loc[:,b]
    plt.ylabel(b)

    plt.scatter(A, B, color='g')
    # plt.show()
    plt.savefig("scatter.jpg")

@app.get("/hist/{x}/{name}", responses={200: {"content" : {"image/jpg": {"example" : "picture of a vector image."}}}},response_class=FileResponse)
async def hist(x:str,name:str):
    histogram(x,name)
    return FileResponse("hist.jpg",media_type="image/jpg")

def histogram(x,name):
    data = pd.read_csv('D:/aryabhatta/training' + '/Regression/' + 'sahith' + '/data/'+name)
    df = pd.DataFrame(data)
    plt.clf()
    X = (df.loc[:, x])
    plt.xlabel(x+" (binned)")
    plt.ylabel("count of "+x)
    plt.hist(X, bins=21)
    plt.savefig("hist.jpg")
    # plt.show()

@app.get("/time/{x}/y/{y}/{name}", responses={200: {"content" : {"image/jpg" : {"example" : "picture of a vector image."}}}},response_class=FileResponse)
async def time(x:str,y:str,name:str):
    timeseries(x,y,name)
    return FileResponse("time.jpg",media_type="image/jpg")

def timeseries(c,d,name):
    data = pd.read_csv('D:/aryabhatta/training' + '/Regression/' + 'sahith' + '/data/'+name)
    df = pd.DataFrame(data)
    plt.clf()
    df=df.set_index(c)
    plt.xlabel("xcol")
    D = df.loc[:, d]
    plt.ylabel(d)
    plt.plot(D,marker="o")
    plt.tight_layout()
    plt.savefig("time.jpg")
    # plt.show()

async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    try:
        upload_file.file.seek(0)
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8888)
