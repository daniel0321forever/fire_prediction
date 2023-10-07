'''
*Version: 2.0 Published: 2021/03/09* Source: [NASA POWER](https://power.larc.nasa.gov/)
POWER API Multi-Point Download
This is an overview of the process to request data from multiple data points from the POWER API.
'''

import os, json, random
import datetime
from datetime import timedelta

import requests as re
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import heapq


"""
TODO:
1. process daily data in the init of FPDataset
2. return 2 values, dims in FPDataset
3. read 2 dims in pipeline
4. read 2 values in training, val, testing, pred step
5. read 2 values in foward
6. concat two RNN
"""

random.seed(datetime.datetime.now().timestamp())

def append_dataset(date_time: datetime, loc: tuple, is_fire: bool, params=["T2M", "T2MDEW", "T2MWET", "TS", "WS10M", "WS2M", "WD2M"], community="RE"):
    # make file path
    CSV_PATH = os.path.join("dataset", "dataset_temp.csv")
    initial = not os.path.exists(CSV_PATH)
    if initial:
        f = open(CSV_PATH, 'w')
        f.close()

    # process params
    params_str = ",".join(params)

    # process date (get the last five hours)
    start_datetime = date_time - timedelta(hours=5)
    start_date = start_datetime.__str__().split(" ")[0]
    start_date = f"{start_date.split('-')[0]}{start_date.split('-')[1]}{start_date.split('-')[2]}"
    end_date = date_time.__str__().split(" ")[0]
    end_date = f"{end_date.split('-')[0]}{end_date.split('-')[1]}{end_date.split('-')[2]}"
    
    # get API url
    url = f"https://power.larc.nasa.gov/api/temporal/hourly/point?header=false&parameters={params_str}&community={community}&longitude={loc[0]}&latitude={loc[1]}&start={start_date}&end={end_date}&format=JSON"

    # response
    response = re.get(url=url, verify=True, timeout=30.00)
    content = response.content.decode('utf-8')
    datas = json.loads(content)["properties"]["parameter"] # content dict
    
    append_data = {
        "is_fire": int(is_fire),
        "lng": loc[0],
        "lat": loc[1],
        "date": end_date,
        "time": date_time.__str__().split(" ")[1].split(":")[0]
    }
    try:
        for param in datas.keys():
            for delta_hours in range(5):
                # process datetime index
                date, time = (date_time - timedelta(hours=delta_hours)).__str__().split(" ")
                y, m, d = date.split("-")
                h = time.split(":")[0]
                date_time_str = f"{y}{m}{d}{h}"
                append_data[f"{param}_{delta_hours}h"] = [datas[param][date_time_str]]

        df = pd.DataFrame(append_data)
        df.to_csv(CSV_PATH, mode='a', header=initial, index=False)
        return df
    except KeyError:
        print("Key Error Occurs")
        print(datas)
        return False

def append_fire_index(year=2022):
    # make file path
    CSV_PATH = os.path.join("dataset", "fire_index_temp.csv")

    # grab file
    date = datetime.date(year, 1, 1)

    for m in range(1, 13):
        while date.month == m:
            date_str = date.__str__()
            print(f"today is {date_str}")

            url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/9eb5b659a75b5b17fe1d93884538e1cb/VIIRS_SNPP_NRT/world/1/{date_str}"
            content = re.get(url).content.decode('utf-8')
            rows = content.splitlines()

            if len(rows) <= 1: # if no fire happens
                print(content)
                print("no fire today")
                date = date + timedelta(days=1)
                continue
            
            dict = {}
            keys = rows[0].split(",")
            for i in range(1, len(rows)):
                attrs = rows[i].split(",")
                for j in range(len(keys)):
                    dict.setdefault(keys[j], [])
                    dict[keys[j]].append(attrs[j])
            
            # modify csv
            new_data = pd.DataFrame(dict).loc[:, ["longitude", "latitude", "acq_date", "acq_time"]]
            max_row_num = new_data.shape[0] // 1000
            new_data = new_data.iloc[:max_row_num, :]
            print(new_data)

            initial = not os.path.exists(CSV_PATH)
            if initial:
                f = open(CSV_PATH, 'w')
                f.close()       
            new_data.to_csv(CSV_PATH, mode='a', header=initial, index=False)

            date = date + timedelta(days=1)

def append_no_fire_index(year=2022):
    # make file path
    CSV_PATH = os.path.join("dataset", "no_fire_index_temp.csv")

    date_time = datetime.datetime(year, 1, 1, 0)

    for month in range(1, 13):
        # dict 
        no_fire_index_dict = {
            "longitude": [],
            "latitude": [],
            "acq_date": [],
            "acq_time": [],
        }

        while date_time.month == month:
            date_str = date_time.date().__str__() # 2022-01-01
            i = 0
            while i < 30:
                lng = random.uniform(-180, 180)
                lat = random.uniform(-90, 90)
                hour = random.randint(0, 24)

                print(f"lng: {lng}, lat: {lat}, date: {date_str}, hour: {hour}")
                url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/9eb5b659a75b5b17fe1d93884538e1cb/VIIRS_SNPP_NRT/{(lng-1):.2f},{(lat-1):.2f},{(lng+1):.2f},{(lat+1):.2f}/1/{date_str}"

                content = re.get(url).content.decode("utf-8") # csv
                 
                if len(content.splitlines()) <= 1: # if there is indeed no fire

                    no_fire_index_dict["longitude"].append(lng)
                    no_fire_index_dict["latitude"].append(lat)
                    no_fire_index_dict["acq_date"].append(date_str)
                    no_fire_index_dict["acq_time"].append(hour)
                    i += 1
                else:
                    print("fire happends")

            date_time = date_time + timedelta(days=2)
        
        # check if file exist
        initial = not os.path.exists(CSV_PATH)
        if initial:
            f = open(CSV_PATH, 'w')
            f.close()
        
        # store csv
        df = pd.DataFrame(no_fire_index_dict)
        df.to_csv(CSV_PATH, mode='a', header=initial, index=False)

def read_fire_index(path="dataset/fire_index.csv"):
    df = pd.read_csv(path)
    lngs = df["longitude"].tolist()
    lats = df["latitude"].tolist()
    years = df["acq_date"].str.split("-").str[0]
    months = df["acq_date"].str.split("-").str[1]
    days = df["acq_date"].str.split("-").str[2]
    hours = 18

    date_times = []
    for i in range(len(years)):
        y = int(years[i])
        m = int(months[i])
        d = int(days[i])
        h = int(hours)
        date_time = datetime.datetime(y,m,d,h)
        date_times.append(date_time)

    print(len(lngs), len(lats), len(date_times))
    return lngs, lats, date_times

def read_no_fire_index(path="dataset/no_fire_index.csv"):
    df = pd.read_csv(path)
    lngs = df["longitude"].tolist()
    lats = df["latitude"].tolist()
    years = df["acq_date"].str.split("-").str[0]
    months = df["acq_date"].str.split("-").str[1]
    days = df["acq_date"].str.split("-").str[2]
    hours = df["acq_time"]

    date_times = []
    for i in range(len(years)):
        y = int(years[i])
        m = int(months[i])
        d = int(days[i])
        h = int(hours[i])
        date_time = datetime.datetime(y,m,d,h)
        date_times.append(date_time)

    print(len(lngs), len(lats), len(date_times))
    return lngs, lats, date_times

def make_dataset():
    # date_time = datetime.datetime(2023, 3, 21, 1)
    # append_dataset(date_time=date_time, loc=(122, 23.4))

    lngs, lats, date_times = read_fire_index()
    n_lngs, n_lats, n_date_times = read_no_fire_index()

    for i in range(len(lngs)):
        print(f"lng: {lngs[i]}, lat: {lats[i]}, datetime: {date_times[i].__str__()}")
        append_dataset(loc=(lngs[i], lats[i]), date_time=date_times[i], is_fire=True)  
    
    # for i in range(len(n_lngs)):
    #     print(f"lng: {n_lngs[i]}, lat: {n_lats[i]}, datetime: {n_date_times[i].__str__()}")
    #     append_dataset(loc=(n_lngs[i], n_lats[i]), date_time=n_date_times[i], is_fire=False)  

def push_data(source="dataset/dataset_temp.csv", target="dataset/dataset.csv"):
    initial =  not os.path.exists(target)

    if not os.path.exists(source):
        print("cannot find source file")
        exit()

    if not os.path.exists(target):
        f = open(target, 'w')
        f.close()

    df = pd.read_csv(source)
    df.to_csv(target, mode='a', header=initial, index=False)

    print(df.tail())

def clean_cluster(path="dataset/landsat_12-22_fire.csv"):
    df = pd.read_csv(path).iloc[:, 1:]
    
    def is_far(loc1: tuple, loc2: tuple):
        if abs(loc1[0] - loc2[0]) < 0.1 and abs(loc1[1] - loc2[1]) < 0.1:
            return False
        return True

    df_concat = []
    all_dates = df["acq_date"].unique()

    for date in all_dates:
        distinct_loc = []
        del_ind = []

        df_date = df[df["acq_date"] == date].reset_index()
        print(df_date)

        for i in range(df_date.shape[0]):
            s_lng = round(df_date["longitude"].iloc[i].item(), 2)
            s_lat = round(df_date["latitude"].iloc[i].item(), 2)
            s_loc = (s_lng, s_lat)
            # print(i)
            # print(s_loc)

            if len(distinct_loc) == 0:
                distinct_loc.append(s_loc)
            else:
                delete = False
                for loc in distinct_loc:
                    # print("loc is ", loc)
                    if not is_far(loc, s_loc):
                        # print("close, delete it")
                        del_ind.append(i)
                        delete = True
                        break

                if delete == False:
                    # print("far, append it")
                    distinct_loc.append(s_loc)
            
        df_date = df_date.drop(del_ind)
        df_concat.append(df_date)

    new_df = pd.concat(df_concat)
    new_df.to_csv("landsat_12-22_fire_new.csv")

class FPDataset():
    def __init__(self, stage="train", mode="DNN") -> None:
        super().__init__()
        self.stage = stage
        self.mode = mode
        TRAIN_DIR = "dataset/dataset.csv"
        TEST_DIR = "dataset/dataset.csv"

        # prepare dataset
        if stage == "train":
            df = pd.read_csv(TRAIN_DIR)
            is_fire = df.pop("is_fire")
            self.is_fire = torch.from_numpy(is_fire.to_numpy()).to(torch.float32)
        elif stage == "test":
            df = pd.read_csv(TEST_DIR)
        else:
            print("wrong stage")
            raise ValueError

        # normalize
        numTypeCol = df.select_dtypes(["int", "float", "double"]).columns
        df[numTypeCol] = (df[numTypeCol] - df[numTypeCol].mean()) / df[numTypeCol].std()

        if mode == "DNN":
            # get k best
            if stage == "train":
                one_hot_pd = pd.get_dummies(df)
                scores = list(SelectKBest(score_func=f_regression).fit(df, self.is_fire).scores_)
                feats = [x for x in map(scores.index, heapq.nlargest(3, scores))]
                print("Columns with highest correlation", one_hot_pd.columns[feats])

            # one hot
            # str_cols = list(df.select_dtypes("object").columns)
            # str_df = df[str_cols]
            # df.drop(str_cols, axis=1, inplace=True)

            # if stage == "train":
            #     train_str_df = str_df
            # elif stage == "test":
            #     if os.path.exists(TRAIN_DIR):
            #         train_df = pd.read_csv(TRAIN_DIR).drop("單價", axis=1)
            #         train_str_df = train_df[str_cols]
            #     else:
            #         print("please prepare train data in advance")
            #         raise FileNotFoundError
            
            # enc = OneHotEncoder(handle_unknown='ignore', drop='if_binary').fit(train_str_df) # only encode string part
            # one_hot = pd.DataFrame(enc.transform(str_df).toarray())
            # one_hot = pd.concat([df, one_hot], axis=1)
            # one_hot_np = one_hot.to_numpy()

            # get dataset
            self.dataset = torch.from_numpy(df.to_numpy()).to(torch.float32)

        elif mode == "RNN":
            # TODO: Add day data
            max_hours = 5
            cols = df.columns
            cols_h = []
            for h in range(max_hours):
                cols_h.append([x for x in cols if (f"{h}h" in x)])

            data_array = []
            for i in range(df.shape[0]):
                data_array.append([])
                for h in range(max_hours):
                    hour_data = df.iloc[i, :].loc[cols_h[h]]
                    # print(hour_data)
                    hour_data = hour_data.to_list()
                    data_array[-1].append(hour_data)
            data_torch = torch.Tensor(data_array).to(dtype=torch.float32)
            self.dataset_hours = data_torch
            # print(data_torch.shape)
                
        else:
            print("wrong mode")
            raise ValueError
    
    def __getitem__(self, index):
        if self.mode == "DNN":
            if self.stage == "train":
                return self.dataset[index], self.is_fire[index]
            elif self.stage == "test":
                return self.dataset[index]
            else:
                print("wrong input")
        
        elif self.mode == "RNN":
            # TODO: Days dataset
            if self.stage == "train":
                return self.dataset_hours[index], self.is_fire[index]
            elif self.stage == "test":
                return self.dataset[index]
            else:
                print("wrong input")
        
    def __len__(self):
        if self.mode == "DNN":
            return self.dataset.shape[0]
        elif self.mode == "RNN":
            return self.dataset_hours.shape[0]
    
    def dim(self):
        if self.mode == "DNN":
            return self.dataset.shape[-1]

        elif self.mode == "RNN":
            return self.dataset_hours.shape[-1]



if __name__ == '__main__':

    # ds = FPDataset(stage="train", mode="RNN")
    # print(ds.dim())
    # data = DataLoader(ds)
    # for i, (x, y) in enumerate(data):
    #     if i > 0:
    #         break
    #     print(x.shape)

    df = pd.read_csv("dataset/no_fire_index.csv")