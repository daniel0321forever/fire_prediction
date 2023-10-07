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

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import heapq
import requests


random.seed(datetime.datetime.now().timestamp())

def append_dataset(date_time: datetime, loc: tuple, is_fire: bool):
    # make file path
    CSV_PATH = os.path.join("dataset", "dataset_temp.csv")
    initial = not os.path.exists(CSV_PATH)
    if initial:
        f = open(CSV_PATH, 'w')
        f.close()

    # process date (get the last five hours)
    start_datetime = date_time - timedelta(hours=5)
    # print(start_datetime)
    start_date_ymd = start_datetime.__str__().split(" ")[0]
    start_date_hms = start_datetime.__str__().split(" ")[1]

    end_date_ymd = date_time.__str__().split(" ")[0]
    end_date_hms = date_time.__str__().split(" ")[1]

    date_ymd = f"{end_date_ymd.split('-')[0]}{end_date_ymd.split('-')[1]}{end_date_ymd.split('-')[2]}"
    time_h = f"{end_date_hms.split(':')[0]}"
    append_data = {
        "is_fire": int(is_fire),
        "lng": loc[0],
        "lat": loc[1],
        "date": date_ymd,
        "time": time_h
    }

    feature_5hrs = ["soil_type:idx", "relative_humidity_2m:p", "t_2m:C", "wind_speed_2m:ms", "wind_dir_2m:d", "forest_fire_warning:idx"]
    for i in range(len(feature_5hrs)):
        print(feature_5hrs)
        # url = f"https://api.meteomatics.com/2022-10-07T01:00:00Z--2023-10-07T02:00:00Z:PT1H/weather_symbol_24h:idx/52.520551,13.461804/json"
        # url = f"https://api.meteomatics.com/{start_date_ymd}T{start_date_hms}Z--{end_date_ymd}T{end_date_hms}Z:PT1H/weather_symbol_24h:idx/{loc[1]},{loc[0]}/json"
        url = f"https://api.meteomatics.com/{start_date_ymd}T{start_date_hms}Z--{end_date_ymd}T{end_date_hms}Z:PT1H/{feature_5hrs[i]}/{loc[1]},{loc[0]}/json"
        # Define the username and password
        username = 'nationaltaiwanuniersity_chou_yichieh'
        password = '1PO7Ukg0v9'

        # response
        try:
            # Send the HTTP GET or POST request with Basic Authentication
            response = requests.get(url, auth=(username, password))
            # response = re.get(url=url, verify=True, timeout=30.00)

            # Check the response status code and process the data accordingly
            if response.status_code == 200:
                # Successfully received data
                content = response.content.decode('utf-8')
                datas = json.loads(content) # content dict
                print(datas)
            else:
                # Handle error cases
                print(f"Request failed with status code {response.status_code}")
                print(response.text)  # Print the error response if needed

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
        
        try:        
            for data_item in datas.get('data', []):  # Iterate through the 'data' list
                param = data_item.get('parameter')  # Get the 'parameter' value
                coordinates = data_item.get('coordinates', [])  # Get the 'coordinates' list
                
                date_time_base = date_time
                for delta_hours in range(1, 6):
                    date_time = date_time_base - timedelta(hours=delta_hours)  # Calculate datetime
                    date_time_str = date_time.strftime("%Y-%m-%dT%H:%M:%SZ")  # Format datetime as a string

                    # Search for the corresponding value in the 'coordinates' list
                    value = None
                    for coordinate in coordinates:
                        dates = coordinate.get('dates', [])
                        for date in dates:
                            if date.get('date') == date_time_str:
                                value = date.get('value')
                                break  # Exit the loop if the value is found

                    # Store the value in the 'append_data' dictionary
                    if param and value is not None:
                        append_data[f"{param}_{delta_hours}h"] = [value]
                        
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

    # for i in range(len(lngs)):
    #     print(f"lng: {lngs[i]}, lat: {lats[i]}, datetime: {date_times[i].__str__()}")
    #     append_dataset(loc=(lngs[i], lats[i]), date_time=date_times[i], is_fire=True)  
    
    # "sat_ndvi:idx"
    for i in range(len(n_lngs)):
        print(f"lng: {n_lngs[i]}, lat: {n_lats[i]}, datetime: {n_date_times[i].__str__()}")
        append_dataset(loc=(n_lngs[i], n_lats[i]), date_time=n_date_times[i], is_fire=False)

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


class FPDataset():
    def __init__(self, stage="train", modify="apply") -> None:
        super().__init__()
        self.stage = stage
        TRAIN_DIR = "dataset/dataset.csv"
        TEST_DIR = "dataset/dataset.csv"
        
        file_made = os.path.exists(TRAIN_DIR) if stage=="train" else os.path.exists(TEST_DIR)

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

    
    def __getitem__(self, index):
        if self.stage == "train" or self.stage == "make_train":
            return self.dataset[index], self.is_fire[index]
        elif self.stage == "test":
            return self.dataset[index]
        else:
            print("wrong input")
    
    def __len__(self):
        return self.dataset.shape[0]
    
    def dim(self):
        return self.dataset.shape[1]

if __name__ == '__main__':
    data = DataLoader(FPDataset(stage="train"))
    make_dataset()
    # for i, (x, y) in enumerate(data):
    #     if i > 0:
    #         break
    #     print(x.shape)