import Dataset
import Model
from callback_utils import PrintingCallback, TestingCallback

import torch
from torchsummary import summary
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import lightning as L

import requests as re
import pandas as pd
import yaml
import json

import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import time

OUTPUT_DIR = "."
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"

def process(df: pd.DataFrame):
    # ensure, should not change
    MIN_DAYS_DELTA = 3
    HOURS_DELTA = 12
    DAYS_DELTA = 12

    # TODO: Add day data
    cols = df.columns
    cols_d = []

    for d in range(MIN_DAYS_DELTA, DAYS_DELTA + 1):
        cols_d.append([x for x in cols if (f"_{d}d" in x)])

    # days
    data_array = []
    for i in range(df.shape[0]):
        data_array.append([])
        for d in range(DAYS_DELTA - MIN_DAYS_DELTA + 1): # the 3, 4, ...., 12 -> 0, 1, ..., 9
            day_data = df.iloc[i, :].loc[cols_d[d]]
            # print(hour_data)
            day_data = day_data.to_list()
            data_array[-1].append(day_data)
    data_torch = torch.Tensor(data_array).to(dtype=torch.float32)
    dataset_days = data_torch
    print(data_torch.shape)
    
    return dataset_days

def get_current_climate_data(loc: tuple):
    date_time = datetime.datetime.now()
    now_date = date_time.__str__().split(" ")[0]
    now_time = date_time.__str__().split(" ")[1].split(":")[0]
    print(f"date_time: {date_time}, loc: {loc}")

    try:
        community="RE"
        params_hour = ["T2M", "T2MDEW", "T2MWET", "TS", "WS2M", "WD2M"]
        params_day = ["T2M", "T2MDEW", "T2MWET", "TS", "WS2M", "WD2M", "TS_MAX"]
        params_month = ["T2M", "T2MDEW", "T2MWET", "TS", "WS2M", "WD2M", "TS_MAX"]

        # process params
        params_str_hours = ",".join(params_hour)
        params_str_days = ",".join(params_day)
        params_str_months = ",".join(params_month)

        return_data_dict = {
            "lng": [loc[0]],
            "lat": [loc[1]],
            "date": [now_date.__str__().split(" ")[0]],
            "time": [now_time]
        }
        
        ##### HOUR #####
        # process date (get the last five hours)
        have_hours = False
        if have_hours:
            HOURS_DELTA = 5
            start_datetime = date_time - timedelta(hours=HOURS_DELTA)
            start_date = start_datetime.__str__().split(" ")[0]
            start_date = f"{start_date.split('-')[0]}{start_date.split('-')[1]}{start_date.split('-')[2]}"

            end_datetime = date_time - timedelta(hours=1)
            end_date = end_datetime.__str__().split(" ")[0]
            end_date = f"{end_date.split('-')[0]}{end_date.split('-')[1]}{end_date.split('-')[2]}"

            # response
            url = f"https://power.larc.nasa.gov/api/temporal/hourly/point?header=false&parameters={params_str_hours}&community={community}&longitude={loc[0]}&latitude={loc[1]}&start={start_date}&end={end_date}&format=JSON"
            response = re.get(url=url, verify=True, timeout=30.00)
            content = response.content.decode('utf-8')
            datas = json.loads(content)["properties"]["parameter"] # content dict

            for param in datas.keys():
                for delta_hours in range(1, HOURS_DELTA + 1):
                    # process datetime index
                    date, TIME = (date_time - timedelta(days=MIN_DAYS_DELTA, hours=delta_hours)).__str__().split(" ")
                    y, m, d = date.split("-")
                    h = TIME.split(":")[0]
                    date_time_str = f"{y}{m}{d}{h}"
                    return_data_dict[f"{param}_{delta_hours}h"] = [datas[param][date_time_str]]    

        ### DAYS ####
        # process date (get the last five hours)
        DAYS_DELTA = 20
        MIN_DAYS_DELTA = 3

        start_datetime = date_time - timedelta(days=DAYS_DELTA)
        start_date = start_datetime.__str__().split(" ")[0]
        start_date = f"{start_date.split('-')[0]}{start_date.split('-')[1]}{start_date.split('-')[2]}"

        end_datetime = date_time - timedelta(days=MIN_DAYS_DELTA)
        end_date = end_datetime.__str__().split(" ")[0]
        end_date = f"{end_date.split('-')[0]}{end_date.split('-')[1]}{end_date.split('-')[2]}"
        
        # response
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?header=false&parameters={params_str_days}&community={community}&longitude={loc[0]}&latitude={loc[1]}&start={start_date}&end={end_date}&format=JSON"
        response = re.get(url=url, verify=True, timeout=30.00)
        content = response.content.decode('utf-8')
        datas = json.loads(content)["properties"]["parameter"] # content dict


        for param in datas.keys():
            for delta_days in range(MIN_DAYS_DELTA, DAYS_DELTA + 1):
                # process datetime index
                date, TIME = (date_time - timedelta(days=delta_days)).__str__().split(" ")
                y, m, d = date.split("-")
                h = TIME.split(":")[0]
                date_time_str = f"{y}{m}{d}"
                return_data_dict[f"{param}_{delta_days}d"] = [datas[param][date_time_str]]

        df = pd.DataFrame(return_data_dict)
        time.sleep(2)

        return process(df)
    
    except KeyError as e:
        print(response)
        print(e)

        if response.status_code == 429:
            time.sleep(300)
        
        return pd.DataFrame()

def train(epoch=10, data_module=Model.FirePredcitDM(), config_dir="config.yaml"):
    # dataset
    input_dim = data_module.input_dim()

    # configuration
    with open(config_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["input_dim"] = input_dim

    print("==================Training Configuration===================")
    for x, y in config.items():
        print(f"{x}: {y}")
    print("===========================================================")

    # build model
    model = Model.DNN(**config)
    # summary(model)

    # callback
    MyEarlyStopping = EarlyStopping(
        monitor="val_loss",  # monitor string is the label in logging
        min_delta=0,
        patience=config["earlystopping_patience"],
        mode='min'
    )

    SaveBestCheckpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename='BESTcheckpoint--{epoch}-{val_loss:.2f}',
        save_top_k=1,
    )

    # train
    trainer = L.Trainer(
        max_epochs=epoch, 
        default_root_dir=OUTPUT_DIR, 
        accelerator=ACCELERATOR, 
        callbacks=[PrintingCallback(), MyEarlyStopping, SaveBestCheckpoint],
        devices="auto")
    trainer.fit(model, data_module) # the data from module would be move to the same device as model defaulty by lightning

def train_rnn(epoch=10, data_module=Model.FirePredcitDM(mode="RNN"), config_dir="config.yaml"):
    torch.set_float32_matmul_precision('medium')

    # input dim
    input_dim_days = data_module.input_dim() #TODO: This should have two values after daily data is appended
    
    # configuration
    with open(config_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["input_dim_days"] = input_dim_days

    # build model
    model = Model.FirPRNN(**config)
    # summary(model)

    # callback
    MyEarlyStopping = EarlyStopping(
        monitor="val_loss",  # monitor string is the label in logging
        min_delta=0,
        patience=config["earlystopping_patience"],
        mode='min'
    )

    SaveBestCheckpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename='BESTcheckpoint--{epoch}-{val_loss:.2f}',
        save_top_k=1,
    )

    # train
    trainer = L.Trainer(
        max_epochs=epoch, 
        default_root_dir=OUTPUT_DIR, 
        accelerator=ACCELERATOR, 
        callbacks=[PrintingCallback(), MyEarlyStopping, SaveBestCheckpoint],
        devices="auto")
    trainer.fit(model, data_module) # the data from module would be move to the same device as model defaulty by lightning

def predict(ckpt_dir: str):
    CKPT_DIR = ckpt_dir
    HP_DIR = "/".join(CKPT_DIR.split("/")[:2]) + "/hparams.yaml"

    # build model
    model = Model.FirPRNN.load_from_checkpoint(CKPT_DIR, hparams_file=HP_DIR, map_location=torch.device(DEVICE))
    model.eval() # evaluation mode
    model.freeze()

    result_dict = {
        "lng": [],
        "lat": [],
        "fire_probibility": []
    }

    with torch.no_grad():
        input_tensor = torch.Tensor()
        # for lng in range(-124.75, -64.75, 0.5):
        #     for lat in range(24.75, 54.75, 0.5):
        cent_lng = -118.8
        cent_lat = 34

        for lng in np.arange(cent_lng - 2, cent_lng + 2, 0.5):
            for lat in np.arange(cent_lat - 0.5, cent_lat + 2, 0.5):
                # the input is initially on cpu, while the model is loaded to GPU by default
                input_day = get_current_climate_data((lng, lat)) # 
                input_day = input_day.to(device=DEVICE)
                
                # make predict
                fire_prob = model(input_day).item()

                # store result
                result_dict["lng"].append(lng)
                result_dict["lat"].append(lat)
                result_dict["fire_probibility"].append(fire_prob)

                # wait
                time.sleep(3)

    
    result = pd.DataFrame(result_dict)
    result.to_csv("result.csv", index=False)

# predict("lightning_logs/version_30/checkpoints/BESTcheckpoint--epoch=8-val_loss=0.45.ckpt")
train_rnn(epoch=2000)