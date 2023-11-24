import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler
# print(plt.style.available)
plt.style.use('seaborn-v0_8-whitegrid')
myColors = [(0.00784313725490196, 0.4470588235294118, 0.6352941176470588), (0.6235294117647059, 0.7647058823529411, 0.4666666666666667), (0.792156862745098, 0.043137254901960784, 0.011764705882352941), (0.6470588235294118, 0.00784313725490196, 0.34509803921568627), (0.8431372549019608, 0.7803921568627451, 0.011764705882352941), (0.5333333333333333, 0.792156862745098, 0.8549019607843137)]

mc = LinearSegmentedColormap.from_list('', myColors)
newColor = mc(np.linspace(0, 1, 8))
# print( )
custom = cycler(color=newColor)
plt.rc('axes', prop_cycle=custom)

from datetime import datetime
dateparse = lambda x,y: datetime.strptime(x+" "+y, '%d/%m/%y %H:%M:%S')
bamparse = lambda x: datetime.strptime(x, '%H:%M %m/%d/%Y')
anotherparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split as split_data
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from scipy.stats import pearsonr, spearmanr
# from scipy.stats import tstd as std_deviation
from numpy import std as std_deviation
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import r2_score
from my_metrics import precision
from my_metrics import mean_percent_precision as mpp
from my_metrics import mean_bias_error as mbs

def clean_umi(data):
    # print(data["Umidade"].iloc[0])
    for i in range(1,len(data["Umidade"].values)):
        if abs(data["Umidade"].iloc[i] - data["Umidade"].iloc[i-1])>15:
            data["Umidade"].iloc[i] = data["Umidade"].iloc[i-1]


# Converte os arquivos individuais para um só
def convert(save=False):
    test = pd.DataFrame()
    for mes in range(8,12):
        for dia in range(1,32):
            try:
                # print(str(dia)+str(mes)+"23.csv")
                aux = pd.read_csv(str(dia)+str(mes)+"23.csv", header=None, sep=";", decimal=".")

                if test.empty:
                    test = aux
                else:
                    # print("conca")
                    test = test.append(aux, ignore_index=True)
                    # print(test.tail())
            except:
                print(f"Não tem dia {dia}/{mes}/23")
    if save:
        test.to_csv("excel/DUSTAI.csv", sep=";", decimal=",", index=False, header=["Data","Hora","Num","Temperatura","Umidade","MP2,5_1","MP10_1","MP2.5_2","MP10_2","NaN"])
    return test

def load_all_mp_bc(mp10=False):
    sensor_list = ["SDS011a", "SDS011b", "SDS018", "HPMAa", "HPMAb", "PMS7003a", "PMS7003b"]
    # alg_list = ["Multiple Linear", "Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost", "LGBM", "CatBoost"]
    ref = "BAM"
    mp_ponto = "MP2.5"
    mp_virgula = "MP2,5"
    if mp10:
        mp_ponto = "MP10"
        mp_virgula = "MP10"
    data = pd.read_csv("minuto/SDS011a.csv", decimal=".", parse_dates=["Data/Hora"], sep=";", date_parser=anotherparse)
    # data.columns = ["Data/Hora", mp_virgula+" SDS011a", "Temperatura", "Umidade", mp_virgula+" BAM 1020", mp_virgula+" SDS011a Multiple Linear"]
    data = data.set_index(["Data/Hora"])
    data = data.mask(data<0)
    data = data.mask(data["Umidade"]>95)
    # print(data.duplicated(keep="first"))
    data = data.resample("H").mean()
    # data.reset_index(inplace=True, drop=True)
    # data.index = pd.to_datetime(data.index)
    # print(data.head())
    for sensor in sensor_list:
        if sensor == "SDS011a":
            continue
        aux = pd.read_csv("minuto/"+sensor+".csv", decimal=".", parse_dates=["Data/Hora"], usecols=[0,3,4], sep=";", date_parser=anotherparse)
        # aux.columns = ["Data/Hora", mp_virgula+" "+sensor]
        aux = aux.set_index(["Data/Hora"])
        aux = aux.mask(aux<0)
        # aux.reset_index(inplace=True, drop=True)
        aux = aux.resample("H").mean()
        data = pd.concat([data, aux], axis=1)
    # for sensor in sensor_list:
    #     for alg in alg_list:
    #         if sensor == "SDS011a" and alg == "Multiple Linear":
    #             continue
    #         aux = pd.read_csv("finalized/"+mp_ponto+" "+sensor+" "+mp_ponto+" "+ref+"/"+alg+".csv", decimal=",", parse_dates=["Data/Hora"], usecols=[0,5], sep=";")
    #         aux.columns = ["Data/Hora", mp_virgula+" "+sensor+" "+alg]
    #         aux = aux.set_index(["Data/Hora"])
    #         data = pd.concat([data, aux], axis=1)

    # data = data.resample("H").mean()
    return data

def load():
    data = pd.read_csv("excel/DUSTAI.csv", header=0, sep=";", decimal=",", usecols=[0,1,3,4,5,6,7,8], parse_dates=[["Data", "Hora"]], date_parser=dateparse)
    data = data.set_index(["Data_Hora"])
    # data = data.drop(data[data["Umidade"]>95].index)
    data = data.mask(data["Umidade"]>95)
    data = data.mask(data<0)
    clean_umi(data)
    data = data.fillna(np.nan, axis=0)
    # data = data.resample("10min").mean()
    data = data.resample("H").mean()
    # data = data.drop(data[data["Umidade"]<29].index)
    # data[data["Umidade"]<29] = None
    # data.fillna(method="ffill", inplace=True)
    return data

def plot(data, bam=False, cal=False):
    fig, axes = plt.subplots(nrows=3, ncols=1)
    # fig.tight_layout()
    to25 = ["MP2,5_1","MP2,5_2"]
    to10 = ["MP10_1","MP10_2"]
    if cal:
        to25 = ["MP2,5_1_Cal","MP2,5_2_Cal"]
        to10 = ["MP10_1_Cal","MP10_2_Cal"]
    if bam:
        to25 += ["MP2,5"]
        to10 += ["MP10"]

    data[to25].plot(ax=axes[0], sharex=True, ylabel="$ug/m^3$").legend(loc="upper left", frameon=True)
    color1 = next(axes[0]._get_lines.prop_cycler)['color']
    color2 = next(axes[0]._get_lines.prop_cycler)['color']
    color3 = next(axes[0]._get_lines.prop_cycler)['color']
    data[to10].plot(ax=axes[1], sharex=True, color=[color1, color2, color3], ylabel="$ug/m^3$").legend(loc="upper left", frameon=True)
    color1 = next(axes[0]._get_lines.prop_cycler)['color']
    color2 = next(axes[0]._get_lines.prop_cycler)['color']
    # color2 = next(axes[0]._get_lines.prop_cycler)['color']
    data[["Temperatura","Umidade"]].plot(ax=axes[2], sharex=True, color=[color1, color2], ylabel="$\u00B0$C/%", xlabel="Data").legend(loc="upper left", frameon=True)
    start, end = axes[0].get_xlim()
    date_start, date_end = min(data.index), max(data.index)
    axes[0].xaxis.set_ticks(np.linspace(start, end, 6))
    axes[1].xaxis.set_ticks(np.linspace(start, end, 6))
    test = np.linspace(start, end, 6)
    test_date = np.linspace(datetime.timestamp(date_start), datetime.timestamp(date_end), 6)
    axes[2].xaxis.set_ticks(test)
    axes[2].set_xticklabels([datetime.fromtimestamp(x).strftime("%d/%m/%y") for x in test_date], rotation=45)
    # plt.gcf().autofmt_xdate()

    # axes[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    
def plot_corr(data):
    plt.figure("Heatmap")
    corr = data.corr(method="pearson")
    sns.heatmap(corr, cmap="coolwarm", annot=True)

def load_bam(file="excel/BAM_DUSTAI.csv"):
    data = pd.read_csv(file, header=0, delimiter=';', parse_dates=True, dayfirst=True, decimal=",", date_parser=bamparse)
    # data = data.dropna(how='any', axis=0)
    data = data.set_index("Data/Hora")
    data.index = pd.to_datetime(data.index, format='%H:%M %m/%d/%Y')
    data.index -= pd.Timedelta('2H')
    # print(data.head())

    data = data.mask(data<0)
    data = data.mask(data>370)
    data = data.fillna(np.nan, axis=0)
    data = data.resample("H").mean()
    print(data.describe())
    # data = data.dropna(how='any', axis=0)

    return data

def load_bam_before():
    data = pd.read_csv("minuto/BAM.csv", header=0, delimiter=';', parse_dates=True, decimal=".", date_parser=bamparse)
    data = data.set_index("Data/Hora")
    data.index = pd.to_datetime(data.index, format='%H:%M %m/%d/%Y')
    data.index -= pd.Timedelta('2H')
    # print(data.head())

    data = data.mask(data<0)
    data = data.mask(data>370)
    # print(data.head())
    # data = data.mask(data.index>"2023-03-20")
    data = data.fillna(np.nan, axis=0)
    data = data.resample("H").mean()
    # print(data.describe())
    return data

def plot_bam(data):
    # plt.figure("bam")
    fig, axes = plt.subplots(nrows=3, ncols=1)
    # fig.tight_layout()
    data[["MP2,5"]].plot(ax=axes[0], sharex=True, ylabel="$ug/m^3$").legend(loc="upper left", frameon=True)
    color1 = next(axes[0]._get_lines.prop_cycler)['color']
    color2 = next(axes[0]._get_lines.prop_cycler)['color']
    data[["MP10"]].plot(ax=axes[1], sharex=True, color=[color1, color2], ylabel="$ug/m^3$").legend(loc="upper left", frameon=True)
    color1 = next(axes[0]._get_lines.prop_cycler)['color']
    color2 = next(axes[0]._get_lines.prop_cycler)['color']
    data[["Temperatura","Umidade"]].plot(ax=axes[2], sharex=True, color=[color1, color2], ylabel="$\u00B0$C/%", xlabel="Data").legend(loc="upper left", frameon=True)
    start, end = axes[0].get_xlim()
    date_start, date_end = min(data.index), max(data.index)
    axes[0].xaxis.set_ticks(np.linspace(start, end, 6))
    axes[1].xaxis.set_ticks(np.linspace(start, end, 6))
    test = np.linspace(start, end, 6)
    test_date = np.linspace(datetime.timestamp(date_start), datetime.timestamp(date_end), 6)
    axes[2].xaxis.set_ticks(test)
    axes[2].set_xticklabels([datetime.fromtimestamp(x).strftime("%d/%m/%y") for x in test_date], rotation=45)

def calib(data):
    # CALIB DUAS SEMASN
    # train, test = split_data(data, train_size=0.7, test_size=0.3, random_state=100) # todos os dados
    # train, test = split_data(data, train_size=168, random_state=100, shuffle=False) # duas semanas
    # train, test = split_data(data[data.index<"2023-09-25"], train_size=336, random_state=100, shuffle=False) # duas semanas com inicio em 25/09

    # aux = joblib.load("MP2.5 HPMAa MP2.5 BAM/CatBoost.joblib")
    # model = CatBoostRegressor()
    # model.set_params(**aux.get_params())
    # model.fit(train[["MP2,5_1", "Temperatura", "Umidade"]].values, train["MP2,5"].values)
    # data["MP2,5_1_Cal"] = model.predict(data[["MP2,5_1", "Temperatura", "Umidade"]].values)

    # aux = joblib.load("MP2.5 HPMAa MP2.5 BAM/Gradient Boosting.joblib")
    # model = GradientBoostingRegressor()
    # model.set_params(**aux.get_params())
    # model.fit(train[["MP10_1", "Temperatura", "Umidade"]].values, train["MP10"].values)
    # data["MP10_1_Cal"] = model.predict(data[["MP10_1", "Temperatura", "Umidade"]].values)

    # aux = joblib.load("MP2.5 SDS011a MP2.5 BAM/XGBoost.joblib")
    # model = XGBRegressor()
    # model.set_params(**aux.get_params())
    # model.fit(train[["MP2,5_2", "Temperatura", "Umidade"]].values, train["MP2,5"].values)
    # data["MP2,5_2_Cal"] = model.predict(data[["MP2,5_2", "Temperatura", "Umidade"]].values)

    # aux = joblib.load("MP10 SDS011a MP10 BAM/XGBoost.joblib")
    # model = XGBRegressor()
    # model.set_params(**aux.get_params())
    # model.fit(train[["MP10_2", "Temperatura", "Umidade"]].values, train["MP10"].values)
    # data["MP10_2_Cal"] = model.predict(data[["MP10_2", "Temperatura", "Umidade"]].values)

    # CALB MODEL ANTERIOR
    model = joblib.load("MP2.5 HPMAa MP2.5 BAM/CatBoost.joblib")
    data["MP2,5_1_Cal"] = model.predict(data[["MP2,5_1", "Temperatura", "Umidade"]].values)
    model = joblib.load("MP10 HPMAa MP10 BAM/Gradient Boosting.joblib")
    data["MP10_1_Cal"] = model.predict(data[["MP10_1", "Temperatura", "Umidade"]].values)
    model = joblib.load("MP2.5 SDS011a MP2.5 BAM/XGBoost.joblib")
    data["MP2,5_2_Cal"] = model.predict(data[["MP2,5_2", "Temperatura", "Umidade"]].values)
    model = joblib.load("MP10 SDS011a MP10 BAM/XGBoost.joblib")
    data["MP10_2_Cal"] = model.predict(data[["MP10_2", "Temperatura", "Umidade"]].values)
    return data

def calib_before(data):
    alg_list = ["Multiple Linear", "Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost", "LGBM", "CatBoost"]
    for alg in alg_list:
        model = joblib.load("finalized/MP2.5 SDS011a MP2.5 BAM/"+alg+".joblib")
        data["MP2.5 "+alg] = model.predict(data[["MP2.5 SDS011a", "Temperatura", "Umidade"]].values)
        model = joblib.load("finalized/MP10 SDS011a MP10 BAM/"+alg+".joblib")
        data["MP10 "+alg] = model.predict(data[["MP10 SDS011a", "Temperatura", "Umidade"]].values)
        # aux = pd.read_csv("finalized/"+mp_ponto+" "+sensor+" "+mp_ponto+" "+ref+"/"+alg+".csv", decimal=",", parse_dates=["Data/Hora"], usecols=[0,5], sep=";")
        # aux.columns = ["Data/Hora", mp_virgula+" "+sensor+" "+alg]
        # aux = aux.set_index(["Data/Hora"])
        # data = pd.concat([data, aux], axis=1)
    return data

def calc_metrics(data):
    metrics = ["R2", "PearsonR", "SpearmanR", "MAE", "MSE", "Bias", "StdDev"]
    cols = ["MP2,5_1","MP2,5_1_Cal","MP10_1","MP10_1_Cal","MP2,5_2","MP2,5_2_Cal","MP10_2","MP10_2_Cal"]
    results_dataframe = pd.DataFrame(columns=data[cols].columns.values, index=metrics)
    for sensor in cols:
        ref = "MP10"
        if sensor.startswith("MP2,5"):
            ref = "MP2,5"
        results_dataframe[sensor].R2 = r2_score(data[ref], data[sensor])
        results_dataframe[sensor].PearsonR = pearsonr(data[sensor], data[ref])[0]
        results_dataframe[sensor].SpearmanR = spearmanr(data[sensor].values, data[ref].values).statistic
        results_dataframe[sensor].MAE = mae(data[ref].values, data[sensor].values)
        results_dataframe[sensor].MSE = mse(data[ref].values, data[sensor].values)
        results_dataframe[sensor].Bias = mbs(data[ref].values, data[sensor].values)
        results_dataframe[sensor].StdDev = data[sensor].std(ddof=0)
    print(results_dataframe.head(20))

def plot_before(data, col1, col2):
    fig, axes = plt.subplots(nrows=3, ncols=1)
    # print(data.shape)
    data = data[data.index>"2023-03-20"]


    data[col1].plot(ax=axes[0], sharex=True, ylabel="$ug/m^3$").legend(loc="upper left", frameon=True, fontsize=8)
    # color1 = next(axes[0]._get_lines.prop_cycler)['color']
    # color2 = next(axes[0]._get_lines.prop_cycler)['color']
    # color3 = next(axes[0]._get_lines.prop_cycler)['color']
    data[col2].plot(ax=axes[1], sharex=True, ylabel="$ug/m^3$").legend(loc="upper left", frameon=True, fontsize=8)
    # color1 = next(axes[0]._get_lines.prop_cycler)['color']
    # color2 = next(axes[0]._get_lines.prop_cycler)['color']
    # color2 = next(axes[0]._get_lines.prop_cycler)['color']
    data[["Temperatura","Umidade"]].plot(ax=axes[2], sharex=True, ylabel="$\u00B0$C/%", xlabel="Data").legend(loc="upper left", frameon=True, fontsize=8)
    start, end = axes[0].get_xlim()
    date_start, date_end = min(data.index), max(data.index)
    axes[0].xaxis.set_ticks(np.linspace(start, end, 6))
    axes[1].xaxis.set_ticks(np.linspace(start, end, 6))
    test = np.linspace(start, end, 6)
    test_date = np.linspace(datetime.timestamp(date_start), datetime.timestamp(date_end), 6)
    axes[2].xaxis.set_ticks(test)
    axes[2].set_xticklabels([datetime.fromtimestamp(x).strftime("%d/%m/%y") for x in test_date], rotation=45)

def plot_before_toledo(data, col1):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    # print(data.shape)
    # data = data[(data.index>"2021-11-10") & (data.index<"2021-12-11")] #TEOM
    data = data[(data.index>"2021-09-22") & (data.index<"2021-12-31")] #PDR
 

    data[col1].plot(ax=axes[0], sharex=True, ylabel="$ug/m^3$").legend(loc="upper left", frameon=True, fontsize=8)
    # color1 = next(axes[0]._get_lines.prop_cycler)['color']
    # color2 = next(axes[0]._get_lines.prop_cycler)['color']
    # color3 = next(axes[0]._get_lines.prop_cycler)['color']
    # data[col2].plot(ax=axes[1], sharex=True, ylabel="$ug/m^3$").legend(loc="upper left", frameon=True, fontsize=8)
    # color1 = next(axes[0]._get_lines.prop_cycler)['color']
    # color2 = next(axes[0]._get_lines.prop_cycler)['color']
    # color2 = next(axes[0]._get_lines.prop_cycler)['color']
    data[["Temperatura","Umidade"]].plot(ax=axes[1], sharex=True, ylabel="$\u00B0$C/%", xlabel="Data").legend(loc="upper left", frameon=True, fontsize=8)
    start, end = axes[0].get_xlim()
    date_start, date_end = min(data.index), max(data.index)
    axes[0].xaxis.set_ticks(np.linspace(start, end, 6))
    # axes[1].xaxis.set_ticks(np.linspace(start, end, 6))
    test = np.linspace(start, end, 6)
    test_date = np.linspace(datetime.timestamp(date_start), datetime.timestamp(date_end), 6)
    axes[1].xaxis.set_ticks(test)
    axes[1].set_xticklabels([datetime.fromtimestamp(x).strftime("%d/%m/%y") for x in test_date], rotation=45)

def calib_toledo(data):
    alg_list = ["Multiple Linear", "Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost", "LGBM", "CatBoost"]
    for alg in alg_list:
        # model = joblib.load("finalized/MP10 PMS7003 MP10 PDR/"+alg+".joblib")
        model = joblib.load("finalized/MP10 PMS7003 MP10 PDR/"+alg+".joblib")
        data["MP10 "+alg] = model.predict(data[["MP10 PMS7003", "Temperatura", "Umidade"]].values)
        # model = joblib.load("finalized/MP10 SDS011a MP10 BAM/"+alg+".joblib")
        # data["MP10 "+alg] = model.predict(data[["MP10 SDS011a", "Temperatura", "Umidade"]].values)
        # aux = pd.read_csv("finalized/"+mp_ponto+" "+sensor+" "+mp_ponto+" "+ref+"/"+alg+".csv", decimal=",", parse_dates=["Data/Hora"], usecols=[0,5], sep=";")
        # aux.columns = ["Data/Hora", mp_virgula+" "+sensor+" "+alg]
        # aux = aux.set_index(["Data/Hora"])
        # data = pd.concat([data, aux], axis=1)
    return data
    
if __name__ == "__dustai__":
    # convert(True)
    data = load()
    # print(data.describe())
    # plot(data)
    # plot_corr(data)

    bam = load_bam()
    # plot_bam(bam)

    
    # print(data.head(20))
    # print(bam.head(20))
    # data = data.append(bam[["MP2,5", "MP10"]])
    data = pd.concat([data, bam[["MP2,5", "MP10"]]], axis=1)
    data = calib(data.dropna(how='any', axis=0))
    data = data.resample("H").mean()
    print(data.describe())
    plot(data, True, True)


    # print(data.head(50))

    # plot(data, True)
    # plot_corr(data.dropna(how='any', axis=0))
    plot_corr(data[["Temperatura", "Umidade", "MP2,5_1_Cal", "MP10_1_Cal", "MP2,5_2_Cal", "MP10_2_Cal", "MP2,5", "MP10"]].dropna(how='any', axis=0))
    calc_metrics(data.dropna(how='any', axis=0))


    plt.show()

def load_teom():
    data = pd.read_csv("toledo/TEOM.csv", header=0, sep=';', parse_dates=True, decimal=".", date_parser=anotherparse, usecols=[0,1])
    # data = data.dropna(how='any', axis=0)
    data = data.set_index("Data/Hora")
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')
    # data.index -= pd.Timedelta('2H')
    # print(data.head())

    # data = data.mask(data<0)
    # data = data.mask(data["MP10 TEOM"]>370)
    # data = data.fillna(np.nan, axis=0)
    # data = data.resample("H").mean()
    # print(data.describe())
    return data

def load_pdr():
    data = pd.read_csv("toledo/PDR.csv", header=0, sep=';', parse_dates=True, decimal=".", date_parser=anotherparse, usecols=[0,1])
    # data = data.dropna(how='any', axis=0)
    data = data.set_index("Data/Hora")
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')
    # data.index -= pd.Timedelta('2H')
    # print(data.head())

    data = data.mask(data<0)
    data = data.mask(data["MP10 PDR"]>370)
    # data = data.fillna(np.nan, axis=0)
    data = data.resample("H").mean()
    # print(data.describe())
    return data

def load_toledo(mp10=True):
    sensor_list = ["SDS011", "SDS018", "HPMA", "HPMA", "PMS7003", "MIX6070"]
    # alg_list = ["Multiple Linear", "Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost", "LGBM", "CatBoost"]
    ref = "BAM"
    mp_ponto = "MP2.5"
    mp_virgula = "MP2,5"
    if mp10:
        mp_ponto = "MP10"
        mp_virgula = "MP10"
    data = pd.read_csv("toledo/SDS011.csv", decimal=".", parse_dates=["Data/Hora"], sep=";")
    # data.columns = ["Data/Hora", mp_virgula+" SDS011a", "Temperatura", "Umidade", mp_virgula+" BAM 1020", mp_virgula+" SDS011a Multiple Linear"]
    data = data.set_index(["Data/Hora"])
    data = data.mask(data<0)
    data = data.mask(data["Umidade"]>95)
    data = data.mask(data["MP10 SDS011"]>800)
    # print(data.describe())
    data = data.resample("H").mean()
    # data.reset_index(inplace=True, drop=True)
    # data.index = pd.to_datetime(data.index)
    # print(data.head())
    for sensor in sensor_list:
        if sensor == "SDS011":
            continue
        aux = pd.read_csv("toledo/"+sensor+".csv", decimal=".", parse_dates=["Data/Hora"], usecols=[0,3], sep=";")
        # aux.columns = ["Data/Hora", mp_virgula+" "+sensor]
        aux = aux.set_index(["Data/Hora"])
        aux = aux.mask(aux<0)
        # aux.reset_index(inplace=True, drop=True)
        aux = aux.resample("H").mean()
        data = pd.concat([data, aux], axis=1)
    # for sensor in sensor_list:
    #     for alg in alg_list:
    #         if sensor == "SDS011a" and alg == "Multiple Linear":
    #             continue
    #         aux = pd.read_csv("finalized/"+mp_ponto+" "+sensor+" "+mp_ponto+" "+ref+"/"+alg+".csv", decimal=",", parse_dates=["Data/Hora"], usecols=[0,5], sep=";")
    #         aux.columns = ["Data/Hora", mp_virgula+" "+sensor+" "+alg]
    #         aux = aux.set_index(["Data/Hora"])
    #         data = pd.concat([data, aux], axis=1)

    # data = data.resample("H").mean()
    return data

if __name__ == "__bam__":
    data = load_all_mp_bc()
    bam = load_bam_before()
    # bam.plot()
    data = pd.concat([data, bam], axis=1)
    plot_corr(data.dropna(how='any', axis=0))
    # data.plot()
    plot_before(data, ["MP2.5 SDS011a", "MP2.5 SDS011b", "MP2.5 SDS018", "MP2.5 HPMAa", "MP2.5 HPMAb", "MP2.5 PMS7003a", "MP2.5 PMS7003b", "MP2.5 BAM"], 
                ["MP10 SDS011a", "MP10 SDS011b", "MP10 SDS018", "MP10 HPMAa", "MP10 HPMAb", "MP10 PMS7003a", "MP10 PMS7003b", "MP10 BAM"])
    data = calib_before(data[["MP2.5 SDS011a", "MP10 SDS011a", "MP2.5 BAM", "MP10 BAM", "Temperatura","Umidade"]].dropna(how='any', axis=0))
    print(data.describe())
    alg_list25 = ["MP2.5 Random Forest", "MP2.5 Extra Trees", "MP2.5 Gradient Boosting", "MP2.5 XGBoost", "MP2.5 LGBM", "MP2.5 CatBoost", "MP2.5 BAM"]
    alg_list10 = ["MP10 Random Forest", "MP10 Extra Trees", "MP10 Gradient Boosting", "MP10 XGBoost", "MP10 LGBM", "MP10 CatBoost", "MP10 BAM"]
    data = data.resample("H").mean()
    plot_before(data, alg_list25, alg_list10)
    print(data.head())
    plt.show()

if __name__ == "__main__":
    data = load_toledo()
    teom = load_teom()
    pdr = load_pdr()
    # print(data.describe())
    # print(pdr.describe())
    # print(teom.describe())

    data = pd.concat([data, teom], axis=1)
    # data = pd.concat([data, pdr], axis=1)
    print(data.dropna(how="any", axis=0).describe())
    # # plot_corr(data.dropna(how='any', axis=0))

    # # plot_before_toledo(data, ["MP10 SDS011", "MP10 SDS018", "MP10 HPMA", "MP10 PMS7003", "MP10 MIX6070", "MP10 TEOM"])

    # data = calib_toledo(data.dropna(how='any', axis=0))
    # data = data.resample("H").mean()
    # alg_list10 = ["MP10 PMS7003", "MP10 Random Forest", "MP10 Extra Trees", "MP10 Gradient Boosting", "MP10 XGBoost", "MP10 LGBM", "MP10 CatBoost", "MP10 PDR"]
    # plot_before_toledo(data, alg_list10)
    # plot_corr(data[alg_list10].dropna(how='any', axis=0))

    plt.show()