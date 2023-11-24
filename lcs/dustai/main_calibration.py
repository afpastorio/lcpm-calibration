# Manipulação de dados
import pandas as pd
import numpy as np
# from load_data import remove_outlier_iqr as remove_outlier
# Gráficos
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
# Métricas
from scipy.stats import pearsonr, spearmanr
# from scipy.stats import tstd as std_deviation
from numpy import std as std_deviation
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import r2_score
# from my_metrics import precision
# from my_metrics import mean_percent_precision as mpp
# from my_metrics import mean_bias_error as mbs
# Utilidades
from random import randrange
from sklearn.model_selection import train_test_split as split_data
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
# from datetime import datetime
# custom_date_parser = lambda x: datetime.strptime(x, "%H:%M %m/%d/%Y")
# Algoritmos de calibração
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# from supervised.automl import AutoML

from main_data import myData, newColor, r2_adj

import joblib
from cycler import cycler
from matplotlib.dates import num2date, date2num
import datetime
import os

from model_config import *
from copy import deepcopy

# 25/09/23 16:10 COMEÇO 2 SEMANAS PROTO

def custom_pearson(y, y_pred, **kwargs):
    return pearsonr(y, y_pred)[0]

class myCalibrationModel:
    def __init__(self, myDataObject, reference:str, mp10=False):
        self._save = True
        self._train, self._test = split_data(myDataObject._allData, test_size=0.3, random_state=100, shuffle=True)
        self._data = myDataObject._allData

        # toScale = myDataObject._allData.columns.values
        # toScale = toScale[toScale != reference]

        # self._train[toScale] = StandardScaler().fit_transform(self._train[toScale])
        # self._test[toScale] = StandardScaler().fit_transform(self._test[toScale])
        # self._data[toScale] = StandardScaler().fit_transform(self._data[toScale])
        # self._timeCV = TimeSeriesSplit(n_splits=5, max_train_size=168, test_size=168)
        # self._timeCV = TimeSeriesSplit(n_splits=5, max_train_size=72, test_size=24)
        self._timeCV = 10 # CV random 
        self._reference = reference
        self._debug = False
        self._mp10 = mp10
        self._n_iter = 300
        sensor_list = []
        self._rawData = []
        self.metrics = ["R2", "PearsonR", "SpearmanR", "MAE", "MSE", "Bias", "StdDev"]
        self._myModels = dict()
        self._myParams = dict()

        if mp10:
            sensor_list = myDataObject.columns10
            for sensor in sensor_list:
                self._myModels[sensor] = deepcopy(my_models)
                self._myParams[sensor] = deepcopy(model_params)
            self._rawData = myDataObject._rawData10Scores
        else:
            sensor_list = myDataObject.columns25
            for sensor in sensor_list:
                self._myModels[sensor] = deepcopy(my_models)
                self._myParams[sensor] = deepcopy(model_params)
            self._rawData = myDataObject._rawData25Scores

        self._results = dict()
        datasets = ["TREINO", "TESTE", "TOTAL", "FINAL"]
        for sensor in sensor_list:
            self._results[sensor] = dict()
            for data in datasets:
                column_names = [sensor]
                for key, value in my_models.items():
                    column_names.append(key)
                self._results[sensor][data] =  pd.DataFrame(columns=column_names, index=self.metrics)
                self._results[sensor][data][sensor] = pd.DataFrame.copy(self._rawData[sensor], deep=True).to_frame()

        # trained = []
        # for sensor in sensor_list:
        #     for model_name, model in self._myModels[sensor].items():
        #         if os.path.isfile("lcs/aires/finalized/"+sensor+reference+"/"+model_name+".joblib"):
        #             print("Found model"+model_name+" for "+sensor)
        #             if sensor not in trained:
        #                 trained.append(sensor)
        #             model = joblib.load("lcs/aires/finalized/"+sensor+reference+"/"+model_name+".joblib")

        # for t in trained:
        #     sensor_list.remove(t)

        self.doModels(sensor_list)
        self.finalizeModels(sensor_list)

        # for t in trained:
        #     sensor_list.append(t)



        self.buildResultsData(sensor_list)

        plt.style.use('seaborn-v0_8-whitegrid')
        custom = cycler(color=[(0.00784313725490196, 0.4470588235294118, 0.6352941176470588), (0.6235294117647059, 0.7647058823529411, 0.4666666666666667), (0.792156862745098, 0.043137254901960784, 0.011764705882352941), (0.6470588235294118, 0.00784313725490196, 0.34509803921568627), (0.8431372549019608, 0.7803921568627451, 0.011764705882352941), (0.5333333333333333, 0.792156862745098, 0.8549019607843137)])
        plt.rc('axes', prop_cycle=custom)
        return
    
    def doModels(self, sensor_list):
        for sensor in sensor_list:
            for model_name, model in self._myModels[sensor].items():
                print("Doing "+model_name+" for "+sensor)

                if os.path.isfile("lcs/aires/finalized/"+sensor+" "+self._reference+"/"+model_name+".joblib"):
                    print("Found model "+model_name+" for "+sensor)
                    aux = joblib.load("lcs/aires/finalized/"+sensor+" "+self._reference+"/"+model_name+".joblib")
                    model.set_params(**aux.get_params(False))
                    self._myParams[sensor][model_name] = aux.get_params()
                    model.fit(self._train[[sensor, "Temperatura", "Umidade"]].values, self._train[self._reference].values)
                else:
                    if(model_name == "Multiple Linear"):
                        model.fit(self._train[[sensor, "Temperatura", "Umidade"]].values, self._train[self._reference].values)
                    else:
                        random_model = RandomizedSearchCV(model, self._myParams[sensor][model_name], n_iter=self._n_iter, cv=self._timeCV, verbose=0, n_jobs=-1, random_state=0)
                        random_model.fit(self._train[[sensor, "Temperatura", "Umidade"]].values, self._train[self._reference].values)
                        model = random_model.best_estimator_
                        self._myParams[sensor][model_name] = random_model.best_params_
                        print(random_model.best_params_)

                predict = model.predict(self._train[[sensor, "Temperatura", "Umidade"]].values)
                self.calcStats(self._results[sensor]["TREINO"][model_name], sensor, predict, self._train[self._reference])
                predict = model.predict(self._test[[sensor, "Temperatura", "Umidade"]].values)
                self.calcStats(self._results[sensor]["TESTE"][model_name], sensor, predict, self._test[self._reference])
                predict = model.predict(self._data[[sensor, "Temperatura", "Umidade"]].values)
                self.calcStats(self._results[sensor]["TOTAL"][model_name], sensor, predict, self._data[self._reference])

    def doAutoML(self, sensor_list):
        self._autoModels = dict()

        # val = {
        #     "validation_type": "split",
        #     "train_ratio": 0.7,
        #     "shuffle": True,
        #     "stratify": False,
        #     "random_seed": 32
        # }

        for sensor in sensor_list:
            self._autoModels[sensor] = AutoML(results_path="ENSEMBLE AUTOML/AUTOML_"+sensor,
                           mode="Compete",
                           optuna_verbose=False, 
                           algorithms=["Random Forest", "LightGBM", "Extra Trees", "CatBoost", "Xgboost", "Nearest Neighbors"],
                           ml_task='regression',
                           golden_features=False,
                           random_state=32,
                           train_ensemble=True,
                           features_selection=False,
                           kmeans_features=False,
                           validation_strategy={
                                "validation_type": "kfold",
                                "k_folds": 5,
                                "shuffle": True,
                                "stratify": False,
                                "random_seed": 32
                            },
                           eval_metric='rmse')
            self._autoModels[sensor].fit(self._train[[sensor, "Temperatura", "Umidade"]], self._train[self._reference])
            
        self._autoResults = dict()
        self._autoResults["TREINO"] = pd.DataFrame(columns=sensor_list, index=self.metrics)
        self._autoResults["TESTE"] = pd.DataFrame(columns=sensor_list, index=self.metrics)
        self._autoResults["TOTAL"] = pd.DataFrame(columns=sensor_list, index=self.metrics)
        for sensor in sensor_list:
            # self._autoModels[sensor] = AutoML(results_path="AUTOML_"+sensor,random_state=32, train_ensemble=False)#, 
            # self._autoModels[sensor] = RandomForestRegressor(random_state=32) # Cria
            # self._autoModels[sensor].fit(self._train[[sensor, "Temperatura", "Umidade"]].values, self._train[self._reference].values) # Treina

            auto_predict = self._autoModels[sensor].predict(self._train[[sensor, "Temperatura", "Umidade"]]) # Testa com dataset de treino
            self.calcStats(self._autoResults["TREINO"], sensor, auto_predict, self._train[self._reference])

            auto_predict = self._autoModels[sensor].predict(self._test[[sensor, "Temperatura", "Umidade"]]) # Testa com dataset de teste
            self.calcStats(self._autoResults["TESTE"], sensor, auto_predict, self._test[self._reference])
            
            auto_predict = self._autoModels[sensor].predict(self._data[[sensor, "Temperatura", "Umidade"]]) # Testa com dataset inteiro
            self.calcStats(self._autoResults["TOTAL"], sensor, auto_predict, self._data[self._reference])

        if self._debug:
            print("---------------------------AUTO---------------------------")
            print(self._autoResults.head(10))
            # _auto.report()
            # print("Test R^2:", _auto.score(self._test[sensor], self._test[self._reference]))

    def finalizeModels(self, sensor_list):
        for sensor in sensor_list:
            for model_name, model in self._myModels[sensor].items():
                print("Finalizing "+model_name+" for "+sensor)

                if os.path.isfile("lcs/aires/finalized/"+sensor+" "+self._reference+"/"+model_name+".joblib"):
                    print("Found model "+model_name+" for "+sensor)
                    model = joblib.load("lcs/aires/finalized/"+sensor+" "+self._reference+"/"+model_name+".joblib")
                else:
                    if(model_name != "Multiple Linear"):
                        model.set_params(**self._myParams[sensor][model_name])
                    model.fit(self._data[[sensor, "Temperatura", "Umidade"]].values, self._data[self._reference].values)

                predict = model.predict(self._data[[sensor, "Temperatura", "Umidade"]].values)
                self.calcStats(self._results[sensor]["FINAL"][model_name], sensor, predict, self._data[self._reference])

                if self._save:
                    if not os.path.exists("lcs/aires/finalized/"+sensor+" "+self._reference):
                        os.mkdir("lcs/aires/finalized/"+sensor+" "+self._reference, mode=777)
                    joblib.dump(model, "lcs/aires/finalized/"+sensor+" "+self._reference+"/"+model_name+".joblib") # saving model to disk
                    with open("lcs/aires/finalized/"+sensor+" "+self._reference+"/"+model_name+".config", "w+") as f:
                        f.write(str(model.get_params()))
                    # with open(model, "finalized/"+sensor+"/"+model_name+".csv") as f:
                    toSave = pd.DataFrame.copy(self._data[[sensor, "Temperatura", "Umidade", self._reference]], deep=True)
                    toSave[sensor+" CAL"] = predict
                    toSave.to_csv("lcs/aires/finalized/"+sensor+" "+self._reference+"/"+model_name+".csv", sep=";", lineterminator="\n", decimal=",")
        
    def calcStats(self, results_dataframe, sensor, predict, ref):
        results_dataframe.R2 = r2_score(ref, predict)
        results_dataframe.PearsonR = pearsonr(predict, ref)[0]
        results_dataframe.SpearmanR = spearmanr(predict, ref.values).statistic
        results_dataframe.MAE = mae(ref.values, predict)
        results_dataframe.MSE = mse(ref.values, predict)
        results_dataframe.Bias = mbs(ref.values, predict)
        results_dataframe.StdDev = std_deviation(predict)
    
    def buildResultsData(self, sensor_list):
        datasetlist = ["TREINO", "TESTE", "FINAL"] 
        # datasetlist = ["TESTE"]  
        for sensor in sensor_list:
            print("|:--------------------------------------------------------------------:| " + sensor + " |:---------------------------------------------------------------------:|")
            print("")
            for dataset in datasetlist:
                print("|:------------------------------------------------------------------------:| "+dataset+" |:------------------------------------------------------------------------:|")
                print(self._results[sensor][dataset].head(10).to_markdown())
                print("")
    
    def predict(self, x, sensor, model_name):
        return self._myModels[sensor][model_name].predict(x)
    
    def plot(self, sensor_list):
        # i=1
        dataset = "TESTE"
        for sensor in sensor_list:
            fig = plt.figure(sensor[5:].replace(" ", "")+"series")
            ax = plt.subplot(1, 2, 1 if not self._mp10 else 2)
            plt.plot(self._data[self._reference].values, label=self._reference, linewidth=1)
            plt.plot(self._data[sensor].values, label=sensor, linewidth=1)
            best = self._results[sensor][dataset].columns[0]
            for model in self._results[sensor][dataset].columns:
                if self._results[sensor][dataset][best].MSE > self._results[sensor][dataset][model].MSE:
                    best = model

            print(sensor+best)
            if best == 'RF':
                plt.plot(self._rfModels[sensor].predict(self._data[[sensor, "Temperatura", "Umidade"]].values), label=sensor+" RF", linewidth=1)
            elif best == 'LINEAR':
                plt.plot(self._linearModels[sensor].predict(self._data[[sensor, "Temperatura", "Umidade"]]), label=sensor+" LINEAR", linewidth=1)
            elif best == 'GB':
                plt.plot(self._gbModels[sensor].predict(self._data[[sensor, "Temperatura", "Umidade"]].values), label=sensor+" GB", linewidth=1)
            elif best == 'XGB':
                plt.plot(self._xgbModels[sensor].predict(self._data[[sensor, "Temperatura", "Umidade"]].values), label=sensor+" XGB", linewidth=1)
            elif best == 'LGBM':
                plt.plot(self._lgbmModels[sensor].predict(self._data[[sensor, "Temperatura", "Umidade"]].values), label=sensor+" LGBM", linewidth=1)
            elif best == 'ET':
                plt.plot(self._etModels[sensor].predict(self._data[[sensor, "Temperatura", "Umidade"]].values), label=sensor+" ET", linewidth=1)
            elif best == 'CAT':
                plt.plot(self._cbModels[sensor].predict(self._data[[sensor, "Temperatura", "Umidade"]].values), label=sensor+" CAT", linewidth=1)
            elif best == 'AUTO':
                plt.plot(self._autoModels[sensor].predict(self._data[[sensor, "Temperatura", "Umidade"]]), label=sensor+" AUTO", linewidth=1)
            elif best == 'POLY':
                plt.plot(self._polyModels[sensor].predict(self.poly.transform(self._data[[sensor, "Temperatura", "Umidade"]].values)), label=sensor+" POLY", linewidth=1)
            
            x = date2num([datetime.datetime.strptime(str(d), '%d/%m/%Y %H:%M:%S').date() for d in self._data.index])
            aux = []
            passo = int(len(x)/8)
            i = 0
            datas = []
            while i < len(x):
                if int(i)%int(passo) == 0:
                    aux.append(x[i])
                    datas.append(num2date(x[i]).strftime('%d/%m/%y'))
                i = i + 1
            x = aux

            ax.set_xticks(range(0, len(self._data.index), passo))
            ax.set_xticklabels(datas, size=8, rotation=45)
            ax.yaxis.set_tick_params(labelsize=8)
            plt.legend(loc='upper left', fontsize=8)
            # i+=1
    
    def plotStats(self, sensor_list):
        dataset = "TESTE"
        for sensor in sensor_list:
            fig = plt.figure(sensor[5:].replace(" ", "")+"stats")
            ax = plt.subplot(2, 2, 1)
            # self._rfResults["TESTE"][sensor].MAE.values
            a = ax.bar(sensor[:5].replace(" ", "")+" RF", round(self._rfResults[dataset][sensor].MAE, 2), color=newColor[0], label="RF")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" GB", round(self._gbResults[dataset][sensor].MAE, 2), color=newColor[1], label="GB")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" XGB", round(self._xgbResults[dataset][sensor].MAE, 2), color=newColor[2], label="XGB")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" ET", round(self._etResults[dataset][sensor].MAE, 2), color=newColor[3], label="ET")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" CB", round(self._cbResults[dataset][sensor].MAE, 2), color=newColor[4], label="CB")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" LGBM", round(self._lgbmResults[dataset][sensor].MAE, 2), color=newColor[5], label="LGBM")
            ax.bar_label(a, label_type="edge", fontsize=8)
            # a = ax.bar(sensor[:5].replace(" ", "")+" POLY", round(self._polyResults[dataset][sensor].MAE, 2), color=newColor[6], label="POLY")
            # ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" KN", round(self._knResults[dataset][sensor].MAE, 2), color=newColor[6], label="KN")
            ax.bar_label(a, label_type="edge", fontsize=8)
            ax.set_title("MAE")
            ax.yaxis.set_tick_params(labelsize=8)
            # ax.xaxis.set_tick_params(labelsize=0, rotation=45)
            ax.xaxis.set_ticks([])

            ax = plt.subplot(2, 2, 2)
            # self._rfResults[dataset][sensor].MAE.values
            a = ax.bar(sensor[:5].replace(" ", "")+" RF", round(self._rfResults[dataset][sensor].MSE, 2), color=newColor[0], label="RF")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" GB", round(self._gbResults[dataset][sensor].MSE, 2), color=newColor[1], label="GB")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" XGB", round(self._xgbResults[dataset][sensor].MSE, 2), color=newColor[2], label="XGB")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" ET", round(self._etResults[dataset][sensor].MSE, 2), color=newColor[3], label="ET")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" CB", round(self._cbResults[dataset][sensor].MSE, 2), color=newColor[4], label="CB")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" LGBM", round(self._lgbmResults[dataset][sensor].MSE, 2), color=newColor[5], label="LGBM")
            ax.bar_label(a, label_type="edge", fontsize=8)
            # a = ax.bar(sensor[:5].replace(" ", "")+" POLY", round(self._polyResults[dataset][sensor].MSE, 2), color=newColor[6], label="POLY")
            # ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" KN", round(self._knResults[dataset][sensor].MSE, 2), color=newColor[6], label="KN")
            ax.bar_label(a, label_type="edge", fontsize=8)
            ax.set_title("MSE")
            ax.yaxis.set_tick_params(labelsize=8)
            # ax.xaxis.set_tick_params(labelsize=0, rotation=45)
            ax.xaxis.set_ticks([])

            ax = plt.subplot(2, 2, 3)
            # self._rfResults["TESTE"][sensor].MAE.values
            a = ax.bar(sensor[:5].replace(" ", "")+" RF", round(self._rfResults[dataset][sensor].R2, 2), color=newColor[0], label="RF")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" GB", round(self._gbResults[dataset][sensor].R2, 2), color=newColor[1], label="GB")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" XGB", round(self._xgbResults[dataset][sensor].R2, 2), color=newColor[2], label="XGB")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" ET", round(self._etResults[dataset][sensor].R2, 2), color=newColor[3], label="ET")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" CB", round(self._cbResults[dataset][sensor].R2, 2), color=newColor[4], label="CB")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" LGBM", round(self._lgbmResults[dataset][sensor].R2, 2), color=newColor[5], label="LGBM")
            ax.bar_label(a, label_type="edge", fontsize=8)
            # a = ax.bar(sensor[:5].replace(" ", "")+" POLY", round(self._polyResults[dataset][sensor].R2, 2), color=newColor[6], label="POLY")
            # ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" KN", round(self._knResults[dataset][sensor].R2, 2), color=newColor[6], label="KN")
            ax.bar_label(a, label_type="edge", fontsize=8)
            ax.set_title("$R^2$")
            ax.yaxis.set_tick_params(labelsize=8)
            ax.xaxis.set_tick_params(labelsize=8, rotation=45)
            ax.xaxis.grid(False)

            ax = plt.subplot(2, 2, 4)
            # self._rfResults["TESTE"][sensor].MAE.values
            a = ax.bar(sensor[:5].replace(" ", "")+" RF", round(self._rfResults[dataset][sensor].PearsonR, 2), color=newColor[0], label="RF")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" GB", round(self._gbResults[dataset][sensor].PearsonR, 2), color=newColor[1], label="GB")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" XGB", round(self._xgbResults[dataset][sensor].PearsonR, 2), color=newColor[2], label="XGB")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" ET", round(self._etResults[dataset][sensor].PearsonR, 2), color=newColor[3], label="ET")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" CB", round(self._cbResults[dataset][sensor].PearsonR, 2), color=newColor[4], label="CB")
            ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" LGBM", round(self._lgbmResults[dataset][sensor].PearsonR, 2), color=newColor[5], label="LGBM")
            ax.bar_label(a, label_type="edge", fontsize=8)
            # a = ax.bar(sensor[:5].replace(" ", "")+" POLY", round(self._polyResults[dataset][sensor].PearsonR, 2), color=newColor[6], label="POLY")
            # ax.bar_label(a, label_type="edge", fontsize=8)
            a = ax.bar(sensor[:5].replace(" ", "")+" KN", round(self._knResults[dataset][sensor].PearsonR, 2), color=newColor[6], label="KN")
            ax.bar_label(a, label_type="edge", fontsize=8)
            ax.set_title("Pearson R")
            ax.yaxis.set_tick_params(labelsize=8)
            ax.xaxis.set_tick_params(labelsize=8, rotation=45)
            ax.xaxis.grid(False)

            # ax.set_xticklabels("")
            # plt.plot(self._data[self._reference].values, label=self._reference, linewidth=1)
            # plt.plot(self._data[sensor].values, label=sensor, linewidth=1)
            # plt.plot(self._etModels[sensor].predict(self._data[[sensor, "Temperatura", "Umidade"]].values), label=sensor+" ET CALIB", linewidth=1)

            # plt.legend(loc='upper left', fontsize=6)
    
    def plotLinear(self, sensor_list, ref):
        # plt.figure(fig_name, figsize=(10, 7))
        # for i in range(len(list_of_sensors)):
        #     for j in range(len(list_of_sensors)):
        #         line = np.linspace(0, max(self._lcSensorsDataset[list_of_sensors[j]].values),  2)
        #         ax = plt.subplot(len(list_of_sensors), len(list_of_sensors), (j+1)+len(list_of_sensors)*i)
        #         plt.scatter(self._lcSensorsDataset[list_of_sensors[j]].values, self._lcSensorsDataset[list_of_sensors[i]].values, color="black", alpha=0.4, s=3, rasterized=True)
        #         plt.plot(line, line, color="red", linewidth=1)
        #         if j==0:
        #             ax.set_ylabel(list_of_sensors[i], fontsize=8)
        #         if i==len(list_of_sensors)-1:
        #             ax.set_xlabel(list_of_sensors[j], fontsize=8)
        #         r = round(pearsonr(self._lcSensorsDataset[list_of_sensors[j]].values, self._lcSensorsDataset[list_of_sensors[i]].values)[0],2)
        #         at = AnchoredText(f"Pearson R: {r}", prop=dict(size=8), frameon=True, loc='upper left')
        #         at.patch.set_boxstyle("round,pad=0.,rounding_size=0.3")
        #         ax.add_artist(at)
        #         plt.tick_params(axis='x', labelsize=8)
        #         plt.tick_params(axis='y', labelsize=8)
        #         plt.subplots_adjust(wspace=0.24, bottom=0.07, left=0.065, top=0.98, right=0.98)
        for sensor in sensor_list:
            fig = plt.figure(sensor+"LIN")
            line = np.linspace(0, max(self._data[ref].values),  2)
            ax = plt.scatter(self._data[ref].values, self._xgbModels[sensor].predict(self._data[[sensor, "Temperatura", "Umidade"]].values), alpha=0.4, s=18, rasterized=True, label=["as"])
            plt.plot(line, line, color="red", linewidth=1)
            r = round(self._results[sensor]["TOTAL"]["XGB"].PearsonR, 2)
            r2 = round(self._results[sensor]["TOTAL"]["XGB"].R2, 2)
            at = AnchoredText(f"Pearson R: {r}\n$R^2$: {r2}", prop=dict(size=10), frameon=True, loc='upper left')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.3")
            # ax = plt.subplot()
            # ax.axes.add_artist
            ax.axes.add_artist(at)
            ax.axes.set_ylabel(sensor+" $ug/m^3$", fontsize=10)
            ax.axes.set_xlabel(ref+"1020 $ug/m^3$", fontsize=10)
                

if __name__ == "__main__":
    testData = myData(["SDS011a"], "BAM", False)
    testCalib10 = myCalibrationModel(testData, "MP2.5 BAM", False)

    testData = myData(["SDS011b"], "BAM", False)
    testCalib10 = myCalibrationModel(testData, "MP2.5 BAM", False)

    testData = myData(["SDS018"], "BAM", False)
    testCalib10 = myCalibrationModel(testData, "MP2.5 BAM", False)

    testData = myData(["HPMAa"], "BAM", False)
    testCalib10 = myCalibrationModel(testData, "MP2.5 BAM", False)

    testData = myData(["HPMAb"], "BAM", False)
    testCalib10 = myCalibrationModel(testData, "MP2.5 BAM", False)

    testData = myData(["PMS7003a"], "BAM", False)
    testCalib10 = myCalibrationModel(testData, "MP2.5 BAM", False)

    testData = myData(["PMS7003b"], "BAM", False)
    testCalib10 = myCalibrationModel(testData, "MP2.5 BAM", False)

    plt.show()

