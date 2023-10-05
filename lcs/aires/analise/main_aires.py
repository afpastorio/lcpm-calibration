# Manipulação de dados
import pandas as pd
import numpy as np
# from load_data import remove_outlier_iqr as remove_outlier
# Gráficos
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler
# print(plt.style.available)
plt.style.use('seaborn-v0_8-whitegrid')
myColors = [(0.00784313725490196, 0.4470588235294118, 0.6352941176470588), (0.6235294117647059, 0.7647058823529411, 0.4666666666666667), (0.792156862745098, 0.043137254901960784, 0.011764705882352941), (0.6470588235294118, 0.00784313725490196, 0.34509803921568627), (0.8431372549019608, 0.7803921568627451, 0.011764705882352941), (0.5333333333333333, 0.792156862745098, 0.8549019607843137)]

mc = LinearSegmentedColormap.from_list('', myColors)
newColor = mc(np.linspace(0, 1, 7))
# print( )
custom = cycler(color=myColors)
plt.rc('axes', prop_cycle=custom)
# print(matplotlib.rcParams["axes.prop_cycle"])
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
# Métricas
from scipy.stats import pearsonr
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
# Utilidades
from random import randrange
# from datetime import datetime
# custom_date_parser = lambda x: datetime.strptime(x, "%H:%M %m/%d/%Y")
# Algoritmos de calibração
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import datetime
# Modelos com melhores métricas, tanto no pycaret quanto no lazy: RF, XGBoost, GB, ExtraTrees, KNeighbors, Catboost, LightGBM
# DIA 27 as 11hrs mudei o a leitura do HPMAb pra ver se melhora as respostas

class myData:
    _save = False
    # _include_yt = True
    def __init__(self, lc_list, gm_list=None, yt=False):
        print("------------ANTES------------")
        self.loadDataLCS(lc_list)
        self.loadDataREF()
        self.columns25 = []
        self.columns10 = []
        self.sensor_list = lc_list

        print("------------TIRANDO NULOS------------")
        self._lcSensorsDataset = self._lcSensorsDataset.mask(self._lcSensorsDataset<0)
        self._lcSensorsDataset.dropna(inplace=True, how='any', axis=0)
        print(self._lcSensorsDataset.describe())

        print("------------TIRANDO TEMP e UMI FALHA------------")
        self._lcSensorsDataset = self._lcSensorsDataset = self._lcSensorsDataset.drop(self._lcSensorsDataset[self._lcSensorsDataset["Umidade"]>100].index) #Falha
        # self._lcSensorsDataset.dropna(inplace=True, how='any', axis=0)
        print(self._lcSensorsDataset.describe())

        print("------------TIRANDO UMI LIMITE SDS 70%------------")
        self._lcSensorsDataset = self._lcSensorsDataset = self._lcSensorsDataset.drop(self._lcSensorsDataset[self._lcSensorsDataset["Umidade"]>95].index) #Falha
        # self._lcSensorsDataset.dropna(inplace=True, how='any', axis=0)
        print(self._lcSensorsDataset.describe())

        print("------------REFERENCIA TIRANDO ERROS------------")
        self._refSensorsDataset = self._refSensorsDataset.mask(self._refSensorsDataset<0)
        self._refSensorsDataset = self._refSensorsDataset.mask(self._refSensorsDataset>370)
        self._refSensorsDataset = self._refSensorsDataset.dropna(how='any', axis=0)
        print(self._refSensorsDataset.describe())

        print("------------MEDIA HORARIA------------")
        self._lcSensorsDataset = self._lcSensorsDataset.resample('H').mean()
        print(self._lcSensorsDataset.describe())

        print("------------JOIN------------")
        # self._lcSensorsDataset = self._lcSensorsDataset.join(self._refSensorsDataset[["MP2.5 BAM", "MP10 BAM"]])
        self._allData = self._lcSensorsDataset.join(self._refSensorsDataset[["MP2.5 BAM", "MP10 BAM"]])
        self._allData.dropna(inplace=True, how='any', axis=0)
        print(self._allData.describe())
        
        for sensor in lc_list:
            self.columns25.append("MP2.5 "+sensor)
            self.columns10.append("MP10 "+sensor)

        metrics = ["R2", "PearsonR", "MAE", "MAPE", "MSE", "MSLE", "Precision", "MPP", "Bias", "StdDev"]


        # self._allData = self._allData.dropna(how='any', axis=0)

        self._rawData25Scores = pd.DataFrame(columns=self.columns25, index=metrics)
        
        self._rawData10Scores = pd.DataFrame(columns=self.columns10, index=metrics)
        # self.calcStatsData()
    def calcStatsData(self):
        _ref25 = "MP2.5 BAM"
        sensor_list = self.columns25

        for sensor in sensor_list:
            self._rawData25Scores[sensor].R2 = r2_score(self._allData[_ref25], self._allData[sensor])
            self._rawData25Scores[sensor].PearsonR = pearsonr(self._allData[sensor], self._allData[_ref25])[0]
            self._rawData25Scores[sensor].MAE = mae(self._allData[_ref25].values, self._allData[sensor].values)
            # self._rawData25Scores[sensor].MAPE = mape(self._allData[_ref25].values, self._allData[sensor].values)
            self._rawData25Scores[sensor].MSE = mse(self._allData[_ref25].values, self._allData[sensor].values)
            self._rawData25Scores[sensor].MSLE = msle(self._allData[_ref25].values, self._allData[sensor].values)
            # self._rawData25Scores[sensor].Precision = precision(self._allData[sensor].values)
            # self._rawData25Scores[sensor].MPP = mpp(self._allData[_ref25])
            self._rawData25Scores[sensor].Bias = mbs(self._allData[_ref25].values, self._allData[sensor].values)
            self._rawData25Scores[sensor].StdDev = self._allData[sensor].std(ddof=0)

        _ref10 = "MP10 BAM"
        sensor_list = self.columns10

        for sensor in sensor_list:
            self._rawData10Scores[sensor].R2 = r2_score(self._allData[_ref10], self._allData[sensor])
            self._rawData10Scores[sensor].PearsonR = pearsonr(self._allData[sensor], self._allData[_ref10])[0]
            self._rawData10Scores[sensor].MAE = mae(self._allData[_ref10].values, self._allData[sensor].values)
            # self._rawData10Scores[sensor].MAPE = mape(self._allData[_ref10].values, self._allData[sensor].values)
            self._rawData10Scores[sensor].MSE = mse(self._allData[_ref10].values, self._allData[sensor].values)
            self._rawData10Scores[sensor].MSLE = msle(self._allData[_ref10].values, self._allData[sensor].values)
            # self._rawData10Scores[sensor].Precision = precision(self._allData[sensor].values)
            # self._rawData10Scores[sensor].MPP = mpp(self._allData[_ref10])
            self._rawData10Scores[sensor].Bias = mbs(self._allData[_ref10].values, self._allData[sensor].values)
            self._rawData10Scores[sensor].StdDev = self._allData[sensor].std(ddof=0)
        print("---------------------------RAW---------------------------")
        print(self._rawData25Scores.head(10))
        print(self._rawData10Scores.head(10))

    def loadDataLCS(self, lc_list):
        self._lcSensorsDataset = pd.read_csv(lc_list[0]+".csv", header=0, delimiter=';', parse_dates=True, index_col=0, dayfirst=True)
        for index in range(1,len(lc_list)):
            self._lcSensorsDataset = self._lcSensorsDataset.join(pd.read_csv(lc_list[index]+".csv", header=0, delimiter=';', parse_dates=True, index_col=0, dayfirst=True, usecols=[0,3,4]))
        # print(self._lcSensorsDataset.head(10))
        # self._lcSensorsDataset = self._lcSensorsDataset.drop(self._lcSensorsDataset[self._lcSensorsDataset["Umidade"]>95].index)
        # self._lcSensorsDataset = self._lcSensorsDataset.mask(self._lcSensorsDataset<0)

        # self._lcSensorsDataset = self._lcSensorsDataset.resample('H').mean() # H por hora é o correto
        # self._lcSensorsDataset.index = pd.to_datetime(self._lcSensorsDataset.index, format='%d/%m/%Y %H:%M:%S').strftime('%d/%m/%Y %H:%M:%S')
        # self._lcSensorsDataset = self._lcSensorsDataset.dropna(how='any', axis=0)
        # if "OPC" in lc_list:
        #     self._lcSensorsDataset[["MP2.5 OPC", "MP10 OPC"]] = self._lcSensorsDataset[["MP2.5 OPC", "MP10 OPC"]].multiply(100) # Multiplicando OPC por 100 fica legal, pode ser algum fator de calibração ja setado
        print(self._lcSensorsDataset.describe())

    def loadYong(self):
        self._yong = pd.read_csv("lcs/aires/YONG.csv", header=0, delimiter=';', parse_dates=True, index_col=0, dayfirst=True, usecols=[0,1,2,3,4,5])
        self._yong = self._yong.mask(self._yong>1000)
        self._yong = self._yong.resample('H').mean()
        self._yong.index = pd.to_datetime(self._yong.index, format='%d/%m/%Y %H:%M:%S').strftime('%d/%m/%Y %H:%M:%S')
        self._yong = self._yong.dropna(how='any', axis=0)

    def loadDataREF(self):
        self._refSensorsDataset = pd.read_csv("BAM.csv", header=0, delimiter=';', parse_dates=True, index_col=0, dayfirst=False, usecols=[0,1,2], na_values='.')
        # self._refSensorsDataset = self._refSensorsDataset.dropna(how='any', axis=0)
        self._refSensorsDataset[["MP2.5 BAM", "MP10 BAM"]] = self._refSensorsDataset[["MP2.5 BAM", "MP10 BAM"]].apply(pd.to_numeric, errors='coerce')
        self._refSensorsDataset.index = pd.to_datetime(self._refSensorsDataset.index, format='%H:%M %m/%d/%Y')
        self._refSensorsDataset.index -= pd.Timedelta('1H')
        # self._refSensorsDataset.index = pd.to_datetime(self._refSensorsDataset.index, format='%H:%M %m/%d/%Y').strftime('%d/%m/%Y %H:%M:%S')
        # print(self._refSensorsDataset.head(10))
        # self._refSensorsDataset = self._refSensorsDataset.mask(self._refSensorsDataset<0)
        # self._refSensorsDataset = self._refSensorsDataset.mask(self._refSensorsDataset>370)
        # self._refSensorsDataset = self._refSensorsDataset.dropna(how='any', axis=0)
        print(self._refSensorsDataset.describe())
    
    def loadGM(self, num):
        self._gm[num] = pd.read_csv("lcs/aires/GM"+num+".csv", delimiter=',', index_col=0, parse_dates=True, dayfirst=False, squeeze=0, usecols=[0,26,28,48,50])
        self._gm[num].columns = ["MP2.5C GM"+num, "MP10C GM"+num, "MP2.5 GM"+num, "MP10 GM"+num]
        self._gm[num].index.name = "Data/Hora"
        self._gm[num].sort_index(inplace=True)
        self._gm[num]["MP2.5 GM"+num] = self._gm[num]["MP2.5 GM"+num].apply(pd.to_numeric, errors='coerce')
        # print(self._gm[num].head(10))
        # print(np.argmin(self._gm[num].applymap(np.isreal).all(1)))
        self._gm[num] = self._gm[num].resample('H').mean()

        self._gm[num].index = pd.to_datetime(self._gm[num].index, format='%Y-%m-%d %H:%M:%S').strftime('%d/%m/%Y %H:%M:%S')
        # self._gm[num]= self._gm[num].join(self._refSensorsDataset[['MP2.5 BAM', 'MP10 BAM']])
        self._gm[num] = self._gm[num].dropna(how='any', axis=0)
        # print(self._gm[num].head(10))

    def plotRefLinear(self, fig_name, list_of_sensors, particle="MP2.5"):
        _bam = particle+" BAM"
        plt.figure(fig_name, figsize=(10, 4))
        # for i in range(len(list_of_sensors)):
        for j in range(len(list_of_sensors)):
            line = np.linspace(0, max(self._allData[list_of_sensors[j]].values),  2)
            ax = plt.subplot(1, len(list_of_sensors), j+1)
            plt.scatter(self._allData[list_of_sensors[j]].values, self._allData[_bam].values, color="black", alpha=0.4, s=3, rasterized=True)
            plt.plot(line, line, color="red", linewidth=1)
            if j==0:
                ax.set_ylabel(_bam, fontsize=8)
            # if i==len(list_of_sensors)-1:
            ax.set_xlabel(list_of_sensors[j], fontsize=8)
            r = round(pearsonr(self._allData[list_of_sensors[j]].values, self._allData[_bam].values)[0],2)
            at = AnchoredText(f"Pearson R: {r}", prop=dict(size=8), frameon=True, loc='upper left')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.3")
            ax.add_artist(at)
            plt.tick_params(axis='x', labelsize=8)
            plt.tick_params(axis='y', labelsize=8)
            plt.subplots_adjust(wspace=0.24, bottom=0.07, left=0.065, top=0.98, right=0.98)
        if self._save:
            plt.savefig('sensor_comp/%s.svg'%(fig_name), dpi=300, format='svg')
        return
    
    def plotSensorLinear(self, fig_name, list_of_sensors=None):
        if list_of_sensors == None:
            list_of_sensors = self.columns25 + self.columns10
        plt.figure(fig_name, figsize=(10, 7))
        for i in range(len(list_of_sensors)):
            for j in range(len(list_of_sensors)):
                line = np.linspace(0, max(self._lcSensorsDataset[list_of_sensors[j]].values),  2)
                ax = plt.subplot(len(list_of_sensors), len(list_of_sensors), (j+1)+len(list_of_sensors)*i)
                plt.scatter(self._lcSensorsDataset[list_of_sensors[j]].values, self._lcSensorsDataset[list_of_sensors[i]].values, color="black", alpha=0.4, s=3, rasterized=True)
                plt.plot(line, line, color="red", linewidth=1)
                if j==0:
                    ax.set_ylabel(list_of_sensors[i], fontsize=8)
                if i==len(list_of_sensors)-1:
                    ax.set_xlabel(list_of_sensors[j], fontsize=8)
                r = round(pearsonr(self._lcSensorsDataset[list_of_sensors[j]].values, self._lcSensorsDataset[list_of_sensors[i]].values)[0],2)
                at = AnchoredText(f"Pearson R: {r}", prop=dict(size=8), frameon=True, loc='upper left')
                at.patch.set_boxstyle("round,pad=0.,rounding_size=0.3")
                ax.add_artist(at)
                plt.tick_params(axis='x', labelsize=8)
                plt.tick_params(axis='y', labelsize=8)
                plt.subplots_adjust(wspace=0.24, bottom=0.07, left=0.065, top=0.98, right=0.98)
        if self._save:
            plt.savefig('sensor_comp/%s.svg'%(fig_name), dpi=300, format='svg')
        return
    
    def plotStats(self, fig_num, sensor_list=None):
        if sensor_list == None:
            sensor_list = self.columns25 + self.columns10
        plt.figure(fig_num, figsize=(8.5, 3.5))
        name = ""
        if sensor_list[0][:5] == "MP2.5":
            name = "MP2.5 $(ug/m^3)$"
        else:
            name = "MP10 $(ug/m^3)$"
        # self._allData.plot.box()
        ax1 = plt.subplot(111)
        # # sns.catplot(data=self._lcSensorsDataset[sensor_list], kind='box')
        if "MP10 BAM" in sensor_list or "MP2.5 BAM" in sensor_list:
            self._allData[sensor_list].plot.box(ax=ax1)
        else:
            self._lcSensorsDataset[sensor_list].plot.box(ax=ax1)
        ax1.set_ylabel(name)
        # ax1.set_xlabel(sensor_list)
        # ax1.set_xticks([])
        # ax2 = plt.subplot(122)
        # self._lcSensorsDataset[sensor_list].plot.hist(ax=ax2, alpha=0.3)
        # ax3 = ax2.twinx()
        # self._lcSensorsDataset[sensor_list].plot.kde(ax= ax3)
        # ax2.set_ylabel('Frequência')
        # ax3.set_ylabel('Densidade')
        # ax2.set_xlabel('(b)')
        # plt.subplots_adjust(wspace=0.3, bottom=0.14)
        return
    
    def plotResiduals(self, fig_num, sensor=None):
        # if sensor_list == None:
        #     sensor_list = self.columns25 + self.columns10
        name = ""

        if sensor[:5] == "MP2.5":
            name = "MP2.5 BAM"
        else:
            name = "MP10 BAM"
        # teste = "MP2.5 PMS7003a"
        # for sensor in sensor_list:
        #     ref = ""
        #     if sensor[:5] == "MP2.5":
        #         ref = "MP2.5 BAM"
        #     else:
        #         ref = "MP10 BAM"
        self._allData["Erro"] = (self._allData[name] - self._allData[sensor])
        # print(self._allData.tail(50))
        # self._allData["residual"] = (self._allData["residual"]/self._allData[name])
        # self._allData["residual"] = (self._allData["residual"] - self._allData["residual"].mean())/self._allData["residual"].std()
        # print(self._allData["residual"].mean())
        line = [
            min(self._allData[sensor].values),
            max(self._allData[sensor].values)
        ]
        # # self._allData.plot.hist(y="residual", x=teste)
        ax = self._allData.plot.scatter(y="Erro", x=sensor, color="black", alpha=0.4, s=3, rasterized=True)
        ax.plot(line,[0,0], color="red", linewidth=1, ls='--')

        # line = [
        #     min(self._allData["MP2.5 SDS011a"].values),
        #     max(self._allData["MP2.5 SDS011a"].values)
        # ]
        # # # self._allData.plot.hist(y="residual", x=teste)
        # ax = self._allData.plot.scatter(y="residualMP2.5 BAMMP2.5 SDS011a", x="MP2.5 SDS011a", color="black", alpha=0.4, s=3, rasterized=True)
        # ax.plot(line,[0,0], color="red", linewidth=1, ls='--')
        # plt.scatter(x=self._refSensorsDataset[name], y=self.resid["MP2.5 OPC"])
        # sns.residplot(x="MP2.5 SDS011a", y="MP10 BAM", data=self._allData, lowess=False)
        # for sensor in sensor_list:
        #     sb.residplot(x=sensor, y=name, data=self._allData)
    def plotAll(self, fig_name, list_of_sensors=None):
        if list_of_sensors == None:
            list_of_sensors = self.columns25 + self.columns10
        self._allData[list_of_sensors].plot(linewidth=1)

    def plotGM(self, fig_name, num):
        self._gm[num].plot()
    def toCsv(self, file_name):
        self._lcSensorsDataset.describe().to_csv("lcs/aires/"+file_name+"descript.csv",decimal=",", sep=";")
        self._lcSensorsDataset.to_csv("lcs/aires/"+file_name+"data.csv", decimal=",", sep=";")
        return
if __name__ == "__main__":
    test = myData(["OPC"])
    # test.plotResiduals(0, "MP2.5 SDS011a")
    # test.plotAll(1,["MP2.5 SDS011a", "MP2.5 BAM"])
    # test.plotSensorLinear(0)
    # test.plotStats(1)
    # test.plotResiduals(0, ["MP2.5 SDS011a", "MP2.5 SDS011b", "MP2.5 SDS018", "MP2.5 HPMAa", "MP2.5 HPMAb", "MP2.5 PMS7003a", "MP2.5 PMS7003b", "MP2.5 OPC"])
    # test.plotStats(0, ["MP2.5 SDS011a", "MP2.5 SDS011b", "MP2.5 SDS018", "MP2.5 HPMAa", "MP2.5 HPMAb", "MP2.5 PMS7003a", "MP2.5 PMS7003b", "MP2.5 OPC", "MP2.5 YT"])
    # test.plotStats(1, ["MP10 SDS011a", "MP10 SDS011b", "MP10 SDS018", "MP10 HPMAa", "MP10 HPMAb", "MP10 PMS7003a", "MP10 PMS7003b", "MP10 OPC", "MP10 YT"])
    # test.plotStats(2, ["MP2.5 SDS011a", "MP2.5 SDS011b", "MP2.5 SDS018", "MP2.5 HPMAa", "MP2.5 HPMAb", "MP2.5 PMS7003a", "MP2.5 PMS7003b", "MP2.5 OPC", "MP2.5 YT", "MP2.5 BAM"])
    # test.plotStats(3, ["MP10 SDS011a", "MP10 SDS011b", "MP10 SDS018", "MP10 HPMAa", "MP10 HPMAb", "MP10 PMS7003a", "MP10 PMS7003b", "MP10 OPC", "MP10 YT", "MP10 BAM"])
    # test.plotRefLinear(0, ["MP2.5 SDS011a", "MP2.5 SDS011b", "MP2.5 SDS018", "MP2.5 HPMAa", "MP2.5 HPMAb", "MP2.5 PMS7003a", "MP2.5 PMS7003b", "MP2.5 OPC", "MP2.5 YT"], "MP2.5")
    # test.plotRefLinear(1,  ["MP10 SDS011a", "MP10 SDS011b", "MP10 SDS018", "MP10 HPMAa", "MP10 HPMAb", "MP10 PMS7003a", "MP10 PMS7003b", "MP10 OPC", "MP10 YT"], "MP10")
    # test.plotAll(1,  ["MP10 SDS011a", "MP10 SDS011b", "MP10 SDS018", "MP10 HPMAa", "MP10 HPMAb", "MP10 PMS7003a", "MP10 PMS7003b", "MP10 OPC", "MP10 YT", "MP10 BAM"])
    # test.plotAll(2,  ["MP10 YT", "MP10 BAM"])
    # test.plotSensorLinear(6, ["MP2.5 YT", "MP10 YT", "PTS YT"])
    # test.plotAll(5,  ["MP2.5 YT", "MP2.5 BAM"])
    # test.plotRefLinear(3,  ["MP2.5 YT", "MP10 YT"])
    # test.plotRefLinear(4,  ["MP2.5 YT", "MP10 YT"], "MP10")
    # test.plotRefLinear25(0, ["MP2.5 SDS011a", "MP2.5 SDS011b", "MP2.5 SDS018", "MP2.5 HPMAa", "MP2.5 HPMAb", "MP2.5 PMS7003a", "MP2.5 PMS7003b", "MP2.5 OPC"])
    # test.plotRefLinear10(1,  ["MP10 SDS011a", "MP10 SDS011b", "MP10 SDS018", "MP10 HPMAa", "MP10 HPMAb", "MP10 PMS7003a", "MP10 PMS7003b", "MP10 OPC"])
    # test.plotAll(4,  ["MP10 SDS011a", "MP10 SDS011b", "MP10 SDS018", "MP10 HPMAa", "MP10 HPMAb", "MP10 OPC", "MP10 BAM"])
    # test.plotAll(5,  ["MP2.5 SDS011a", "MP2.5 SDS011b", "MP2.5 SDS018", "MP2.5 HPMAa", "MP2.5 HPMAb", "MP2.5 OPC", "MP2.5 BAM"])
    # test.plotStats(1,  ["MP10 SDS011a", "MP10 SDS011b", "MP10 SDS018", "MP10 HPMAa", "MP10 HPMAb", "MP10 PMS7003a", "MP10 PMS7003b", "MP10 OPC", "MP10 YT"])
    # test.plotSensorLinear(0, ["MP2.5 SDS011a", "MP10 SDS011a", "MP2.5 SDS011b", "MP10 SDS011b", "MP2.5 SDS018", "MP10 SDS018"])
    # test.plotSensorLinear(1, ["MP2.5 HPMAa", "MP10 HPMAa", "MP2.5 HPMAb", "MP10 HPMAb"])
    # test.plotSensorLinear(2, ["MP2.5 PMS7003a", "MP10 PMS7003a", "MP2.5 PMS7003b", "MP10 PMS7003b"])
    # test.plotSensorLinear(3, ["MP2.5 OPC", "MP10 OPC"])
    # test.plotAll(4, ["MP2.5 GM010", "MP2.5C GM010", "MP2.5 BAM"])
    # test.plotAll(5, ["MP2.5 GM011", "MP2.5C GM011", "MP2.5 BAM"])
    # test.plotAll(6, ["MP2.5 GM012", "MP2.5C GM012", "MP2.5 BAM"])
    # test.plotAll(7, ["MP2.5 GM022", "MP2.5C GM022", "MP2.5 BAM"])

    # test.plotAll(4, ["MP10 GM010", "MP10C GM010", "MP10 BAM"])
    # test.plotAll(5, ["MP10 GM011", "MP10C GM011", "MP10 BAM"])
    # test.plotAll(6, ["MP10 GM012", "MP10C GM012", "MP10 BAM"])
    # test.plotAll(7, ["MP10 GM022", "MP10C GM022", "MP10 BAM"])

    # test.plotAll(4, ["MP2.5 SDS011a", "MP2.5 SDS011b", "MP2.5 SDS018", "MP2.5 HPMAa", "MP2.5 PMS7003a", "MP2.5 PMS7003b", "MP2.5 OPC", "MP2.5 YT", "MP2.5C GM010", "MP2.5C GM011", "MP2.5C GM012", "MP2.5 BAM"])
    # test.plotSensorLinear(5, ["MP10 SDS011a", "MP10 SDS011b", "MP10 SDS018", "MP10 HPMAa", "MP10 PMS7003a", "MP10 PMS7003b", "MP10 OPC", "MP10 YT", "MP10 GM010", "MP10 GM011", "MP10 GM012"])
    plt.show()