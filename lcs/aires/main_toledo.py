import pandas as pd

# all_data = pd.read_csv("dados-10-05.csv", header=0, delimiter=';', usecols=[0,1,2,4,6,8,11,14], index_col=0, parse_dates=True, dayfirst=True)

all_data = pd.read_csv("TEOM.csv", header=0, delimiter=',', usecols=[0,1], index_col=0, parse_dates=True, dayfirst=True)

all_data.to_csv("toledo/TEOM.csv" ,sep=";", lineterminator="\n")

# sds011 = all_data[["Temperatura", "Umidade", "MP10 SDS011"]].copy(deep=True)

# sds018 = all_data[["Temperatura", "Umidade", "MP10 SDS018"]].copy(deep=True)

# hpma = all_data[["Temperatura", "Umidade", "MP10 HPMA"]].copy(deep=True)

# pms = all_data[["Temperatura", "Umidade", "MP10 PMS7003"]].copy(deep=True)

# mix = all_data[["Temperatura", "Umidade", "MP10 MIX6070"]].copy(deep=True)

# sds011.to_csv("toledo/SDS011.csv" ,sep=";", lineterminator="\n")
# sds018.to_csv("toledo/SDS018.csv" ,sep=";", lineterminator="\n")
# hpma.to_csv("toledo/HPMA.csv" ,sep=";", lineterminator="\n")
# pms.to_csv("toledo/PMS7003.csv" ,sep=";", lineterminator="\n")
# mix.to_csv("toledo/MIX6070.csv" ,sep=";", lineterminator="\n")