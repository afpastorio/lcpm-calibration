import pandas as pd

temp_umi = pd.read_csv("SDS.csv", header=0, delimiter=';', usecols=[0,1,2], index_col=0, parse_dates=True, dayfirst=True)
# temp_umi = temp_umi.drop(temp_umi[temp_umi["Umidade"]>95].index)

sds011a = pd.read_csv("SDS.csv", header=0, delimiter=';', usecols=[0,3,4], index_col=0, parse_dates=True, dayfirst=True)
sds011a = temp_umi.join(sds011a)
# sds011a = sds011a.mask(sds011a<0)
# sds011a = sds011a.dropna(how='any', axis=0)

sds011b = pd.read_csv("SDS.csv", header=0, delimiter=';', usecols=[0,5,6], index_col=0, parse_dates=True, dayfirst=True)
sds011b = temp_umi.join(sds011b)
# sds011b = sds011b.mask(sds011b<0)
# sds011b = sds011b.dropna(how='any', axis=0)

sds018 = pd.read_csv("SDS.csv", header=0, delimiter=';', usecols=[0,7,8], index_col=0, parse_dates=True, dayfirst=True)
sds018 = temp_umi.join(sds018)
# sds018 = sds018.mask(sds018<0)
# sds018 = sds018.dropna(how='any', axis=0)

opc = pd.read_csv("SDS.csv", header=0, delimiter=';', usecols=[0,9,10], index_col=0, parse_dates=True, dayfirst=True)
opc = temp_umi.join(opc)
# opc = opc.mask(opc<0)
# opc = opc.dropna(how='any', axis=0)


hpmaa = pd.read_csv("HPMAa.csv", header=0, delimiter=';', index_col=0, parse_dates=True, dayfirst=True)
hpmaa = temp_umi.join(hpmaa)
# hpmaa = hpmaa.mask(hpmaa<0)
hpmaa = hpmaa.dropna(how='any', axis=0)

hpmab = pd.read_csv("HPMAb.csv", header=0, delimiter=';', index_col=0, parse_dates=True, dayfirst=True)
hpmab = temp_umi.join(hpmab)
# print(hpmab.describe())
# hpmab = hpmab.mask(hpmab<0)
hpmab = hpmab.dropna(how='any', axis=0)

pmsa = pd.read_csv("PMS.csv", header=0, delimiter=';', usecols=[0,1,2], index_col=0, parse_dates=True, dayfirst=True)
pmsa = temp_umi.join(pmsa)
# pmsa = pmsa.mask(pmsa<0)
pmsa = pmsa.dropna(how='any', axis=0)

pmsb = pd.read_csv("PMS.csv", header=0, delimiter=';', usecols=[0,3,4], index_col=0, parse_dates=True, dayfirst=True)
pmsb = temp_umi.join(pmsb)
# pmsb = pmsb.mask(pmsb<0)
pmsb = pmsb.dropna(how='any', axis=0)

sds011a.to_csv("SDS011a.csv", sep=";", lineterminator="\n")
sds011b.to_csv("SDS011b.csv", sep=";", lineterminator="\n")
sds018.to_csv("SDS018.csv", sep=";", lineterminator="\n")
hpmaa.to_csv("HPMAa2.csv", sep=";", lineterminator="\n")
hpmab.to_csv("HPMAb2.csv", sep=";", lineterminator="\n")
pmsa.to_csv("PMS7003a.csv", sep=";", lineterminator="\n")
pmsb.to_csv("PMS7003b.csv", sep=";", lineterminator="\n")
opc.to_csv("OPC.csv", sep=";", lineterminator="\n")
# print(sds011a.describe())