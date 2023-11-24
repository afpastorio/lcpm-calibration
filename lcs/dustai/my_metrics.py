from scipy.stats import sem as std_error
from scipy.stats import tstd as std_deviation

# Calcula valor de precisão. Precisão pode ser dada como um valor acima ou abaixo da medida real, ou da média
def precision(y):
    return std_deviation(y)/(sum(y)/len(y))

# Calcula a porcentagem de precisão com relação a média da amostra
def mean_percent_precision(y):
    return 1 - precision(y)/(sum(y)/len(y))

# Calcula o bias error entre duas amostras de mesmo tamanho
def mean_bias_error(y_true, y_pred):
    accum = 0
    for i in range(len(y_true)):
        accum+=(y_pred[i]-y_true[i])

    return accum/len(y_pred)

# Calcula o erro de bias de acordo com o documento da EPA
def bias_error(y_true, y_pred):
    if (sum(y_true)/len(y_true)) > 0:
        return (sum(y_pred)/len(y_pred)) / (sum(y_true)/len(y_true)) - 1
    
if __name__ == "__main__":
    print(mean_bias_error([6, 138, 278, 411, 545], [0, 139, 278, 417, 556]))
    print(bias_error([6, 138, 278, 411, 545], [0, 139, 278, 417, 556]))
    print(precision([1, 1, 1, 1, 1, 1, 1, 1, 1, 1.2]))
    print(mean_percent_precision([1, 1, 1, 1, 1, 1, 1, 1, 1, 1.2]))