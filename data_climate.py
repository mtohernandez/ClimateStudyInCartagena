import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

"""
This is an in-depth analysis of linear regression

Made by: Mateo Hernandez

01/04/2022
"""

# Read the CSV file
data  = pd.read_csv('climatedata.csv', sep=';', encoding='ISO-8859-1')
df = pd.DataFrame(data, columns=['T2M', 'TS', 'PRECTOTCORR', 'QV2M'])

x = df['T2M']
y = df['QV2M']

x_label = 'Temperature 2 meters'
y_label = 'Specific Humidity at 2 meters'


# Linear regression with scipy stats
slope, intercept, r ,p, std_err = stats.linregress(x, y)

def linear_regression(x):
    """
    Returns the model evaluated on the value x
    """
    return slope * x + intercept


def mode_median_mean(v):
    """
    Returns mean, median, mode and mean of a variable
    """
    mode = stats.mode(v)
    median = np.median(v)
    mean = np.mean(v)

    return mode, median, mean


def range_variance_std_cv(v):
    """
    Returns range, variance, standar deviation, coefficient of variation
    """
    range_ = np.max(v) - np.min(v) 
    variance = np.var(v)
    std = np.std(v)
    cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100

    return range_, variance, std, cv(v)


def quantiles_deciles_percentiles(v):
    """
    Returns quatiles 25%, 50% and 75%, deciles 10% to 90% and percentile 65%
    """
    quantiles = np.quantile (v, [0.25,0.5,0.75])
    deciles = np.percentile(v, np.arange(0, 100, 10))
    percentiles = np.percentile(v, 65)
    
    return quantiles, deciles, percentiles

def kurtosis_asymmetry_coef(v):
    """
    Returns Kurtosis and asymmetry coefficient of v
    """
    kurtosis = stats.kurtosis(v)
    asymmetry_coef = stats.skew(v)

    return kurtosis, asymmetry_coef

def normal_anormal_frio(v):
    """
    Returns the count of every number
    """
    count = pd.Series(v).value_counts()

    return count

# Tendencia central
modeX, medianX, meanX = mode_median_mean(x)
modeY, medianY, meanY = mode_median_mean(y)


# Dispersion
range_X, varianceX, stdX, cvX = range_variance_std_cv(x)
range_Y, varianceY, stdY, cvY = range_variance_std_cv(y)

# Location
quatilesX, decilesX, percentilesX = quantiles_deciles_percentiles(x)
quatilesY, decilesY, percentilesY = quantiles_deciles_percentiles(y)

# Frecuencia
kurtosisX, asymmetry_coefX = kurtosis_asymmetry_coef(x)
kurtosisY, asymmetry_coefY = kurtosis_asymmetry_coef(y)


# Linear regresion model
model = list(map(linear_regression,x))


#Graphs (All data)
sns.pairplot(df)


#Graphs (Linear Regression)
fig_rXY = plt.figure(figsize=(10,7))
plt.scatter(x,y)
plt.plot(x,model,"r-")
plt.title('Temperature X Humidity at 2 meters')
plt.xlabel(x_label)
plt.ylabel(y_label)


#Graphs (Boxplots)
fig_bX = plt.figure(figsize=(10,7))
plt.boxplot((x), vert=False)
plt.title('Box Plot X')
plt.xlabel(x_label)

fig_bY = plt.figure(figsize=(10,7))
plt.boxplot((y), vert=False)
plt.title('Box Plot Y')
plt.xlabel(y_label)

#Graphs (Histograms)
fig_hX = plt.figure(figsize=(10,7))
plt.hist(x)
plt.title('Histogram of X')
plt.xlabel(x_label)
plt.ylabel('Frecuencia')

fig_hY = plt.figure(figsize=(10,7))
plt.hist(y)
plt.title('Histrogram of Y')
plt.xlabel(y_label)
plt.ylabel('Frecuencia')

# Q-Q plots
figX, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

sm.qqplot(
    df['T2M'],
    fit   = True,
    line  = 'q',
    alpha = 0.4,
    lw    = 2,
    ax    = axs[0]
)

axs[0].set_title('Gráfico Q-Q temperatura', fontsize = 10, fontweight = "bold")
axs[0].tick_params(labelsize = 7)


figY, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

sm.qqplot(
    df['QV2M'],
    fit   = True,
    line  = 'q',
    alpha = 0.4,
    lw    = 2,
    ax    = axs[0]
)

axs[0].set_title('Gráfico Q-Q humedad', fontsize = 10, fontweight = "bold")
axs[0].tick_params(labelsize = 7)

# Impresion through console
print(f"Impresion de datos X: {x_label} Y:{y_label}")
print(f"\n\nTendencia Central:\n\nX:\nModa: {modeX}\nMediana: {medianX}\nMean: {round(meanX)}\n\nY:\nModa: {modeY}\nMediana: {medianY}\nMean: {round(meanY)}")
print(f"\n\n\nDispersion:\n\nX:\nRango: {range_X}\nVarianza: {varianceX}\nDesviacion Estandar: {stdX}\nCoeficiente de variacion: {cvX}\n\nY:\nRango: {range_Y}\nVarianza: {varianceY}\nDesviacion Estandar: {stdY}\nCoeficiente de variacion: {cvY}")
print(f"\n\n\nLocalizacion:\n\nX:\nQuartiles [25%, 50%, 75%]: {quatilesX}\nDeciles [10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%]: {decilesX}\nPercentil 65%: {percentilesX}\n\nY:\nQuartiles [25%, 50%, 75%]: {quatilesY}\nDeciles [10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%]: {decilesY}\nPercentil 65%: {percentilesY}")
print(f"\n\n\nFrecuencia:\n\nX:\nCurtosis: {kurtosisX}\nCoeficiente de asimetria: {asymmetry_coefX}\n\nY:\nCurtosis: {kurtosisY}\nCoeficiente de asimetria: {asymmetry_coefY}")
print(f"\n\n\nRegresion Lineal:\nPendiente: {slope}\nIntercepto: {intercept}\nCoeficiente de correlacion: {r}\nP-value: {p}\nDesviacion Estandar Error: {std_err}\nValor R2: {r2_score(y, linear_regression(x))}")

print(f"Mean Squared error: {mean_squared_error(y, model)}")

last_fg = plt.figure(figsize=(10,7))
plt.scatter(x,y)
plt.scatter(x, model, "r", alpha=0.4)


plt.show()