import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/data_processed_outliers_removed.pkl")

predictor_columns = list(df.columns[:6])

#plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = [20,5]
plt.rcParams["figure.dpi"] = 200
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
for col in predictor_columns:
    df[col] = df[col].interpolate() #interplote missing values

df.info()
# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"] == 5]["acc_z"].plot()

duration = df[df["set"]==5].index[-1] - df[df["set"]==5].index[0]
duration.seconds

for s in df["set"].unique():
    duration = df[df["set"]==s].index[-1] - df[df["set"]==s].index[0]
    df.loc[df["set"]==s, "duration"] = duration.seconds

df["duration"].unique()

duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0] / 5 #heavy set total duration , instances 
duration_df.iloc[1] / 10 #medium set total duration, instances

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 1000 / 200 # result to 5 instances per second
cutoff = 1.3 # cutoff values depends on us by the graph, the more the value less the smoothness

df_lowpass = LowPass.low_pass_filter(df_lowpass,"acc_y",fs,cutoff,order=5)

subset = df_lowpass[df_lowpass["set"]==45]

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw_data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="lowpass_filtered")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), shadow=True, fancybox=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), shadow=True, fancybox=True)


for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]
# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()  

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns) #PCA values for acc, gyro

plt.figure(figsize=(10,10))  #to find pricipal component number
plt.plot(range(1,len(predictor_columns)+1),pc_values)
plt.xlabel("Principal Component number")
plt.ylabel("Explained variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3) #Dimension reduction to 3

subset = df_pca[df_pca["set"]==45]
subset[["pca_1","pca_2","pca_3"]].plot()


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"]**2 + df_squared["acc_y"]**2 + df_squared["acc_z"]**2
gyro_r = df_squared["gyro_x"]**2 + df_squared["gyro_y"]**2 + df_squared["gyro_z"]**2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyro_r"] = np.sqrt(gyro_r)

subset = df_squared[df_squared["set"]==45]
subset[["acc_r","gyro_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r","gyro_r"] 

ws = int(1000/200) # 1 second time window as our data is 5 instances per second

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")

df_temporal_list = []  #Loop over each set as the above one has drawback 
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"]==s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

subset[["acc_y","acc_y_temp_mean_ws_5","acc_y_temp_std_ws_5"]].plot()
subset[["gyro_y","gyro_y_temp_mean_ws_5","gyro_y_temp_std_ws_5"]].plot()       
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
