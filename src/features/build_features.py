import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

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

df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

fs = int(1000/200) # 5 instances per second
ws = int(2800/200) #2800 milli seconds is avg time

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"],ws,fs )

subset = df_freq[df_freq["set"]==45]
subset[["acc_y"]].plot()
subset[["acc_y_max_freq","acc_y_freq_weighted","acc_y_pse","acc_y_freq_1.429_Hz_ws_14","acc_y_freq_2.5_Hz_ws_14"]].plot()

df_freq_list = []  #Loop over each set 
for s in df_freq["set"].unique():
    print(f"Applying frequency abstraction on set {s}")
    subset = df_freq[df_freq["set"]==s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]  #every 2nd row

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_col = ["acc_x","acc_y","acc_z"]
k_val = range(2,10)
inertia = []

for k in k_val:    #to find the optimal K value using elbow
    subset = df_cluster[cluster_col]
    kmeans = KMeans(n_clusters=k,n_init=20,random_state=0)
    cluster_label = kmeans.fit_predict(subset)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10,10))
plt.plot(k_val,inertia)
plt.xlabel("K")
plt.ylabel("Inertia - Sum of squared distances")
plt.show()

kmeans = KMeans(n_clusters=5,n_init=20,random_state=0)
subset = df_cluster[cluster_col]
df_cluster["cluster"] = kmeans.fit_predict(subset)

#Plot clusters
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"]==c]
    ax.scatter(subset["acc_x"],subset["acc_y"],subset["acc_z"],label=f"Cluster {c}")

ax.set_xlabel("X_axis")
ax.set_ylabel("Y_axis")
ax.set_zlabel("Z_axis")
plt.legend()
plt.show()

#plot accelerometer data to compare
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"]==l]
    ax.scatter(subset["acc_x"],subset["acc_y"],subset["acc_z"],label=f"Label {l}")

ax.set_xlabel("X_axis")
ax.set_ylabel("Y_axis")
ax.set_zlabel("Z_axis")
plt.legend()
plt.show()
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/data_featured.pkl")