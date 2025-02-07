import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

single_df_road = pd.read_excel("../motiv/data/road/Ride_652.xlsx")

single_df_rail = pd.read_excel("../motiv/data/rail/sanjay_143.xlsx")

df_road = single_df_road.dropna(subset=['latitude','longitude'])
df_road.reset_index(drop=True,inplace=True)

#combine road dataframe

folder_path = os.path.normpath("../motiv/data/road/*xlsx")
files = glob.glob(folder_path)

rides_summary = []

for file in files:
    df = pd.read_excel(file)
    
    
    ride_id = int(os.path.basename(file).split("_")[1].split(".")[0])  
    
    ride_stats = {
        "ride_id": ride_id,
        "max_speed": df["speed"].max(),
        "min_speed": df["speed"].min(),
        "avg_speed": df["speed"].mean(),
        "std_speed": df["speed"].std()
    }

    rides_summary.append(ride_stats)

road_ride_df = pd.DataFrame(rides_summary)

#visualize road ride avg_speed and std_speed

plt.figure(figsize=(20,5))

plt.subplot(2, 1, 1)
sns.lineplot(x=road_ride_df["ride_id"], y=road_ride_df["avg_speed"], marker="o", color="b", label="Avg Speed")
plt.ylabel("Avg Speed")
plt.title("Average Speed per Ride")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(2, 1, 2) 
sns.lineplot(x=road_ride_df["ride_id"], y=road_ride_df["std_speed"], marker="s", color="r", label="Std Dev of Speed")
plt.ylabel("Std Dev of Speed")
plt.xlabel("Ride ID")
plt.title("Standard Deviation of Speed per Ride")
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()

plt.show()

#combine rail dataframe

single_df_rail = pd.read_excel("../motiv/data/rail/sanjay_143.xlsx")

folder_path = os.path.normpath("../motiv/data/rail/*xlsx")
files = glob.glob(folder_path)

rides_summary = []

for file in files:
    df = pd.read_excel(file)
    
    
    ride_id = int(os.path.basename(file).split("_")[1].split(".")[0])  
    
    ride_stats = {
        "ride_id": ride_id,
        "max_speed": df["speed"].max(),
        "min_speed": df["speed"].min(),
        "avg_speed": df["speed"].mean(),
        "std_speed": df["speed"].std()
    }

    rides_summary.append(ride_stats)

rail_ride_df = pd.DataFrame(rides_summary)


plt.figure(figsize=(20,5))

plt.subplot(2, 1, 1)
sns.lineplot(x=rail_ride_df["ride_id"], y=rail_ride_df["avg_speed"], marker="o", color="b", label="Avg Speed")
plt.ylabel("Avg Speed")
plt.title("Average Speed per Ride")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(2, 1, 2) 
sns.lineplot(x=rail_ride_df["ride_id"], y=rail_ride_df["std_speed"], marker="s", color="r", label="Std Dev of Speed")
plt.ylabel("Std Dev of Speed")
plt.xlabel("Ride ID")
plt.title("Standard Deviation of Speed per Ride")
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()

plt.show()