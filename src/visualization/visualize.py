import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = df[df["set"] == 1]
plt.plot(set_df["acc_x"]) #Index is set to timestamp originally

plt.plot(set_df["acc_x"].reset_index(drop=True)) #Index is set to row number ie rest_index


# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

df["label"].unique()

for label in df["label"].unique():  #Iterate over the labels to plot each lables
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset["acc_x"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

for label in df["label"].unique():  #Iterate over the first 100 of the rows to plot each lables
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.style.available
mpl.style.use("seaborn-v0_8-darkgrid")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index() # Query the data

fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()  # PLot the heavy and medium sets of participant A when doing squats
ax.set_ylabel("acc_y")
ax.set_xlabel("Category heavy/medium")
plt.legend()


category_df = df.query("label == 'ohp'").query("participant == 'A'").reset_index()# Query the data

fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()  # PLot the heavy and medium sets of participant A when doing overhead press
ax.set_ylabel("acc_y")
ax.set_xlabel("Category heavy/medium")
plt.legend()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = df.query("label == 'squat'").sort_values("participant").reset_index() # Query the data, sort_values sort the participants

fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()  # PLot the heavy and medium sets of participant A when doing squats
ax.set_ylabel("acc_y")
ax.set_xlabel("Participants")
plt.legend()


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = "row"
participant = "C"

allaxis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index() # Query the data

fig, ax = plt.subplots()
allaxis_df[["acc_x","acc_y","acc_z"]].plot(ax=ax)  # PLot the heavy and medium sets of participant A when doing squats
ax.set_ylabel("Acceleration")
ax.set_xlabel("Participant C with row")
plt.legend()


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
labels = df["label"].unique()
participants = df["participant"].unique()

#Accelaration plot for all participants with label
for label in labels:
    for participant in participants:
        allaxis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index() # Query the data

        if len(allaxis_df) > 0:
            fig, ax = plt.subplots()
            allaxis_df[["acc_x","acc_y","acc_z"]].plot(ax=ax)  # PLot the heavy and medium sets of participant A when doing squats
            ax.set_ylabel("Acceleration")
            ax.set_xlabel(f"{participant} with {label}")
            plt.legend()
            plt.show()

for label in labels:
    for participant in participants:
        allaxis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index() # Query the data

        if len(allaxis_df) > 0:
            fig, ax = plt.subplots()
            allaxis_df[["gyro_x","gyro_y","gyro_z"]].plot(ax=ax)  # PLot the heavy and medium sets of participant A when doing squats
            ax.set_ylabel("Gyroscope")
            ax.set_xlabel(f"{participant} with {label}")
            plt.legend()
            plt.show()

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


label = "row"
participant = "A"
combined_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index() # Query the data

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
combined_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
combined_df[["gyro_x","gyro_y","gyro_z"]].plot(ax=ax[1])

ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        combined_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index() # Query the data

        if len(combined_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            combined_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
            combined_df[["gyro_x","gyro_y","gyro_z"]].plot(ax=ax[1])

            ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            plt.savefig(f"../../reports/figures/{participant}_{label}.png")

            plt.show()