import numpy as np
import pandas as pd
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 50)


############################### DATASET ###############################
df_ = pd.read_csv("Snail-Data/snail_gens.csv")
df = df_.copy()
df.tail()


############################### FUNCTIONS ###############################

# Preparing Dataset
def dataset_prepare(dataframe, data_save=False):
    # Adding snail no
    dataframe = dataframe.reset_index()
    dataframe.columns = ["Snail_No", "Gene"]
    dataframe["Snail_No"] = [i + 1 for i in dataframe["Snail_No"]]

    # Adding family columns
    dataframe[["Family", "Purity", "Garden", "Helix", "Milk", "Agate", "Atlantis"]] = 0

    # Adding gen sequence columns
    for i in range(1, 21):
        dataframe["Gen" + str(i)] = 0

    if data_save==True:
        dataframe.to_csv('SnailTrailDataset.csv', index=False)

    return dataframe

# Family Describing
def snail_family(dataframe, snail_no, data_save=False):
    snail_no = snail_no - 1

    G = []
    H = []
    M = []
    A = []
    X = []

    for i in range(0, 20):

        if dataframe["Gene"][snail_no][i] == "G":
            G.append(i)
            dataframe["Garden"][snail_no] = len(G)

        elif dataframe["Gene"][snail_no][i] == "H":
            H.append(i)
            dataframe["Helix"][snail_no] = len(H)

        elif dataframe["Gene"][snail_no][i] == "M":
            M.append(i)
            dataframe["Milk"][snail_no] = len(M)

        elif dataframe["Gene"][snail_no][i] == "A":
            A.append(i)
            dataframe["Agate"][snail_no] = len(A)

        else:
            X.append(i)
            dataframe["Atlantis"][snail_no] = len(X)

    dataframe["Family"][snail_no] = dataframe[["Garden", "Helix", "Milk", "Agate", "Atlantis"]].loc[snail_no].sort_values(ascending=False).index[0]
    dataframe["Purity"][snail_no] = dataframe[["Garden", "Helix", "Milk", "Agate", "Atlantis"]].loc[snail_no].sort_values(ascending=False)[0]

    if data_save==True:
        dataframe.to_csv('SnailTrailDataset.csv', index=False)

    return dataframe

# Gen sequence describing
def snail_gen(dataframe, snail_no, data_save=False):
    snail_no = snail_no - 1

    gens = []

    for i in dataframe["Gene"][snail_no]:
        gens.append(i)

    for i in range(1, 21):
        dataframe["Gen" + str(i)][snail_no] = gens[i - 1]

    if data_save==True:
        dataframe.to_csv('SnailTrailDataset.csv', index=False)

    return dataframe


############################### SAVING DATASET ###############################
df = dataset_prepare(df, False)
for i in range(1, len(df)+1):
    snail_family(df, i, False)
for i in range(1, len(df)+1):
    snail_gen(df, i, True)