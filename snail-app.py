######################
# Import libraries
######################

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from PIL import Image
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)

df_ = pd.read_csv(r"SnailTrailDataset.csv")
df = df_.copy()
######################
# Page Title
######################

######################
# Functions
######################

# Preparing Dataset
def dataset_prepare(dataframe, data_save=False):

    """

    Dataset Preparation

    :param dataframe: It is original dataset.
    :param data_save: If you want to save dataset, you can write True.
    :return: Prepared Dataset

    """

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

    """

    Describing the family of Snails

    :param dataframe: It is original dataset.
    :param snail_no: You can write the number of snail you want.
    :param data_save: If you want to save dataset, you can write True.
    :return: Prepared Dataset

    """

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

    """

    Creating new column of gen of Snail

    :param dataframe: It is original dataset.
    :param snail_no: You can write the number of snail you want.
    :param data_save: If you want to save dataset, you can write True.
    :return: Prepared Dataset.

    """

    snail_no = snail_no - 1

    gens = []

    for i in dataframe["Gene"][snail_no]:
        gens.append(i)

    for i in range(1, 21):
        dataframe["Gen" + str(i)][snail_no] = gens[i - 1]

    if data_save==True:
        dataframe.to_csv('SnailTrailDataset.csv', index=False)

    return dataframe

# Gen sequence describing
def reproduction(dataframe, Own_Snail, Partner_Snail, possible_production):

    """

    Reproduction

    :param dataframe: It is original dataset.
    :param Own_Snail: Snail that you have.
    :param Partner_Snail: Snail that you want to product a new with it.
    :param possible_production: How much time DNA Crossing you want. You can give a number. Recommend: 10000
    :return: Dataset of possible snails

    """

    P1 = dataframe[Own_Snail-1 : Own_Snail]
    P2 = dataframe[Partner_Snail-1 : Partner_Snail]

    new_snails = []

    while len(new_snails) < possible_production:
        genom = ['Gen1', 'Gen2', 'Gen3', 'Gen4', 'Gen5', 'Gen6', 'Gen7', 'Gen8', 'Gen9', 'Gen10',
                 'Gen11', 'Gen12', 'Gen13', 'Gen14', 'Gen15', 'Gen16', 'Gen17', 'Gen18', 'Gen19', 'Gen20']
        P1_random_gen = []
        P2_random_gen = []
        for i in range(1, 11):
            i = np.random.choice(genom)
            P1_random_gen.append(i)
            genom.remove(i)
            P2_random_gen = genom

        new_snail = pd.concat([P1[P1_random_gen], P2[P2_random_gen]], axis=0, ignore_index=True)
        new_snail_P1 = pd.DataFrame([new_snail[P1_random_gen].values[0].tolist()], columns=new_snail[P1_random_gen].columns)
        new_snail_P2 = pd.DataFrame([new_snail[P2_random_gen].values[1].tolist()], columns=new_snail[P2_random_gen].columns)
        new_snail = pd.concat([new_snail_P1, new_snail_P2], axis=1)
        new_snail = new_snail[['Gen1', 'Gen2', 'Gen3', 'Gen4', 'Gen5', 'Gen6', 'Gen7', 'Gen8', 'Gen9', 'Gen10',
                               'Gen11', 'Gen12', 'Gen13', 'Gen14', 'Gen15', 'Gen16', 'Gen17', 'Gen18', 'Gen19', 'Gen20']]
        new_snail["Gene"] = new_snail.sum().sum()
        new_snail = new_snail["Gene"]
        new_snails.append(new_snail.values.tolist())
        possible_new_snails = pd.DataFrame(new_snails)
        possible_new_snails.columns = ["Gene"]
        possible_new_snails.drop_duplicates(inplace=True)
        possible_new_snails.reset_index(inplace=True)
        possible_new_snails = possible_new_snails[["Gene"]]

    return possible_new_snails

# Conclusion for new snail possibilities
def conclusion(dataframe, Own_Snail, Partner_Snail):

    """

    Conclusion of new dataset (new snails)

    :param dataframe: New Snails Dataset.
    :param Own_Snail: Snail that you have.
    :param Partner_Snail: Snail that you want to product a new with it.

    """

    P1 = df[Own_Snail-1 : Own_Snail]
    P2 = df[Partner_Snail-1 : Partner_Snail]

    print("######## Own Snail ########")
    print(P1[["Snail_No", "Gene", "Family", "Purity"]])
    print("\n")
    print("######## Partner Snail ########")
    print(P2[["Snail_No", "Gene", "Family", "Purity"]])
    print("\n\n")

    print("Total Snail Possibility: " + str(len(dataframe)))
    print("Total Garden Possibility: " + str(len(dataframe[dataframe["Family"] == "Garden"])))
    print("Total Helix Possibility: " + str(len(dataframe[dataframe["Family"] == "Helix"])))
    print("Total Milk Possibility: " + str(len(dataframe[dataframe["Family"] == "Milk"])))
    print("Total Agate Possibility: " + str(len(dataframe[dataframe["Family"] == "Agate"])))
    print("Total Atlantis Possibility: " + str(len(dataframe[dataframe["Family"] == "Atlantis"])))
    print("###############################\n\n")

    rate_garden = len(dataframe[dataframe["Family"] == "Garden"]) / len(dataframe) * 100
    rate_helix = len(dataframe[dataframe["Family"] == "Helix"]) / len(dataframe) * 100
    rate_milk = len(dataframe[dataframe["Family"] == "Milk"]) / len(dataframe) * 100
    rate_agate = len(dataframe[dataframe["Family"] == "Agate"]) / len(dataframe) * 100
    rate_atlantis = len(dataframe[dataframe["Family"] == "Atlantis"]) / len(dataframe) * 100

    print("######## Family: Garden ########")
    print(f"Possibility: {rate_garden:.2f}%")
    for i in range(0, len(dataframe[dataframe["Family"] == "Garden"]["Purity"].unique())):
        a = dataframe[dataframe["Family"] == "Garden"]["Purity"] == \
            dataframe[dataframe["Family"] == "Garden"]["Purity"].unique()[i]
        print(
            "Purity=" + str(dataframe[dataframe["Family"] == "Garden"]["Purity"].unique()[i]) + " Possibility: " + str(
                int(a[a == True].count() / a.count() * 100)) + "%")
    print("###############################\n\n")

    print("######## Family: Helix ########")
    print(f"Possibility: {rate_helix:.2f}%")
    for i in range(0, len(dataframe[dataframe["Family"] == "Helix"]["Purity"].unique())):
        a = dataframe[dataframe["Family"] == "Helix"]["Purity"] == \
            dataframe[dataframe["Family"] == "Helix"]["Purity"].unique()[i]
        print("Purity=" + str(dataframe[dataframe["Family"] == "Helix"]["Purity"].unique()[i]) + " Possibility: " + str(
            int(a[a == True].count() / a.count() * 100)) + "%")
    print("###############################\n\n")

    print("######## Family: Milk ########")
    print(f"Possibility: {rate_milk:.2f}%")
    for i in range(0, len(dataframe[dataframe["Family"] == "Milk"]["Purity"].unique())):
        a = dataframe[dataframe["Family"] == "Milk"]["Purity"] == \
            dataframe[dataframe["Family"] == "Milk"]["Purity"].unique()[i]
        print("Purity=" + str(dataframe[dataframe["Family"] == "Milk"]["Purity"].unique()[i]) + " Possibility: " + str(
            int(a[a == True].count() / a.count() * 100)) + "%")
    print("###############################\n\n")

    print("######## Family: Agate ########")
    print(f"Possibility: {rate_agate:.2f}%")
    for i in range(0, len(dataframe[dataframe["Family"] == "Agate"]["Purity"].unique())):
        a = dataframe[dataframe["Family"] == "Agate"]["Purity"] == \
            dataframe[dataframe["Family"] == "Agate"]["Purity"].unique()[i]
        print("Purity=" + str(dataframe[dataframe["Family"] == "Agate"]["Purity"].unique()[i]) + " Possibility: " + str(
            int(a[a == True].count() / a.count() * 100)) + "%")
    print("###############################\n\n")

    print("######## Family: Atlantis ########")
    print(f"Possibility: {rate_atlantis:.2f}%")
    for i in range(0, len(dataframe[dataframe["Family"] == "Atlantis"]["Purity"].unique())):
        a = dataframe[dataframe["Family"] == "Atlantis"]["Purity"] == \
            dataframe[dataframe["Family"] == "Atlantis"]["Purity"].unique()[i]
        print("Purity=" + str(
            dataframe[dataframe["Family"] == "Atlantis"]["Purity"].unique()[i]) + " Possibility: " + str(
            int(a[a == True].count() / a.count() * 100)) + "%")
    print("###############################")

# Snail Finder
def snail_finder_gen(dataframe, Gene):
    return dataframe[dataframe["Gene"] == Gene]
def snail_finder_no(df, Snail_No):
    return df.loc[df["Gene"] == str(Snail_No), "Snail_No"].iloc[0]
def snail_family_finder(dataframe, Gene):
    return dataframe[dataframe["Gene"] == Gene]["Family"].values[0]

# Snail Reproduction
def possible_new_snail(dataframe, Own_Snail, Partner_Snail, possible_production, data_save=False):

    """

    Possible New Snails

    :param dataframe: It is original dataset.
    :param Own_Snail: Snail that you have.
    :param Partner_Snail: Snail that you want to product a new with it.
    :param possible_production: How much time DNA Crossing you want. You can give a number. Recommend: 10000
    :param data_save: If you want to save dataset, you can write True.
    :return: Dataset of possible snails

    """

    new_snails = reproduction(dataframe, Own_Snail, Partner_Snail, possible_production)
    new_snails = dataset_prepare(new_snails, False)
    for i in range(1, len(new_snails)+1):
        snail_family(new_snails, i, False)
    new_snails = new_snails[["Snail_No", "Gene", "Family", "Purity", "Garden", "Helix", "Milk", "Agate", "Atlantis"]]

    Family = ["Garden", "Helix", "Milk", "Agate", "Atlantis"]
    for j in Family:
        for i in range(0, len(new_snails[new_snails["Family"] == j]["Purity"].unique())):
            new_snails[str(j) + "_rate"] = int(len(new_snails[new_snails["Family"] == j]) / len(new_snails) * 100)
            purities = new_snails[new_snails["Family"] == j]["Purity"] == new_snails[new_snails["Family"] == j]["Purity"].unique()[i]
            new_snails[str(j) + "_purity_" + str(new_snails[new_snails["Family"] == j]["Purity"].unique()[i])] = int(purities[purities == True].count() / purities.count() * 100)

    conclusion(new_snails, Own_Snail, Partner_Snail)

    if data_save==True:
        new_snails.to_csv('NewSnail_' + str(Own_Snail) + '_' + str(Partner_Snail) + '.csv', index=False)

    return new_snails

# Snail Possibilities
def best_possible_snail_purities(dataframe, Own_Snail, possible_production, data_save=False):

    # Best Snails Dataset
    best_snails = best_snails_data()

    P1 = dataframe[Own_Snail - 1: Own_Snail]
    best_snails["P1_No"] = P1["Snail_No"].values.tolist()
    best_snails["P1_Gen"] = P1["Gene"].values.tolist()
    P2 = dataframe[Own_Snail+1 - 1: Own_Snail+1]
    best_snails["P2_No"] = P2["Snail_No"].values.tolist()
    best_snails["P2_Gen"] = P2["Gene"].values.tolist()

    for i in range(1, 100):
        P1 = dataframe[Own_Snail - 1: Own_Snail]
        best_snails2["P1_No"] = P1["Snail_No"].values.tolist()
        best_snails2["P1_Gen"] = P1["Gene"].values.tolist()
        P3 = dataframe[i - 1: i]
        best_snails2["P2_No"] = P3["Snail_No"].values.tolist()
        best_snails2["P2_Gen"] = P3["Gene"].values.tolist()
        best_snails = best_snails.append(best_snails2, ignore_index=True)

        # New possible snails
        new_snails = reproduction(dataframe, Own_Snail, i, possible_production)
        new_snails = dataset_prepare(new_snails, False)
        for j in range(1, len(new_snails) + 1):
            snail_family(new_snails, j, False)
        new_snails = new_snails[["Snail_No", "Gene", "Family", "Purity", "Garden", "Helix", "Milk", "Agate", "Atlantis"]]

        # Conclusion of new possible snails
        Family = ["Garden", "Helix", "Milk", "Agate", "Atlantis"]
        for k in Family:
            for l in range(0, len(new_snails[new_snails["Family"] == k]["Purity"].unique())):
                best_snails[str(k) + "_rate"] = int(
                    len(new_snails[new_snails["Family"] == k]) / len(new_snails) * 100)
                #purities = new_snails[new_snails["Family"] == k]["Purity"] == \
                           #new_snails[new_snails["Family"] == k]["Purity"].unique()[l]
                #best_snails[
                    #str(k) + "_purity_" + str(new_snails[new_snails["Family"] == j]["Purity"].unique()[l])] = int(
                    #purities[purities == True].count() / purities.count() * 100)

        if data_save == True:
            best_snails.to_csv('BestSnail_' + str(Own_Snail) + '_' + str(k) + '.csv', index=False)

    return best_snails
# doesn't work

# Best Snail Dataset (just columns)
def best_snails_data():
    snail_parent = ["P1_No", "P1_Gen", "P2_No", "P2_Gen"]

    val1 = []
    for i in range(1, len(snail_parent) + 1):
        val1.append(0)

    dataframe1 = pd.DataFrame(data=[val1], columns=snail_parent)

    Family = ["Garden", "Helix", "Milk", "Agate", "Atlantis"]

    col = []
    for j in Family:
        col += [str(j) + "_rate"]
        for i in range(1, 21):
            col += [str(j) + "_purity_" + str(i)]

    val2 = []
    for i in range(1, len(col) + 1):
        val2.append(0)

    dataframe2 = pd.DataFrame(data=[val2], columns=col)

    dataframe = dataframe1.join(dataframe2, how="left")

    return dataframe

# Best Snail Dataset (without gens)
def best_snails_data_without_gens():
    snail_parent = ["P1_No", "P1_Gen", "P2_No", "P2_Gen", "P2_Family"]

    val1 = []
    for i in range(1, len(snail_parent) + 1):
        val1.append(0)

    dataframe1 = pd.DataFrame(data=[val1], columns=snail_parent)

    Family = ["Garden", "Helix", "Milk", "Agate", "Atlantis"]

    col = []
    for j in Family:
        col += [str(j) + "_rate"]

    val2 = []
    for i in range(1, len(col) + 1):
        val2.append(0)

    dataframe2 = pd.DataFrame(data=[val2], columns=col)

    dataframe3 = dataframe1.join(dataframe2, how="left")

    return dataframe3

# Best Snail Dataset (without values, just preparing)
def best_snail_dataset(dataframe, Own_Snail, R1, R2):

    best_snails = best_snails_data_without_gens()
    best_snails2 = best_snails

    for i in range(R1, R2):
        P1 = dataframe[Own_Snail - 1: Own_Snail]
        best_snails2["P1_No"] = P1["Snail_No"].values.tolist()
        best_snails2["P1_Gen"] = P1["Gene"].values.tolist()
        P2 = dataframe[i - 1: i]
        best_snails2["P2_No"] = P2["Snail_No"].values.tolist()
        best_snails2["P2_Gen"] = P2["Gene"].values.tolist()
        best_snails = best_snails.append(best_snails2, ignore_index=True)

    best_snails.drop(0, inplace=True)
    best_snails = best_snails.reset_index()
    best_snails = best_snails.drop("index", axis=1)
    for i in best_snails["P2_No"]:
        if i == best_snails["P1_No"].values[0]:
            best_snails.drop(i - 1, inplace=True)
            best_snails = best_snails.reset_index()
            best_snails = best_snails.drop("index", axis=1)
        else:
            best_snails = best_snails

    p2family = [snail_family_finder(dataframe, i) for i in best_snails["P2_Gen"]]
    best_snails["P2_Family"] = p2family

    return best_snails

# Best Snail Results
def best_snail_conclusion(dataframe, Own_Snail, Partner_Snail, possible_production):
    new_snails_conclusion = best_snails_data_without_gens()

    new_snails = reproduction(dataframe, Own_Snail, Partner_Snail, possible_production)
    new_snails = dataset_prepare(new_snails, False)

    for j in range(1, len(new_snails) + 1):
        snail_family(new_snails, j, False)
    new_snails = new_snails[["Snail_No", "Gene", "Family", "Purity", "Garden", "Helix", "Milk", "Agate", "Atlantis"]]

    Family = ["Garden", "Helix", "Milk", "Agate", "Atlantis"]
    for k in Family:
        new_snails_conclusion[str(k) + "_rate"] = int(
            len(new_snails[new_snails["Family"] == k]) / len(new_snails) * 100)

    return new_snails_conclusion

############################### SNAIL PROGRAMM ###############################


def best_snails(dataframe, Own_Snail, R1, R2, possible_production, Partner_Snail_Family, Target_Snail_Family, data_save=False):
    """

    Best Snail possibilities

    :param dataframe: It is original dataset.
    :param Own_Snail: You Snail Number.
    :param R1: First number of Partner Snail for Range.
    :param R2: Second number of Partner Snail for Range.
    :param possible_production: How much time DNA Crossing you want. You can give a number. Recommend: 10000
    :param data_save: If you want to save data as csv.
    :return:

    """

    best_snail = best_snail_dataset(dataframe, Own_Snail, R1, R2)

    best_snail = best_snail[best_snail["P2_Family"] == Partner_Snail_Family]


    Family = ["Garden", "Helix", "Milk", "Agate", "Atlantis"]
    for i in range(R1-1,len(best_snail)):
        best_snail_result = best_snail_conclusion(df, best_snail["P1_No"].values[0], best_snail["P2_No"].values[i], possible_production)
        for j in Family:
            best_snail[str(j)+"_rate"][i] = best_snail_result[str(j)+"_rate"][0]

    for k in Family:
        if Target_Snail_Family == k:
            best_snail_family = best_snail.sort_values(by=str(k) + "_rate", ascending=False)

    best_snail_family = best_snail_family[["P2_No", "P2_Gen", "Atlantis_rate", "Agate_rate", "Milk_rate", "Helix_rate", "Garden_rate"]]
    best_snail_family.columns = ["P2 Nr", "P2 Gene", "Atlantis Rate(%)", "Agate Rate(%)", "Milk Rate(%)", "Helix Rate(%)", "Garden Rate(%)"]

    snail_link = ["https://www.snailtrail.art/snails/" + str(i) + "/snail" for i in best_snail_family["P2 Nr"]]
    best_snail_family["Link"] = snail_link

    if data_save==True:
        best_snail_family.to_csv('New_Snail_results_SnailNo_' + str(Own_Snail) + '.csv', index=False)

    return best_snail_family


image = Image.open('SnailTrail.png')
st.sidebar.image(image, use_column_width=True)

st.sidebar.header("**Snail Date App**")
sequence = st.sidebar.text_input("Enter Your Snail's Gene")




if sequence in df["Gene"].unique():
    st.markdown('<p style="font-family: Courier;text-align: center;'
                ' color:Purple; font-size: 90px;">Your Snail</p>', unsafe_allow_html=True)

    no = df.loc[df["Gene"] == str(sequence), "Snail_No"].iloc[0]
    family = snail_family_finder(df, sequence)
    image0 = Image.open(f'{family}.png')
    col1, col2, col3,col4 = st.columns([1, 5, 3, 1])
    col2.markdown(f"<h1 style='text-align: left; font-size: 80px; color: grey;'> {family}</h1>", unsafe_allow_html=True)
    col3.image(image0, width=150)
    st.write("  ")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col2.metric("Atlantis", sequence.count('X'))
    col3.metric("Agate", sequence.count('A'))
    col4.metric("Milk", sequence.count('M'))
    col5.metric("Helix", sequence.count('H'))
    col6.metric("Garden", sequence.count('G'))

    def DNA_nucleotide_count(seq):
        d = dict([
            ('X', seq.count('X')),
            ('A', seq.count('A')),
            ('M', seq.count('M')),
            ('H', seq.count('H')),
            ('G', seq.count('G'))
        ])
        return d

    X = DNA_nucleotide_count(sequence)

    dff = pd.DataFrame.from_dict(X, orient='index')
    dff = dff.rename({0: 'count'}, axis='columns')
    dff.reset_index(inplace=True)
    dff = dff.rename(columns={'index': 'Family'})

    p = alt.Chart(dff).mark_circle().encode(x='Family', y='count', size='count', color='Family', tooltip=['Family', 'count']).properties(width=alt.Step(100))

    #st.altair_chart(p)

    # st.write(possible_new_snail(df, 10, 40, 100))

    option = st.sidebar.selectbox(
        'Which family do you want to get?',
        ('Choose Family', 'Agate', 'Atlantis', 'Milk', 'Helix', 'Garden'))

    st.markdown('<p style="font-family: Courier;text-align: center;'
                ' color:Purple; font-size: 55px;"> Gene Production', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns((1, 1, 1, 1, 1))
    col6, col7, col8, col9, col10 = st.columns((1, 1, 1, 1, 1))

    col1.image(image0, use_column_width=True)
    col6.markdown(f"<h5 style='text-align: center; color: white;'>{family}</h1>", unsafe_allow_html=True)
    col2.markdown(f"<h1 style='text-align: center; color: grey;'>+</h1>", unsafe_allow_html=True)


    if option != "Choose Family":
        image1 = Image.open(f'{option}.png')
        #st.sidebar.image(image1, width=130)
        col4.markdown(f"<h1 style='text-align: center; color: grey;'>=</h1>", unsafe_allow_html=True)
        col5.image(image1, use_column_width=True)
        col10.markdown(f"<h5 style='text-align: top; color: white;'>{option}</h1>", unsafe_allow_html=True)

        option2 = st.sidebar.selectbox(
            'Which family do you want to breed with?',
            ('Choose Family', 'Atlantis', 'Agate', 'Milk', 'Helix', 'Garden'))
        if option2 != "Choose Family":
            image2 = Image.open(f'S_{option2}.png')
            #st.sidebar.image(image2, width=150)
            col3.image(image2, use_column_width=True)

            col8.markdown(f"<h5 style='text-align: center; color: white;'>{option2}</h1>", unsafe_allow_html=True)

            df123 = best_snails(df, df.loc[df["Gene"] == str(sequence), "Snail_No"].iloc[0], 1, 180, 10, option2, option)

            st.write("  ")
            st.write("  ")

            st.markdown('<p style="font-family: Courier;text-align: center;'
                        f' color:Purple; font-size: 50px;">Matches', unsafe_allow_html=True)
            st.sidebar.write(" ")
            st.sidebar.write(" ")
            st.sidebar.write(" ")
            st.sidebar.write(" ")
            st.sidebar.write(" ")
            st.sidebar.write(" ")
            st.sidebar.write(" ")
            st.sidebar.write(" ")


            st.sidebar.write("Made by Chambers")
            text_string_variable1 = "--------Github--------"
            url_string_variable1 = "https://github.com/chamberss2022/snail-app"
            link = f'[{text_string_variable1}]({url_string_variable1} ) '
            st.sidebar.markdown(link, unsafe_allow_html=True)

            text_string_variable2 = "-------Linkedin------"
            url_string_variable2 = "https://www.linkedin.com/company/81626136"
            link = f'[{text_string_variable2}]({url_string_variable2} ) '
            st.sidebar.markdown(link, unsafe_allow_html=True)
            
            text_string_variable3= "-------Medium-------"
            url_string_variable3 = "https://medium.com/@haner11/snail-date-app-via-snail-trail-from-chambers-690a2aee89bc"
            link = f'[{text_string_variable3}]({url_string_variable3} ) '
            st.sidebar.markdown(link, unsafe_allow_html=True)

            github = Image.open('chambers_logo.png')
            st.sidebar.image(github, width=125)

            a = 0
            for i in range(len(df123)):
                if df123.iloc[i:(i+1), 0:7][f"{option} Rate(%)"].head(1).iloc[0] != 0:
                    st.balloons()
                    col1, col2, col3, col4 = st.columns((1.75, 1, 1, 1))

                    text_string_variable = df123.iloc[i:(i+1), 0:7]["P2 Gene"].head(1).iloc[0]
                    url_string_variable = f"https://www.snailtrail.art/snails/{df123.iloc[i:(i+1), 0:7]['P2 Nr'].head(1).iloc[0]}/snail"
                    link = f'[{text_string_variable}]({url_string_variable} ) '
                    col2.markdown(link, unsafe_allow_html=True)

                    #col2.write(f"https://www.snailtrail.art/snails/{df123.iloc[i:(i+1), 0:7]['P2 Nr'].head(1).iloc[0]}/snail")

                    #col4.write(str(df123.iloc[i:(i+1), 0:7]["Agate Rate(%)"].head(1).iloc[0])+"%")

                    st.write(df123.iloc[i:(i+1), 2:7])
                    st.write("  ")
                    st.write("  ")
                    st.write("  ")
                    st.write("  ")
                    a = 1
                    
            if a == 0:
                st.write("No matches found, sorry :(")



else:
    st.markdown("""
    <style>
    .big-font {
        font-size:12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<p class="big-font">Example: HHXHMMGMAMAMMMAMMMMH</p>', unsafe_allow_html=True)







