import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

#No duplicates!
#print("Duplicates: ",df.duplicated().sum())

#No nulls! well that was a lie
#print("Nulls: ",df.isnull().sum())

df.drop(columns=['customerID'], inplace=True)

#no null was a trick..
for column in df.columns:
    df[column] = df[column].replace(" ", None)
    df = df.dropna(how='any', axis=0) #removes 11 rows where TotalCharges were missing

def show_clumn_analysis(df):
    for column in df.columns:

        name = column
        count = df[column].count()
        miss_count = df[column].isna().sum()
        miss = (df[column].isna().sum() / len(df[column])) * 100
        card = len(df[column].unique())

        if df[column].dtype not in ["float64", "int64"]:
            print("Name: ", name)
            #there were no missing, so to remove print cluttering i removed them/ mistake
            print("Value count: ", count)
            print("Missing count: ", miss_count)
            print("Missing in %: ", miss)
            print("Unique count: ", card)
            print("Values: ", df[column].unique() if len(df[column].unique()) < 10 else (df[column].unique()[:11],"..."))
            print("*"*80)
#show_clumn_analysis(df)

#---------------------------------------------------------
#OHE ENCODING WHERE POSSIBLE

ohe_num = 5
#ohe_columns = df_cat["Feature"].loc[df_cat["Card."] <= ohe_num ].values
ohe_columns = df.loc[:, df.nunique() <= ohe_num ].columns # : selects all the rows
#print(ohe_columns)

ohe = OneHotEncoder(sparse_output=False).set_output(transform='pandas') # will allow pandas dataset manipulation
df_encoded = ohe.fit_transform(df[ohe_columns])
df = pd.concat([df, df_encoded], axis = 1).drop(columns=ohe_columns)

#---------------------------------------------------------
#LABEL ENCODING THE REMAINING TEXT DATA (Should be none for )

for column in df.select_dtypes(exclude=['float64', 'int']).columns:

    label_encoder = LabelEncoder()
    label_encoder.fit(df[column])
    df[column] = label_encoder.fit_transform(df[column])
    print("Label encoded: ", column)

#---------------------------------------------------------
#Scaling data and turning the scaled data back into a df

scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
#print(df.columns)

def show_correlation(df):

    #Correlations I'm interested in
    custom_corrs = ['Churn_No'] # getting info of what makes people stay

    # Compute the correlation matrix
    corr = df.corr()
    corr = corr.loc[custom_corrs] # I dont need the whole matrix, oonly these two

    print(corr)

    # # Generate a mask for the upper triangle
    # mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots()

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    #sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, vmax=.3, center=0,square=True, linewidths=.5, fmt='0.1f')
    sns.heatmap(corr, cmap=cmap, annot=True, vmax=.3, center=0,square=True, linewidths=.5, fmt='0.1f')


    # ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='right')
    plt.subplots_adjust(bottom=0.3)
    plt.show()
show_correlation(df)