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



def show_column_analysis(df):
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

double_columns = df.loc[:, df.nunique() == 2]
#show_column_analysis(double_columns)

#turning "yes"/"no" columns into 1/0 binary
for column in double_columns.columns:
    if set(df[column].values).issubset(["Yes", "No"]):
        df[column] = df[column].map({'Yes': 1, 'No': 0})

df["Male"] = df["gender"].map({'Male': 1, 'Female': 0})
df.drop(columns="gender", inplace=True)

#checks if everything went ok - it did!
# double_columns.drop(columns="gender", inplace=True)
# double_columns["Male"] = df["Male"]
# print(df[double_columns.columns].head(30))

#---------------------------------------------------------
#OHE ENCODING WHERE POSSIBLE

#how many values max do we want the columns to have for it to get one hot encoded
ohe_num = 5

df_ohe = df.loc[:, (df.nunique() <= ohe_num) & (df.nunique() > 2)] 
#show_column_analysis(df_ohe)

ohe_columns = df_ohe.columns

ohe = OneHotEncoder(sparse_output=False).set_output(transform='pandas') # will allow pandas dataset manipulation
df_encoded = ohe.fit_transform(df[ohe_columns])
df = pd.concat([df, df_encoded], axis = 1).drop(columns=ohe_columns)

#label encoding is now rudimentary

#---------------------------------------------------------
#Scaling data and turning the scaled data back into a df

scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

#---------------------------------------------------------
#Printing correlation matrix

def show_correlation(corr):

    #this will mask the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Dynamically compute figure size based on number of rows and columns
    n_rows, n_cols = corr.shape
    cell_size = 0.5  # inches per cell (adjust as needed)
    fig_width = n_cols * cell_size
    fig_height = n_rows * cell_size

    f, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Generating a custom diverging colormap because with the current one,  it's sad situation
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr, 
        mask=mask, 
        cmap=cmap, 
        annot=True, #i'd like to have numbers inside my matrix 
        annot_kws={"size": 5},  #this should make my font size smaller
        vmax=.3, 
        center=0,
        #square=True, 
        linewidths=.5, 
        fmt='0.1f')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='right')
    plt.subplots_adjust(bottom=0.3)
    plt.show()

#wanted to reorder churn
df = df[['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
       'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'Male',
       'MultipleLines_No', 'MultipleLines_No phone service',
       'MultipleLines_Yes', 'InternetService_DSL',
       'InternetService_Fiber optic', 'InternetService_No',
       'OnlineSecurity_No', 'OnlineSecurity_No internet service',
       'OnlineSecurity_Yes', 'OnlineBackup_No',
       'OnlineBackup_No internet service', 'OnlineBackup_Yes',
       'DeviceProtection_No', 'DeviceProtection_No internet service',
       'DeviceProtection_Yes', 'TechSupport_No',
       'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No',
       'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No internet service',
       'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
       'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 'Churn']]

df_corr = df.corr()
#show_correlation(df_corr)

#---------------------------------------------------------
#Cleaning data to get churn correlations specificly

df_churn_corr = df_corr['Churn']
df_churn_corr = df_churn_corr.reset_index()
df_churn_corr.columns = ['Feature', 'Correlation']
df_churn_corr.drop(df_churn_corr.loc[df_churn_corr['Feature']=='Churn'].index, inplace=True)
df_churn_corr = df_churn_corr.sort_values(by='Correlation', ascending=False)

#---------------------------------------------------------
#Plotting meaningful positive correlation

meaningful_pos_corr = df_churn_corr.loc[(df_churn_corr['Correlation'] > 0.15)]

plt.figure()  # Set figure size for the first plot
ax1 = sns.barplot(data=meaningful_pos_corr, x="Feature", y="Correlation", hue="Correlation", legend=False, alpha=0.8)

# Rotate x-axis labels and align them properly
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

# Annotate bars with correlation values
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()/2), 
                 ha='center', va='bottom', fontsize=10)

plt.title("Meaningful positive correlation for churn (>0.15)")
plt.tight_layout()
plt.show()

#---------------------------------------------------------
#Plotting meaningful negative correlation

meaningful_neg_corr = df_churn_corr.loc[(df_churn_corr['Correlation'] < -0.15)]

plt.figure()  # Set figure size for the second plot
ax2 = sns.barplot(data=meaningful_neg_corr, x="Feature", y="Correlation", hue="Correlation", legend=False, alpha=0.8)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

for p in ax2.patches:
    ax2.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()/2), 
                 ha='center', va='bottom', fontsize=10)

plt.title("Meaningful negative correlation for churn (<-0.15)")
plt.tight_layout()
plt.show()

print(df_churn_corr)

#koreliacija nebereiksminga nuo 0.15 - 0.2 tame tarpe atsirenkinejam
#paziureti klusterizacija su churn ir be churn