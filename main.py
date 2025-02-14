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

def show_correlation_matrix(corr):

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
#show_correlation_matrix(df_corr)


def prepare_corr_data(df, name):
    df = df_corr[name]
    df = df.reset_index()
    df.columns = ['Feature', 'Correlation']
    df.drop(df.loc[df['Feature']==name].index, inplace=True)
    df = df.sort_values(by='Correlation', ascending=False)

    return df

def show_correlation_bars(df, name):

    ax1 = sns.barplot(data=df, x="Feature", y="Correlation", hue="Correlation", legend=False, alpha=0.8)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()/2), 
                    ha='center', va='bottom', fontsize=10)

    plt.title(f"All correlation for {name}")
    plt.tight_layout()
    plt.show()

def show_pos_corr(df, name):

    meaningful_pos_corr = df.loc[(df['Correlation'] > 0.15)]
    ax1 = sns.barplot(data=meaningful_pos_corr, x="Feature", y="Correlation", hue="Correlation", legend=False, alpha=0.8)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()/2), 
                    ha='center', va='bottom', fontsize=10)

    plt.title(f"Meaningful positive correlation for {name} (>0.15)")
    plt.tight_layout()
    plt.show()
    
def show_neg_corr(df, name):

    meaningful_neg_corr = df.loc[(df['Correlation'] < -0.15)]

    plt.figure()  # Set figure size for the second plot
    ax2 = sns.barplot(data=meaningful_neg_corr, x="Feature", y="Correlation", hue="Correlation", legend=False, alpha=0.8)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()/2), 
                    ha='center', va='bottom', fontsize=10)

    plt.title(f"Meaningful negative correlation for {name} (<-0.15)")
    plt.tight_layout()
    plt.show()

#df_churn_corr = prepare_corr_data(df, "Churn")
#show_correlation_bars(df_churn_corr, "churn")
#show_pos_corr(df_churn_corr, "churn")
#show_neg_corr(df_churn_corr, "churn")

##---------------------------------------------------------
##Taking the most incluential from each side and finding their correlations

#df_churn_corr = prepare_corr_data(df, "Contract_Month-to-month")
#show_pos_corr(df_churn_corr, "month to month contract")
#show_neg_corr(df_churn_corr, "month to month contract")

#df_churn_corr = prepare_corr_data(df, "tenure")
#show_pos_corr(df_churn_corr, "tenure")
#show_neg_corr(df_churn_corr, "tenure")

#koreliacija nebereiksminga nuo 0.15 - 0.2 tame tarpe atsirenkinejam
#paziureti klusterizacija su churn ir be churn

#---------------------------------------------------------
#collapsing dimensions

#gender has no correlation to anything, removing
df.drop(columns=["Male"], inplace=True) 

#---------------------------------------------------------
#Kmeans prep

def show_variance_vs_comp_num(df):
    pca = PCA()
    pca.fit_transform(df)
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_) #how much additional components increase variance

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
    # plt.axhline(y=0.95, color='r', linestyle='--')  # 95% variance threshold
    plt.xlabel('Number Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Variance against Number of Components')
    plt.show()

#show_variance_vs_comp_num(df)

#21 maximum, addvantage of every parameter after that is very close to 0. I should try 7, 10, 15 and 21
n_comp = 21 #not sure if I'd use this many...

def show_kmeans_elbow(df, n):
    pca = PCA(n_components=21)
    df_flat = pca.fit_transform(df)

    wcss= []
    for k in range(1,20):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_flat)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1,20), wcss, marker="o", linestyle="--")
    plt.show()

#show_kmeans_elbow(df, n_comp)

#I think 5 works best
n_clust = 5

#---------------------------------------------------------
#Kmeans action

kmeans = KMeans(n_clusters=n_clust, random_state=42)
df['cluster'] = kmeans.fit_predict(df)

def show_2d_PCA_clusters(df):
    pca = PCA(n_components=2, random_state=42)
    pca_components = pca.fit_transform(df)

    pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
    pca_df['cluster'] = df['cluster']

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x=pca_df['PC1'], y=pca_df['PC2'], c=pca_df['cluster'], cmap='viridis', alpha=0.7)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('KMeans Clusters Visualized with PCA')

    # Create a legend for the clusters
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()
#show_2d_PCA_clusters(df)

def show_3d_PCA_clusters(df):

    pca = PCA(n_components=3, random_state=42)
    pca_components = pca.fit_transform(df)

    pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2', 'PC3'])
    pca_df['cluster'] = df['cluster']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df['cluster'], cmap='viridis', alpha=0.7)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D Visualization of KMeans Clusters with PCA')

    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    plt.show()
#show_3d_PCA_clusters(df)

df_churn_corr = prepare_corr_data(df, "Churn")

notable_features = ["cluster", "Churn"]
notable_features.extend(df_churn_corr["Feature"].head(3).values)
notable_features.extend(df_churn_corr["Feature"].tail(3).values)

# Define your custom bubble plot function for off-diagonals
def cluster_intersection_bubble_plot(x, y, hue=None, **kwargs):
    ax = plt.gca()  # Get the current subplot axis
    
    # Create a DataFrame with the x, y, and cluster values
    temp_df = pd.DataFrame({'x': x, 'y': y, 'cluster': hue})
    
    # Count the number of points at each (x, y, cluster) combination
    grouped = temp_df.groupby(['x', 'y', 'cluster']).size().reset_index(name='count')

    # Plot each (x, y) intersection with bubble size based on count
    for _, row in grouped.iterrows():
        ax.scatter(row['x'], row['y'], 
                   s=row['count'] * 50,  # Adjust size multiplier as needed
                   alpha=0.7, 
                   label=f"Cluster {row['cluster']}")

    # Ensure the plot does not duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

grid = sns.PairGrid(df[notable_features].sample(n=1000), hue= "cluster", palette="Set1", diag_sharey=False, corner=True) 
grid.map_diag(sns.histplot, multiple="dodge")
grid.map_offdiag(sns.histplot, multiple="stack", discrete=True, shrink=0.8)  
grid.add_legend()
plt.show()