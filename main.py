import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

#---------------------------------------------------------
#DAta cleanup and preparation

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
#Feature reduction

df_churn_corr = prepare_corr_data(df, "Churn")

def show_meaningful_corr_features(df, cor_range):
    f_num=[]

    for c in cor_range:
        meaningful_corr = df.loc[abs(df['Correlation']) > c]
        f_num.append(meaningful_corr.shape[0])

    plt.figure(figsize=(8, 5))
    plt.plot(cor_range, f_num, marker='o', linestyle='--')
    plt.xlabel('Absolute meaningful correlation')
    plt.ylabel('Number of features selected')
    plt.title('Number of features selected per absolute meaningful correlation')
    plt.show()

meaningful_corrs = [x/100 for x in (range(15, 26))]
#show_meaningful_corr_features(df_churn_corr, meaningful_corrs)

#0.23 would give key 9 features (+1 "churn") which could produce meaningful clustering
key_feature_corr = df_churn_corr.loc[abs(df_churn_corr['Correlation']) > 0.23]
key_features = list(key_feature_corr["Feature"].values)
key_features.append("Churn")

df_reduced = df[key_features]

#---------------------------------------------------------
#KMeans

def show_kmeans_elbow_range(df, n_list:list):

    wcss= []
    for n in n_list:
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)

    plt.plot(n_list, wcss, marker="o", linestyle="--", alpha=0.7)

    plt.xlabel('N clusters')
    plt.ylabel('WCSS')
    plt.title(f'KMeans Elbow')
    plt.show()

def show_kmneans_clust_sil_score(df, k_list:list):
    
    s_score = []
    for k in k_list:

        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(df)

        score = silhouette_score(df, clusters)
        s_score.append(score)

    plt.plot(k_list, s_score, marker="o", linestyle="--")

    plt.xlabel('N clusters')
    plt.ylabel('Silhouette Score')
    plt.title(f'KMeans Silhuette Score for key features')
    plt.legend(title="PCA components")
    plt.show()

def show_cluster_behaviour(df, n_clust):

    kmeans = KMeans(n_clusters=n_clust, random_state=42)
    kmeans.fit(df)

    #having cluster info as part of df, is there a way to not do that?
    df['Cluster'] = kmeans.fit_predict(df)

    grid = sns.PairGrid(df, hue= "Cluster", palette="Set1", diag_sharey=False, corner=True) #grid_kws={"alpha": 0.8},
    grid.map_diag(sns.histplot, multiple="dodge")
    grid.map_offdiag(sns.scatterplot) #, kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids'
    grid.add_legend()
    plt.show()
 
#show_kmeans_elbow_range(df_reduced, range(2, 15))
#show_kmneans_clust_sil_score(df_reduced, range(2, 15))

show_cluster_behaviour(df_reduced, 4)
#---------------------------------------------------------
#Kmeans + PCA tests

def show_variance_vs_comp_num(df):
    pca = PCA()
    pca.fit_transform(df)
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_) #how much additional components increase variance

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
    plt.xlabel('Number Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Variance against Number of Components')
    plt.show()

def show_kmeans_elbow_PCA_range(df, n_list:list):

    colors = plt.cm.viridis(np.linspace(0, 1, len(n_list)))

    for i, n in enumerate(n_list):
        pca = PCA(n_components=n, random_state=42)
        df_flat = pca.fit_transform(df)

        wcss= []
        for k in range(1,20):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df_flat)
            wcss.append(kmeans.inertia_)

        plt.plot(range(1, 20), wcss, marker="o", linestyle="--", alpha=0.7, color=colors[i],label=f"{n} PCA components")

    wcss= []
    for j in range(1,20):
        kmeans = KMeans(n_clusters=j, random_state=42)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 20), wcss, marker="o", linestyle="--", alpha=0.7, color="black",label=f"No PCA applied")

    plt.xlabel('N clusters')
    plt.ylabel('WCSS')
    plt.title(f'KMeans Elbow for a range of PCA components')
    plt.legend(title="PCA components")
    plt.show()

def show_kmneans_PCA_clust_sil_score(df, pca_list:list, k_list:list):
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(pca_list)))

    for i, n in enumerate(pca_list):
        pca = PCA(n_components=n, random_state=42)
        df_flat = pca.fit_transform(df)

        s_score = []
        for k in k_list:

            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(df_flat)

            score = silhouette_score(df_flat, clusters)
            s_score.append(score)

        plt.plot(k_list, s_score, marker="o", linestyle="--", alpha=0.7, color=colors[i], label=f"{n} PCA components")

    s_score = []
    for j in k_list:

        kmeans = KMeans(n_clusters=j, random_state=42)
        clusters = kmeans.fit_predict(df)

        score = silhouette_score(df, clusters)
        s_score.append(score)

    plt.plot(k_list, s_score, marker="o", linestyle="--", alpha=0.7, color="black",label=f"No PCA applied")

    plt.xlabel('N clusters')
    plt.ylabel('WCSS')
    plt.title(f'KMeans Silhuette Score for a Range of PCA components')
    plt.legend(title="PCA components")
    plt.show()

#show_variance_vs_comp_num(df)

n_PCA=[2, 5, 7, 10, 15, 21]

#show_kmeans_elbow_PCA_range(df, n_PCA)
#show_kmneans_PCA_clust_sil_score(df, n_PCA, range(2, 11))

#---------------------------------------------------------
#PCA action

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

def show_PCA_component_matrix(df, feature):
    df_corr = prepare_corr_data(df, feature)

    notable_features = ["cluster", feature]
    notable_features.extend(df_corr["Feature"].head(3).values)
    notable_features.extend(df_corr["Feature"].tail(3).values)

    grid = sns.PairGrid(df[notable_features].sample(n=1000), hue= "cluster", palette="Set1", diag_sharey=False, corner=True) 
    grid.map_diag(sns.histplot, multiple="dodge")
    grid.map_offdiag(sns.histplot, multiple="stack", discrete=True, shrink=0.8)  
    grid.add_legend()
    plt.show()

n_clust = 5
kmeans = KMeans(n_clusters=n_clust, random_state=42)
df['cluster'] = kmeans.fit_predict(df)

#show_2d_PCA_clusters(df)
#show_3d_PCA_clusters(df)
#show_PCA_component_matrix(df, "Churn")

#---------------------------------------------------------
#Hierarchical plotting

def hierarchical_plotting(df):

    linkage_matrix = linkage(df, method="ward") 
    dendrogram(linkage_matrix)

    cut_clusters = 110

    df["cluster"]  = fcluster(linkage_matrix, cut_clusters, criterion="distance")

    plt.title("Hierarchical clustering dendogram")
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    #plt.axhline(y=cut_clusters, color = "r", linestyle="--", label = f"Cut for {cut_clusters} distance")
    plt.show()

#hierarchical_plotting(df)