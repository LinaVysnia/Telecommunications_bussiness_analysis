import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_raw = pd.read_csv("Customer-numerical_raw.csv")
df = pd.read_csv("Customer-churn-reduced-scaled-clusterised.csv")

n_clusters = len(df["Cluster"].unique())
features = list(df.columns)

features.remove("Cluster")

df_selected = df_raw[features]
df_selected["Cluster"] = df["Cluster"]

relevant_features =  ['Contract_Month-to-month', 'OnlineSecurity_No', 'TechSupport_No', 
                      'InternetService_Fiber optic', 'PaymentMethod_Electronic check', 
                      'OnlineBackup_No', 'DeviceProtection_No', 'Contract_Two year', 
                      'tenure', 'Churn', 'Cluster']

features_to_plot = relevant_features
features_to_plot.remove("tenure")
features_to_plot.remove("Cluster")

# features_to_plot=['Churn', 'Cluster']

cluster_colors_int = {
        0: 'red',       # Cluster 0 - Red
        1: 'green',     # Cluster 1 - Green
        2: 'blue',      # Cluster 2 - Blue
        3: 'orange'     # Cluster 3 - Orange
    }

def plot_binary_features_by_cluster_count(df, features, cluster_colors, cluster_column='Cluster'):
    
    df_melted = pd.melt(df, id_vars=[cluster_column], value_vars=features, var_name='Feature', value_name='Value')

    g = sns.FacetGrid(df_melted, col="Feature", height=3, aspect=0.8)

    g.map_dataframe(sns.countplot, "Value", hue=cluster_column, palette=cluster_colors)
    g.set_axis_labels("Value", "Count")
    g.set_titles(template="{col_name}", size=9)
    g.add_legend(title=cluster_column)

    g.figure.tight_layout()
    g.figure.subplots_adjust(left=0.055, right=0.95, top=0.8)
    g.figure.suptitle("Distribution of Features by Cluster", fontsize=16)

    plt.show()

def plot_binary_features_by_cluster_percentage(df, features, cluster_colors, cluster_column='Cluster'):

    df_melted = pd.melt(df, id_vars=[cluster_column], value_vars=features, var_name='Feature', value_name='Value')

    df_grouped = (
        df_melted.groupby([cluster_column, "Feature", "Value"])
        .size()
        .reset_index(name="Count")
    )

    df_totals = df_grouped.groupby([cluster_column, "Feature"])["Count"].sum().reset_index(name="Total")
    df_grouped = df_grouped.merge(df_totals, on=[cluster_column, "Feature"])
    df_grouped["Percentage"] = df_grouped["Count"] / df_grouped["Total"]

    g = sns.FacetGrid(df_grouped, col="Feature", height=3, aspect=0.8)
    g.map_dataframe(sns.barplot, x="Value", y="Percentage", hue=cluster_column, palette=cluster_colors)
    g.set_axis_labels("Feature Value", "Percentage")
    g.set_titles(template="{col_name}", size=9)
    g.add_legend(title="Cluster")


    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Adjust layout
    g.figure.tight_layout()
    g.figure.subplots_adjust(left=0.055, right=0.95, top=0.8)
    g.figure.suptitle("Feature Distribution by Cluster (Normalized)", fontsize=16)

    plt.show()

def show_cluster_tenure(df, cluster_colors, cluster_column='Cluster'):

    plt.figure(figsize=(10, 6))  # Adjust figure size if needed
    sns.boxplot(x=cluster_column, y='tenure', data=df, palette=cluster_colors)
    ax = sns.boxplot(x=cluster_column, y='tenure', data=df, palette=cluster_colors)

    plt.title('Tenure Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Tenure')

    labels = [f'Cluster {cluster}' for cluster in df[cluster_column].unique()]
    handles = [plt.Rectangle((0,0),1,1, color=cluster_colors[str(cluster)]) for cluster in df[cluster_column].unique()]
    
    plt.legend(handles, labels, title="Clusters", bbox_to_anchor=(1.3, 1))
    plt.show()

def show_cluster_tenure(df, cluster_colors, cluster_column='Cluster'):

    plt.figure(figsize=(10, 6))  # Adjust figure size if needed
    sns.boxplot(x=cluster_column, y='tenure', data=df, palette=cluster_colors)
    ax = sns.boxplot(x=cluster_column, y='tenure', data=df, palette=cluster_colors)

    plt.title('Tenure Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Tenure')

    labels = [f'Cluster {cluster}' for cluster in df[cluster_column].unique()]
    handles = [plt.Rectangle((0,0),1,1, color=cluster_colors[str(cluster)]) for cluster in df[cluster_column].unique()]
    
    plt.legend(handles, labels, title="Clusters", bbox_to_anchor=(1.3, 1))
    plt.show()

#plot_binary_features_by_cluster_percentage(df_selected, features_to_plot, cluster_colors_int)
#plot_binary_features_by_cluster_percentage(df_selected, features_to_plot, cluster_colors_int)

#boxplot wont accept integers for pallete, they have to be converted to string
cluster_colors_str = {
        "0": 'red',       # Cluster 0 - Red
        "1": 'green',     # Cluster 1 - Green
        "2": 'blue',      # Cluster 2 - Blue
        "3": 'orange'     # Cluster 3 - Orange
    }

#show_cluster_tenure(df_selected, cluster_colors_str)