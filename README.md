# Data analysis for Telco with a focus on customer churn

## Table Of Contents
*   [Description](#description)
*   [Set-up](#set-up)
*   [Progress](#progress)
*   [Results](#results)
*   [Credits](#credits)

## Description
This project is to practice matplotlib library, creating informative graphs and training different classification models. I aimed to practice data analysis and make informed decisions for ML training.
I prepared the data, analysed correlations, tested the PCA approach, used several methods with notable absolute correlation to find the optimal amount of features, used several methods to find the optimal amount of clusters, clusterised and made practical business-oriented suggestions. Lastly, I have tested several ML classification models to find the most optimal one to predict if the customer is likely to churn.

## Set-up
The only setup is to install the same version of libraries I've used. To do that you can use the requirements.txt file. In the terminal, you could run the command ```$ pip install -r requirements.txt``` or install the packages manually with the versions provided in the file.
> [!WARNING]
> You must have **pip installed** to install these packages!

## Progress
### To-Do List ✅

#### Preparation
- [X] Clean data  
- [X] Convert categorical data to numerical
- [X] ~~Normalise~~ Standardised the data for now instead

#### Exploratory Data Analysis (EDA)

- [X] Visualise distributions
- [X] Create a correlation matrix  
- [X] Analyse client behaviour and focus on tendencies for leaving

#### Unsupervised Machine Learning (clustering)
- [X] Get the optimal cluster number using KNN
- [X] Do hierarchical clustering and get a dendrogram

#### Supervised Machine Learning (prognosis)
- [X] Train DT model
- [X] Train RF model
- [X] Train SVM model
- [X] Train some more models (optional)
- [X] Compare using ~~RMSE, R2 score~~ f1 score (because this is classification, not recursion) and cross-validation. Find the most suitable model
- [ ] Test StratifiedKFold for potentially improving the results

#### Interpretation and Analysis
- [X] Analyse and interpret clusters' behaviours
- [X] Analyse and interpret ~~prognosis~~ classification
- [X] Get the variables the most responsible for client churn
- [X] Present business recommendations to improve client retention

- [X] Write a report with steps taken, conclusions, practical suggestions
- [X] Add pretty pictures of graphs✨

## Results

To be added...

## Credits
Data for the project came from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Me, ChatGPT, Gemini, My Cat and Jesus 🙏 (it got so challenging I went to church for the first time in 15 years.. and finalised the project the same day)

Do you have a suggestion on how I could improve this project? Maybe you have some ideas on how I could improve my model performance. Or maybe you've noticed a flaw in key feature selection? Please fork or do get in touch and introduce me to your ideas!
