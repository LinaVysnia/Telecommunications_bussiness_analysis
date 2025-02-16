import pandas as pd
import seaborn as sns
import matplotlib as plt
import numpy as np

from scipy.stats import randint, uniform, reciprocal

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

#Decision tree would handle all the dimensionality pretty well
df = pd.read_csv("Customer-numerical_raw.csv")

results = df.pop("Churn")

X_train, X_test, y_train, y_test = train_test_split(df, results, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#=========================================================
#Decision tree prediction model

#===================
#Finding best params

# param_grid = {
#     'max_depth': [5, 6, 7 ,8 ,9],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2, 3],
#     'max_features': ['sqrt', 'log2', None],
#     'criterion': ['gini', 'entropy']
# }

# dt = DecisionTreeClassifier(random_state=42)

# grid = GridSearchCV(dt, param_grid, cv=5, scoring='f1', n_jobs=-1)

# grid.fit(X_train, y_train)

# #best_dt = grid.best_estimator_

# #test 1 using f1
# best_dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=2,random_state=42)
# # Accuracy: 0.7739872068230277
# # F1-score: 0.7756494893324082

# #test 2 using accuracy
# #best_dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=2, random_state=42)
# # Accuracy: 0.7732764747690121
# # F1-score: 0.7725883751066696

#===================
#Training the model

best_dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=2,random_state=42)

best_dt.fit(X_train, y_train)
y_pred = best_dt.predict(X_test)

print("Best Decision Tree Model: ", best_dt)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))

#=========================================================
#Random forest prediction model

#===================
#Finding best params

# param_grid = {
#     'n_estimators': [100, 200, 400],
#     #'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],  # Splitting criteria
#     'max_depth': [5, 10, 20, 30],
#     'min_samples_split': [2, 5, 10, 20],
#     'min_samples_leaf': [2, 5, 10],
#     #'max_features': ['sqrt', 'log2', None],     # Number of features to consider at each split
#     #'bootstrap': [True, False]                  # Whether to use bootstrapping
# }

param_distributions = {
    'n_estimators': randint(50, 200),  # Randomly sample between 50 and 200
    'max_depth': [10,20,30,40,50], # None or random integer between 10 and 50
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy'],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}

# rf = RandomForestClassifier(random_state=42)

# n_iter = 100  # Number of random combinations to try (adjust this)
# random_search = RandomizedSearchCV(rf, param_distributions, n_iter=n_iter, cv=5, scoring='f1', n_jobs=-1, verbose=1)
# random_search.fit(X_train, y_train)

# best_rf = random_search.best_estimator_

#Test 1
#best_rf = RandomForestClassifier(class_weight='balanced', max_depth=np.int64(39),
#          max_features='log2', min_samples_leaf=4,min_samples_split=3, n_estimators=148, random_state=42)
# Accuracy: 0.7540867093105899
# F1-score: 0.7635032080257796

#test 2 
# RandomForestClassifier(class_weight='balanced_subsample', max_depth=10,
#                        max_features='log2', min_samples_leaf=4,
#                        min_samples_split=8, n_estimators=62, random_state=42)
# Accuracy: 0.7491115849324804
# F1-score: 0.760556945261212

#test 3
# RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy',
#                        max_depth=10, max_features='log2', min_samples_leaf=3,
#                        min_samples_split=4, n_estimators=171, random_state=42)
# Accuracy: 0.7469793887704336
# F1-score: 0.7582071482471747

#===================
#Training the model

best_rf = RandomForestClassifier(class_weight='balanced', max_depth=np.int64(39),
          max_features='log2', min_samples_leaf=4,min_samples_split=3, n_estimators=148, random_state=42)

best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)

print("Best Random Forest Model: ", best_rf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))

#=========================================================
#Support vector machine model
#===================
#Finding best params

# param_distributions = {
#     'C': reciprocal(1, 100),  # Reduced range for C (1 to 100) - MOST IMPORTANT
#     'kernel': ['rbf'],  # Focus on rbf kernel initially - VERY IMPORTANT
#     'gamma': ['scale', 'auto'] + list(reciprocal(0.001, 0.1).rvs(5)),  # Reduced range and fewer samples for gamma - IMPORTANT
#     'class_weight': [None, 'balanced']  # Keep class_weight, but it's less computationally intensive
# }
# svc = SVC(random_state=42)

# n_iter = 100  
# random_search = RandomizedSearchCV(svc, param_distributions, n_iter=n_iter, cv=5, scoring='f1', n_jobs=-1, verbose=1)  # Use F1-score for classification
# random_search.fit(X_train, y_train)

# best_svc = random_search.best_estimator_

#test 1
#best_svc = SVC(C=np.float64(1.309057363357464), class_weight='balanced',
#     gamma=np.float64(0.023141125053224965), random_state=42)
# Accuracy: 0.7334754797441365
# Best F1-score: 0.6275991221199372

#test 2
#best_svc = SVC(C=np.float64(1.1332667250594013), class_weight='balanced', random_state=42)
# Accuracy: 0.7327647476901208
# Best F1-score: 0.6266565410903586

#===================
#Training the model
best_svc = SVC(C=np.float64(1.309057363357464), class_weight='balanced', gamma=np.float64(0.023141125053224965), random_state=42)

best_svc.fit(X_train, y_train)

y_pred = best_svc.predict(X_test)

print("Best SVC Model: ", best_svc)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Best F1-score:", f1_score(y_test, y_pred, average='weighted'))

#=========================================================
#Gradient boosting classifier model
#===================
#Finding best params

# param_distributions = {
#     'n_estimators': randint(50, 300),  # Number of boosting stages (trees)
#     'learning_rate': uniform(0.01, 0.3),  # Contribution of each tree
#     'max_depth': randint(3, 10),  # Maximum depth of the individual trees
#     'min_samples_split': randint(2, 20),  # Minimum samples to split a node
#     'min_samples_leaf': randint(1, 10),  # Minimum samples in a leaf
#     'subsample': uniform(0.6, 1.0),  # Fraction of samples used for fitting the trees
#     'max_features': ['sqrt', 'log2', None],  # Number of features to consider for splits
#     'criterion': ['friedman_mse', 'squared_error'], # Function to measure the quality of a split.
# }

# gb_clf = GradientBoostingClassifier(random_state=42)

# n_iter = 100 
# random_search = RandomizedSearchCV(gb_clf, param_distributions, n_iter=n_iter, cv=5, scoring='f1', n_jobs=-1, verbose=1)

# random_search.fit(X_train, y_train)

# best_gb_clf = random_search.best_estimator_

#test 1
# Best Gradient Boosting Classifier: GradientBoostingClassifier(criterion='squared_error',
#                            learning_rate=np.float64(0.03307275938883212),
#                            max_depth=4, min_samples_leaf=2, min_samples_split=3,
#                            n_estimators=195, random_state=42,
#                            subsample=np.float64(0.9823357224951418))
# Best F1-score: 0.5972449377862121

#test 2
# Best Gradient Boosting Classifier: 
GradientBoostingClassifier(learning_rate=np.float64(0.047628255111097854),
                           max_depth=4, max_features='log2', min_samples_leaf=9,
                           n_estimators=156, random_state=42,
                           subsample=np.float64(0.6836865782688305))
# Best F1-score: 0.5982972141243003


#===================
#Training the model
best_gb_clf = GradientBoostingClassifier(learning_rate=np.float64(0.047628255111097854),
                           max_depth=4, max_features='log2', min_samples_leaf=9,
                           n_estimators=156, random_state=42,
                           subsample=np.float64(0.6836865782688305))

best_gb_clf.fit(X_train, y_train)
y_pred = best_gb_clf.predict(X_test)

print("Best Gradient Boosting Classifier:", best_gb_clf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Best F1-score:", f1_score(y_test, y_pred, average='weighted'))

y_pred = best_gb_clf.predict(X_test)
