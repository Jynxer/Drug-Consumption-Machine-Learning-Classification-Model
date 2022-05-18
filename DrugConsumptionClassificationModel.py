import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, RFECV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, precision_recall_curve, roc_auc_score, roc_curve

"""
Data loading
"""
# Load drug consumption data
df = pd.read_csv("./drug_consumption.csv")

# Split DataFrame into input features and output classes
inputs = df.loc[:, 'age':'sensationSeeking']
output = df.loc[:, 'nicotineConsumption']

"""
Check if data is clean
"""
# Check for null or NaN entries
dfFull.isnull().sum().sum() + dfFull.isna().sum().sum()

# Check for duplicate records
bool_series = dfFull.duplicated().to_numpy()
bool_series.sum()

"""
Label encoding
"""
# Encode output classes with ordinal labelling
def encodeLabelsForBinary(label):
    if (label=='CL0' or label=='CL1'):
        return(0)
    else:
        return(1)

def encodeLabelsForMulticlass(label):
    if (label=='CL0'):
        return(0)
    elif (label=='CL1'):
        return(1)
    elif (label=='CL2'):
        return(2)
    elif (label=='CL3'):
        return(3)
    elif (label=='CL4'):
        return(4)
    elif (label=='CL5'):
        return(5)
    elif (label=='CL6'):
        return(6)

output = output.apply(encodeLabelsForMulticlass)

"""
Data analysis and visualisation
"""
df.describe()

# Visualising dataset imbalance
df = pd.concat([inputs, output], axis=1)
seaborn.catplot(x='nicotineConsumption', data=df, kind='count')

# Visualising feature distributions
meltedFeatures = inputs.copy()
meltedFeatures = pd.melt(meltedFeatures)
seaborn.boxplot(x="value", y="variable", data=meltedFeatures)

# Visualising all pairwise bivariate distributions between features and nicotineConsumption classification
seaborn.pairplot(df, hue='nicotineConsumption', height=2.5)

# Visualising feature correlation
correlation = df.corr()
plt.figure(figsize=(10, 10))
seaborn.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8})

"""
Train and test split
"""
# Split the dataset into training/validation and testing subsets
X_train, X_test, y_train, y_test = train_test_split(inputs, output, test_size=0.3, shuffle=True, random_state=42)

"""
Polynomial expansion
"""
applyPolynomialExpansion = False
if (applyPolynomialExpansion):
    poly = PolynomialFeatures(degree=2)
    X_train_numpy_array = poly.fit_transform(X_train)
    X_test_numpy_array = poly.fit_transform(X_test)
    features = ['age', 'gender', 'education', 'country', 'ethnicity', 'neuroticism', 'extraversion', 'opennessToExperience', 'agreeableness', 'conscientiousness', 'impulsiveness', 'sensationSeeking']
    newColumns = poly.get_feature_names(features)
    X_train = pd.DataFrame(X_train_numpy_array, columns=newColumns)
    X_test = pd.DataFrame(X_test_numpy_array, columns=newColumns)

"""
Data preprocessing
"""
# Normalise input data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_numpy_array = scaler.transform(X_train)
X_test_numpy_array = scaler.transform(X_test)
X_train = pd.DataFrame(X_train_numpy_array, columns=X_train.columns)
X_test = pd.DataFrame(X_test_numpy_array, columns=X_test.columns)

"""
Create LogisticRegression
"""
# Instantiate sklearn LogisticRegression model
model = LogisticRegression(max_iter=1000)

"""
Feature subset selection
"""
# print("Discarded SelectKBest method")
# Feature selection via univariate tests
# Optimal k found to be 7 after testing model at all k between 12 and 1
# k = 7
# selectKBest = SelectKBest(k=k).fit(X_train, y_train)
# X_train_subset = selectKBest.transform(X_train)

# Set X_train and X_test to new DataFrames including the K best features
# dfScores = pd.DataFrame(selectKBest.scores_)
# dfColumns = pd.DataFrame(X_train.columns)
# featureScores = pd.concat([dfColumns, dfScores], axis=1)
# featureScores.columns = ['Feature', 'Score']
# bestKFeatures = featureScores.nlargest(k, 'Score')['Feature'].to_numpy()
# X_train = X_train.loc[:, X_test.columns.isin(bestKFeatures)]
# X_test = X_test.loc[:, X_test.columns.isin(bestKFeatures)]

# Feature subset selection via recursive feature elimination with cross validation
rfecv = RFECV(estimator = model, step=1, cv=5, scoring='accuracy')
rfecv = rfecv.fit(X_train, y_train)
print('Optimal number of features: {}'.format(rfecv.n_features_))
optimalFeatureSubset = X_train.columns[rfecv.support_]
print('Optimal feature subset: {}'.format(optimalFeatureSubset))
X_train = rfecv.transform(X_train)
X_test = rfecv.transform(X_test)

"""
Hyperparameter optimisation using GridSearchCV
"""
# Set of hyperparameters to optimise
param_grid = [
    {
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-4, 4, 40),
        'solver': ['liblinear'],
        'max_iter': [100, 1000, 2500, 5000],
        'multi_class': ['ovr'],
        'class_weight': ['balanced']
    }
]

# Find best classifier hyperparameters
classifier = GridSearchCV(model, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
bestClassifier = classifier.fit(X_train, y_train)

"""
Model training
"""
# Set model as best estimator found by grid search
model = classifier.best_estimator_

# Train LogisticRegression model
model.fit(X_train, y_train)

"""
Model evaluation
"""
# Set k fold cross validation parameters
kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# Calculate cross validation score of best model
results = cross_val_score(model, X_train, y_train, cv=kfold)

# Print model k fold cross validation accuracy
print("K fold cross validation accuracy: {}% ({}%)".format(results.mean()*100, results.std()*100))

# Print model training accuracy
print("Training accuracy: {}%".format(model.score(X_train, y_train)*100))

# Print model testing accuracy
print("Testing accuracy: {}%".format(model.score(X_test, y_test)*100))

# Calculate model predictions of testing data
y_hat = model.predict(X_test)

# Decode class labels
classLabels = ['CL0', 'CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6']
y_hat = pd.Series(y_hat)

def decodeClassLabelsForBinary(label):
    if (label==0 or label==1):
        return('Non-user')
    else:
        return('User')

def decodeClassLabelsForMulticlass(label):
    return(classLabels[label])
        
# y_test = y_test.apply(decodeClassLabelsForMulticlass)
# y_hat = y_hat.apply(decodeClassLabels)

# Calculate the confusion matrix
dfConfusionMatrix = pd.DataFrame(confusion_matrix(y_test, y_hat), index=classLabels, columns=classLabels)
seaborn.heatmap(dfConfusionMatrix, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# Display classification evaluation metrics
print(classification_report(y_test, y_hat, zero_division=0))
print("Micro average precision = {}".format(round(precision_score(y_test, y_hat, average='micro', zero_division=0), 2)))
print("Micro average recall = {}".format(round(recall_score(y_test, y_hat, average='micro', zero_division=0), 2)))

# Visualise confidence scores for test data
confidence = model.decision_function(X_test)
# fig, ax = plt.subplots(figsize=(10, 100))
seaborn.heatmap(confidence)

# Visualise probability estimates for test data
probEstimates = model.predict_proba(X_test)
# fig, ax = plt.subplots(figsize=(10, 100))
seaborn.heatmap(probEstimates)

# Plot the Precision-Recall curves for each class of nicotine consumption
precision = dict()
recall = dict()
y_test_for_curve = y_test.copy()
for i in range(7):
    precision[i], recall[i], _ = precision_recall_curve(y_test_for_curve, probEstimates[:, i], pos_label=i)
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")

# Calculate the arithmetic mean of areas under the receiver operating characteristic curves
print(roc_auc_score(y_test, probEstimates, multi_class='ovr', average='macro'))

# Calculate the weighted average of areas under the receiver operating characteristic curves
print(roc_auc_score(y_test, probEstimates, multi_class='ovr', average='weighted'))

# Plot the Receiver Operating Characteristic (ROC) curves for each class of nicotine consumption
falsePositiveRate = dict()
truePositiveRate = dict()

for i in range(7):
    falsePositiveRate[i], truePositiveRate[i], _ = roc_curve(y_test_for_curve, probEstimates[:, i], pos_label=i)
    plt.plot(falsePositiveRate[i], truePositiveRate[i], lw=2, label='class {}'.format(i))

plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend(loc="best")
plt.title("ROC curve")

"""
Final results
"""
"""
K fold cross validation accuracy:
    40.49%

Training accuracy:
    39.73%

Testing accuracy:
    40.64%

Model parameters:
    - C = 0.007017038286703823
    - multi_class = 'ovr'
    - penalty = 'l1'
    - solver = 'liblinear'
    - class_weight = 'balanced'
"""