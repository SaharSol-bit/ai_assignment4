import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


# Load the dataset
df = pd.read_pickle("wdbc.pkl")

X = df.drop(columns=["id", "malignant"])
y = df["malignant"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#only benign cases to define "normal"
benign = X_train[y_train == 0]

#define feature groups based on domain knowledge and EDA insights
size_columns = ['radius_2','area_2','perimeter_2']
shape_columns = ['concavity_2','compactness_2','concave points_2']
texture_columns = ['texture_2','smoothness_2']
homogeneity_columns = ['symmetry_2','fractal dimension_2']

thresholds = {}
thresholds['size'] = benign[['radius_2','area_2','perimeter_2']].quantile(0.75).mean()
thresholds['shape'] = benign[['concavity_2','compactness_2','concave points_2']].quantile(0.75).mean()
thresholds['texture'] = benign[['texture_2','smoothness_2']].quantile(0.75).mean()
thresholds['homogeneity'] = benign[['symmetry_2','fractal dimension_2']].quantile(0.75).mean()

#define a simple rule-based classifier that flags a case as malignant if any of the following conditions are met:
def rule_classifier(x): 
    size_abnormal= (x[['radius_2','area_2','perimeter_2']].mean(axis=1) > thresholds['size'])

    shape_abnormal = (x[['concavity_2','compactness_2','concave points_2']].mean(axis=1) > thresholds['shape'])

    texture_abnormal = (x[['texture_2','smoothness_2']].mean(axis=1) > thresholds['texture'])

    homogeneity_abnormal = (x[['symmetry_2','fractal dimension_2']].mean(axis=1) > thresholds['homogeneity'])
    return (size_abnormal | shape_abnormal | texture_abnormal | homogeneity_abnormal).astype(int)

#used a random forest as a more complex model for comparison
rf = RandomForestClassifier( n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rule_preds = rule_classifier(X_test)

#a small decision tree to show how a simple model can still perform reasonably well
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
tree_preds = tree.predict(X_test)

#evaluate the models using accuracy and classification report to show precision, recall, and F1-score for each class
print("Rule Model Accuracy:", accuracy_score(y_test, rule_preds))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("Small Tree Accuracy:", accuracy_score(y_test, tree_preds))

#the classification report provides a more detailed breakdown of performance, showing how well each model identifies malignant and benign cases, which is crucial in a medical context where false negatives can have serious consequences.
print("\nRule Model Report\n", classification_report(y_test, rule_preds))
print("\nRandom Forest Report\n", classification_report(y_test, rf_preds))
print("\nSmall Tree Report\n", classification_report(y_test, tree_preds))
