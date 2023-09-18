from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

train = pd.read_csv("train.csv")

train['label_2'].fillna(train['label_2'].mean(), inplace=True)
train['label_2'] = train['label_2'].astype(int)
train.dropna(subset=['label_4'], inplace=True)


X = train.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y = train['label_4']

clf = RandomForestClassifier()
clf.fit(X, y)

importances = clf.feature_importances_

f_importances = pd.Series(importances, X.columns)

f_importances.sort_values(ascending=False, inplace=True)

f_importances.plot(x='Features', y='Importance',
                   kind='bar', figsize=(16, 9), rot=45)

plt.tight_layout()
plt.show()

importances = clf.feature_importances_

f_importances = pd.Series(importances, X.columns)

f_importances.sort_values(ascending=False, inplace=True)

threshold = 0.004

selected_features = f_importances[f_importances > threshold]

X_selected = X[selected_features.index]


def separate_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = svm.SVC(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    print(f"Training score: {model.score(X_train, y_train)}")
    print(f"Testing score: {model.score(X_test, y_test)}")


def evaluate_model_detailed(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


X_train, X_test, y_train, y_test = separate_dataset(X_selected, y)

model4 = train_model(X_train, y_train)
evaluate_model(model4, X_train, X_test, y_train, y_test)
evaluate_model_detailed(model4, X_test, y_test)

test = pd.read_csv("test.csv")
test = test[test.columns.difference(
    ['label_1', 'label_2', 'label_3', 'label_4'])]
test = test[selected_features.index]
test['label_4'] = model4.predict(test)
test.head()

test.to_csv('test_with_predictions_label4.csv', index=False)
