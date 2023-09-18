import pandas as pd

train = pd.read_csv("train.csv")

import numpy as np

train['label_2'].fillna(train['label_2'].mean(), inplace=True)
train['label_2'] = train['label_2'].astype(int)
train.dropna(subset=['label_4'], inplace=True)

lbl1_train = train[train.columns.difference(['label_2', 'label_3', 'label_4'])]
lbl2_train = train[train.columns.difference(['label_1', 'label_3', 'label_4'])]
lbl3_train = train[train.columns.difference(['label_1', 'label_2', 'label_4'])]
lbl4_train = train[train.columns.difference(['label_1', 'label_2', 'label_3'])]

from sklearn import svm

def train_model(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = svm.SVC(random_state=42)
    model.fit(X, y)

    print(f"Training score: {model.score(X, y)}")
    print(f"Testing score: {model.score(X, y)}")

    return model

model1 = train_model(lbl1_train)
model2 = train_model(lbl2_train)
model3 = train_model(lbl3_train)
model4 = train_model(lbl4_train)

test = pd.read_csv("test.csv")

test = test[test.columns.difference(['label_1', 'label_2', 'label_3', 'label_4'])]

test['label_1'] = model1.predict(test[test.columns.difference(['label_1', 'label_2', 'label_3', 'label_4'])])
test['label_2'] = model2.predict(test[test.columns.difference(['label_1', 'label_2', 'label_3', 'label_4'])])
test['label_3'] = model3.predict(test[test.columns.difference(['label_1', 'label_2', 'label_3', 'label_4'])])
test['label_4'] = model4.predict(test[test.columns.difference(['label_1', 'label_2', 'label_3', 'label_4'])])

test.head()

test.to_csv('test_with_predictions.csv', index=False)