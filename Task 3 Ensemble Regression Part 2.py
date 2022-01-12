import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_excel("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Ensemble Techniques/Coca_Rating_Ensemble.xlsx")

df.columns

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True)
plt.show()

df.isnull().sum()
df.columns
df.dropna(axis=0)
# Dummy variables
df.head()
df.info()

df= df.drop(["Bean_Type"],axis =1)
df= df.drop(["Origin"],axis =1)

lb = LabelEncoder()
df["Company"] = lb.fit_transform(df["Company"])
df["Name"] = lb.fit_transform(df["Name"])
df["Review"] = lb.fit_transform(df["Review"])
df["Company_Location"] = lb.fit_transform(df["Company_Location"])


# Input and Output Split
predictors = df.loc[:, df.columns!="Rating"]
type(predictors)

target = df["Rating"]
type(target)

"Stacking"
# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=42)

estimators = [
    ('lr', RidgeCV()),
    ('svr', LinearSVR(random_state=42))
]

reg = StackingRegressor(
    estimators=estimators,
    final_estimator=RandomForestRegressor(n_estimators=10,
                                          random_state=42)
)

reg.fit(x_train, y_train).score(x_test, y_test) #-0.195


"Voting"

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
r1 = LinearRegression()
r2 = RandomForestRegressor(n_estimators=10, random_state=1)
er = VotingRegressor([('lr', r1), ('rf', r2)])
print(er.fit(predictors, target).predict(predictors)) #[3.50614177 3.24197456 3.18824146 ... 3.35151055 3.33372764 3.22221436]

help(StackingRegressor)
