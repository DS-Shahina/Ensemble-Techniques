import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

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

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)

###############################################################################

########################### Bagging ############################################

from sklearn import tree
regtree = tree.DecisionTreeRegressor()
from sklearn.ensemble import BaggingRegressor


bag_reg = BaggingRegressor(base_estimator = regtree, n_estimators = 500,
                            bootstrap = True, n_jobs = 1, random_state = 42)

bag_reg.fit(x_train, y_train)


from sklearn.metrics import mean_squared_error ,r2_score

# Evaluation on Testing Data
# getting error value
mean_squared_error(y_test, bag_reg.predict(x_test)) # 0.183
# R^2 score value
r2_score(y_test, bag_reg.predict(x_test)) # 0.118

# Evaluation on Training Data

# getting error value
mean_squared_error(y_train, bag_reg.predict(x_train)) #0.025
# R^2 score value
r2_score(y_train, bag_reg.predict(x_train)) #0.890

# overfitting case

"AdaBoostRegressor"

from sklearn.ensemble import AdaBoostRegressor

Ada_reg = AdaBoostRegressor(base_estimator = regtree, n_estimators = 500, random_state = 42)

Ada_reg.fit(x_train, y_train)


from sklearn.metrics import mean_squared_error ,r2_score

# Evaluation on Testing Data
# getting error value
mean_squared_error(y_test, Ada_reg.predict(x_test)) # 0.207
# R^2 score value
r2_score(y_test, Ada_reg.predict(x_test)) # 0.002 - not good

# Evaluation on Training Data

# getting error value
mean_squared_error(y_train, Ada_reg.predict(x_train)) #4.35
# R^2 score value
r2_score(y_train, Ada_reg.predict(x_train)) #0.999

# overfitting case

"GradientBoostingRegressor"

from sklearn.ensemble import GradientBoostingRegressor

Gra_reg = GradientBoostingRegressor(n_estimators = 500, random_state = 42, max_depth = 3,learning_rate = 0.02 )

Gra_reg.fit(x_train, y_train)


from sklearn.metrics import mean_squared_error ,r2_score

# Evaluation on Testing Data
# getting error value
mean_squared_error(y_test, Gra_reg.predict(x_test)) # 0.186
# R^2 score value
r2_score(y_test, Gra_reg.predict(x_test)) # 0.106 

# Evaluation on Training Data

# getting error value
mean_squared_error(y_train, Gra_reg.predict(x_train)) #0.143
# R^2 score value
r2_score(y_train, Ada_reg.predict(x_train)) #0.999

# overfitting case

"xgboost"

from xgboost import XGBRegressor

xg_reg = XGBRegressor(n_estimators = 500, random_state = 42, max_depth = 3,learning_rate = 0.02 )

xg_reg.fit(x_train, y_train)


from sklearn.metrics import mean_squared_error ,r2_score

# Evaluation on Testing Data
# getting error value
mean_squared_error(y_test, xg_reg.predict(x_test)) # 0.187
# R^2 score value
r2_score(y_test, xg_reg.predict(x_test)) # 0.100

# Evaluation on Training Data

# getting error value
mean_squared_error(y_train, xg_reg.predict(x_train)) #0.145
# R^2 score value
r2_score(y_train, xg_reg.predict(x_train)) #0.377

# overfitting case

param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0,9],
               'rag_alpha': [1e-2, 0.1, 1]}

"Grid Search"
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(xg_reg, param_test1, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(x_train, y_train)

print("r2 / variance : ", grid_search.best_score_ ) #nan
print("Residual sum of squares: %.2f" % np.mean((grid_search.predict(x_test) - y_test) ** 2)) # 0.18

grid_search.best_params_


"{'colsample_bytree': 0.8,
 'gamma': 0.1,
 'max_depth': 3,
 'rag_alpha': 0.01,
 'subsample': 0.8}"





