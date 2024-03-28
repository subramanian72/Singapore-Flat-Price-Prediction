import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import joblib

#Load the data
data = pd.read_csv("D:/SingFlatPricePrj/data/ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv")

data = data.drop(['month','block', 'street_name', 'lease_commence_date'], axis='columns')

new_dataset = data.dropna()

s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
# print('No. of. categorical features: ',
#       len(object_cols))

# OH_encoder = OneHotEncoder(sparse_output=False)
# OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
# OH_cols.index = new_dataset.index
# OH_cols.columns = OH_encoder.get_feature_names_out()
# df_final = new_dataset.drop(object_cols, axis=1)
# df_final = pd.concat([df_final, OH_cols], axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

label_encoders = {}

for col in object_cols:
    label_encoders[col] = LabelEncoder()
    new_dataset[col + '_encoded'] = label_encoders[col].fit_transform(new_dataset[col])


print("Finished label encoder")

df_final = new_dataset.drop(object_cols, axis = "columns")

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X = df_final.drop(['resale_price'], axis=1)
y = df_final['resale_price']

# Split the training set into
# training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# #To check the best regressor and best params
# def find_best_model_using_gridsearchcv(X, y):
#   algos = {
#     'Linear_regression': {
#         'model': LinearRegression(),
#         'params': {
#             "n_jobs": [-1],
#             "copy_X" :[True, False],
#             "fit_intercept": [True, False]
#         }
#     },
#     'Gradient_booster':{
#         'model': GradientBoostingRegressor(),
#         'params':{
#               'learning_rate': [0.01,0.02,0.03],
#               'subsample'    : [0.9, 0.5, 0.2],
#               'n_estimators' : [10,20,100],
#               'max_depth'    : [4,6,8]
#         }
#     },
#       'Random-forest':{
#           'model': RandomForestRegressor(),
#           'params': {
#               "n_estimators"      : [10,20,30],
#               "max_features"      : ["sqrt", "log2"],
#               "min_samples_split" : [2,4,8],
#           }
#       }
#   }
#   scores = []
#   for algo_name, config in algos.items():
#     gs = GridSearchCV(config['model'], config['params'], cv=5, return_train_score=False)
#     gs.fit(X,y)
#     scores.append({
#         'model': algo_name,
#         'best_score': gs.best_score_,
#         'best_params': gs.best_params_
#     })
#   return pd.DataFrame(scores, columns=['model','best_score','best_params'])


# #calling the function
# find_best_model_using_gridsearchcv(X_test,Y_test)

#Randomforestregressor model
# rfr_regressor = RandomForestRegressor()

# parameters = {
#                 "n_estimators"      : [10,20,30],
#                 "max_features"      : ["sqrt", "log2"],
#                 "min_samples_split" : [2,4,8]
#               }
# grid_rfr = GridSearchCV(estimator=rfr_regressor, param_grid = parameters, cv = 5, n_jobs=-1)
# grid_rfr.fit(X, y)

model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_test)

# print("Mean absolute percentage", mean_absolute_percentage_error(Y_test, Y_pred))
# print("Mean squared error", mean_squared_error(Y_test, Y_pred))
print("score", model_RFR.score(X_test,Y_test))

#save the model to disk
joblib.dump(model_RFR,"rfrpriceregg.sav")