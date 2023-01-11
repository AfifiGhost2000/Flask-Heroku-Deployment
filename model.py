# Importing the libraries
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import smote as SMOTE

def get_integer_mapping(le):
    '''
    Return a dict mapping labels to their integer values
    from an SKlearn LabelEncoder
    le = a fitted SKlearn LabelEncoder
    '''
    res = {}
    for cl in le.classes_:
        res.update({cl:le.transform([cl])[0]})

    return res




df = pd.read_csv('toy_dataset.csv')
# mapping = {}
# cols = df[['City','Gender','Illness']]
# for col in cols:
#   mapping[col] = {name: i for i, name in enumerate(df[col].unique())}
# def mapping_func(row):
#   return pd.Series([mapping[col][row[col]] for col in cols])

# X = df.apply(mapping_func, axis=1)
label = preprocessing.LabelEncoder()


#df['City'] = df['City'].fillna( method ='ffill', inplace = True)

#df['Gender'] = df['Gender'].fillna( method ='ffill', inplace = True)


df.fillna(0)

df.drop('Illness', axis=1, inplace=True)

#print(df.Gender.unique())


#df.set_index('Number', inplace=True)

df['City'] = label.fit_transform(df['City'])


integerMappingCity = get_integer_mapping(label)

#print(integerMappingCity['Dallas'])

label = preprocessing.LabelEncoder()


df['Gender'] = label.fit_transform(df['Gender'])

df['Gender'] = df['Gender'].astype('category')
df['Gender'] = df['Gender'].cat.codes

integerMappingGender = get_integer_mapping(label)

X = df[['Number', 'City', 'Gender', 'Age']]


y = df['Income']

print(df)

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)
# fit model no training data
# model = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=500, 
#                            silent=True, objective='reg:linear', nthread=-1, gamma=0,
#                            min_child_weight=1, max_delta_step=0, subsample=1, 
#                            colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, 
#                            reg_lambda=1, scale_pos_weight=1, base_score=0.5, 
#                            seed=0, missing=None, use_label_encoder=False)


lr = LinearRegression()
lr.fit(X_train, y_train)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#make predictions for test data
y_pred = lr.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print(mean_absolute_error(y_test, y_pred))

integerMapping = {**integerMappingCity , **integerMappingGender}

#print(integerMapping['Austin'])

def get_label_encode(x):
    try:
        my_int = int(x)
        return my_int
    except ValueError:
        return int(integerMapping[x])



#print(df.dtypes)

# Saving model to disk
pickle.dump(lr, open('model.pkl','wb'))


