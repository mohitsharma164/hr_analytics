import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


employee_df = pd.read_csv("D:/01 DS ML DL & Predictive Modeling/DS for Business_6 Case Studies/1 HR\HR Data & Slides/Human_Resources.csv")
#employee_df.info()
#print(employee_df.describe())
print(employee_df['Age'].mean())
employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['Over18'] = employee_df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)
employee_df['OverTime'] = employee_df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)

#sns.heatmap(employee_df.isnull(), yticklabels= False, cbar=False, cmap='Blues')
#employee_df.hist(bins=10, figsize=(30,30), color = 'red')
#plt.show()
employee_df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)
#print(employee_df)
#print(employee_df)
left_df=employee_df[employee_df['Attrition']==1]
stayed_df=employee_df[employee_df['Attrition']==0]
print(len(employee_df))
print(len(left_df))
print(len(stayed_df))
print("% of employees left", (len(left_df)/len(employee_df))*100, "%")

# correlations = employee_df.corr()
# f, ax = plt.subplots(figsize=(20,20))
# sns.heatmap(correlations, annot=True)

f, ax = plt.subplots(figsize=(20,20))
#sns.countplot(x='Age', hue='Attrition', data=employee_df)
#sns.kdeplot(left_df['DistanceFromHome'], label='Employees who left', shade=True, color='r')
#sns.kdeplot(stayed_df['DistanceFromHome'], label='Employees who stayed back', shade=True, color='g')

#sns.boxplot(x='MonthlyIncome', y = 'Gender', data=employee_df)
#sns.boxplot(x='MonthlyIncome', y = 'JobRole', data=employee_df)
#plt.show()
employee_df.info()
#print(employee_df)


X_cat = employee_df[['BusinessTravel', 'Department', 'EducationField', 'Gender',  'JobRole', 'MaritalStatus']]
#print(X_cat)


import sklearn
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()
print(X_cat)
print(X_cat.shape)
X_cat = pd.DataFrame(X_cat)
print(X_cat)
X_num = employee_df[['Age', 'Attrition', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', \
                     'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',\
                     'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',\
                     'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',\
                     'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole' , 'YearsSinceLastPromotion', 'YearsWithCurrManager']]

X_all = pd.concat([X_cat, X_num], axis=1)
print(X_all)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X_all)
print(X)

y = employee_df['Attrition']
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# model = LogisticRegression()
# model.fit(X_train,y_train)
# y_pred = model.predict(X_test)
# print(y_pred)
#
from sklearn.metrics import  confusion_matrix, classification_report
# print('Accuracy {} % '.format(100*accuracy_score(y_pred, y_test)))
#
# cm = confusion_matrix(y_pred, y_test)
# sns.heatmap(cm, annot=True)
# #plt.show()
# print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier()
#model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# cm = confusion_matrix(y_pred, y_test)
# sns.heatmap(cm, annot=True)
# plt.show()

import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=500, activation='relu', input_shape=(51,)))
model.add(tf.keras.layers.Dense(units=500, activation='relu'))
model.add(tf.keras.layers.Dense(units=500, activation='relu'))
model.add(tf.keras.layers.Dense(units=500, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

print(model.summary())

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
epochs_hist = model.fit(X_train, y_train,epochs=100, batch_size=50)

y_pred = model.predict(X_test)
print(y_pred)
y_pred = (y_pred>0.5)
print(y_pred)
#plt.plot(epochs_hist.history['loss'])
#plt.plot(epochs_hist.history['accuracy'])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)

plt.show()
