import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler

dataset = pd.read_excel('Train_dataset.xlsx')

dataset['comorbidity']=dataset['comorbidity'].fillna(method='pad')
dataset['Mode_transport']=dataset['Mode_transport'].fillna(method='pad')
dataset['Occupation']=dataset['Occupation'].fillna(method='pad')
dataset['Children']=dataset['Children'].fillna(method='pad')
dataset['cardiological pressure']=dataset['cardiological pressure'].fillna(method='pad')

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

#encoding_columns=['Gender', 'Designation','Married','Occupation','Mode_transport','comorbidity','Pulmonary score','cardiological pressure' ]
#dataset([encoding_columns]) = labelencoder_X.fit_transform(dataset(encoding_columns))
dataset['Region'] = labelencoder_X.fit_transform(dataset['Region'])
dataset['Gender'] = labelencoder_X.fit_transform(dataset['Gender'])
dataset['Designation'] = labelencoder_X.fit_transform(dataset['Designation'])
dataset['Married'] = labelencoder_X.fit_transform(dataset['Married'])
dataset['Occupation'] = labelencoder_X.fit_transform(dataset['Occupation'])
dataset['Mode_transport'] = labelencoder_X.fit_transform(dataset['Mode_transport'])
dataset['comorbidity'] = labelencoder_X.fit_transform(dataset['comorbidity'])
dataset['Pulmonary score'] = labelencoder_X.fit_transform(dataset['Pulmonary score'])
dataset['cardiological pressure'] = labelencoder_X.fit_transform(dataset['cardiological pressure'])


dataset = dataset.drop(['Name'], axis = 1)

scaler = StandardScaler()
standardized_features = scaler.fit_transform(dataset.iloc[:,1:26])
#use the above to standardize all columns
#standardized_features = scaler.fit_transform(dataset[['Age', 'Coma score','Diuresis', 'Platelets','HBB','d-dimer','Heart rate','HDL cholesterol', 'Charlson Index','Insurance','salary']])
#dataset[['Age', 'Coma score','Diuresis', 'Platelets','HBB','d-dimer','Heart rate','HDL cholesterol', 'Charlson Index','Insurance','salary']]=standardized_features

features_knn_imputed = KNN(k=100, verbose=0).fit_transform(standardized_features)

dataset.iloc[:,1:26]=features_knn_imputed

correlation = dataset.iloc[:,1:].corr(method='pearson')
columns = correlation.nlargest(25, 'Infect_Prob').index

correlation_map = np.corrcoef(dataset[columns].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlation_map, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns.values, xticklabels=columns.values)

plt.show()

X = dataset[columns]
Y = X['Infect_Prob'].values
X = X.drop('Infect_Prob', axis = 1).values

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.20, random_state=42)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300)
regressor.fit(X_train,Y_train)
y_pred = regressor.predict(X_test)
error = y_pred-Y_test

t_dataset = pd.read_excel('Test_dataset.xlsx')
t_dataset['Region'] = labelencoder_X.fit_transform(t_dataset['Region'])
t_dataset['Gender'] = labelencoder_X.fit_transform(t_dataset['Gender'])
t_dataset['Designation'] = labelencoder_X.fit_transform(t_dataset['Designation'])
t_dataset['Married'] = labelencoder_X.fit_transform(t_dataset['Married'])
t_dataset['Occupation'] = labelencoder_X.fit_transform(t_dataset['Occupation'])
t_dataset['Mode_transport'] = labelencoder_X.fit_transform(t_dataset['Mode_transport'])
t_dataset['comorbidity'] = labelencoder_X.fit_transform(t_dataset['comorbidity'])
t_dataset['Pulmonary score'] = labelencoder_X.fit_transform(t_dataset['Pulmonary score'])
t_dataset['cardiological pressure'] = labelencoder_X.fit_transform(t_dataset['cardiological pressure'])
t_dataset = t_dataset.drop(['Name'], axis = 1)

t_dataset.iloc[:,1:26] = scaler.fit_transform(t_dataset.iloc[:,1:26])
columnst = columns.delete(0)
Xt = t_dataset[columnst]
solution = regressor.predict(Xt)
solution1 = pd.DataFrame(solution)
solution1 = solution1.join(t_dataset.iloc[:,0])
solution1.rename(columns = {0:'Infec_Prob'}, inplace=True)
columns_titles = ["people_ID","Infec_Prob"]
solution1=solution1[["people_ID","Infec_Prob"]]
solution1.to_csv('solution1.csv')