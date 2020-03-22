import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from impyute.imputation.cs import fast_knn
from sklearn import datasets
import seaborn as sns

dataset = pd.read_excel('Train_dataset.xlsx')

dataset['Comorbidity']=dataset['Comorbidity'].fillna(method='pad')
dataset['Mode_transport']=dataset['Mode_transport'].fillna(method='pad')
dataset['Occupation']=dataset['Occupation'].fillna(method='pad')
dataset['Children']=dataset['Children'].fillna(method='pad')
dataset['cardiological pressure']=dataset['cardiological pressure'].fillna(method='pad')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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


onehotencoder = OneHotEncoder(categorical_features =[3])
x = onehotencoder.fit_transform(x).toarray()

correlation = dataset.corr(method='pearson')
columns = correlation.nlargest(10, 'Infect_Prob').index
columns

correlation_map = np.corrcoef(dataset[columns].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlation_map, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns.values, xticklabels=columns.values)

plt.show()