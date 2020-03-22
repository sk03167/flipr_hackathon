#importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 100, metric='minkowski', p=2)
#importing the dataset
dataset = pd.read_excel('Train_dataset.xlsx')
X = dataset.iloc[: , :-1].values
Y = dataset.iloc[: , 27].values
dfX = pd.DataFrame(X)
dfY = pd.DataFrame(Y)
dfX = dfX.dropna(subset=[8])

#missing data
# Standardize the features
#standardized_features = scaler.fit_transform(features)
# Replace the first feature's first value with a missing value
#true_value = standardized_features[0,0]
#standardized_features[0,0] = np.nan
# Predict the missing values in the feature matrix
#features_knn_imputed = KNN(k=5, verbose=0).complete(standardized_features)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(dfX.iloc[:,16:26])
dfX.iloc[:, 16:26] = imputer.transform(dfX.iloc[:,16:26])
 #here using iloc is important as it does not work directly as with the object type matrix.
dfX[6]=dfX[6].replace(np.nan,0)
dfX[7]=dfX[7].replace(np.nan,'unknown')
dfX[11]=dfX[11].replace(np.nan,'unknown')
dfX[14]=dfX[14].replace(np.nan,'unknown')

dfX[15]=dfX[15].replace(np.nan,'unknown')
 
 #encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
dfX[1]=labelencoder_X.fit_transform(dfX[1])
columns={2, 5, 7, 8, 11, 14, 15}
dfX[2]=labelencoder_X.fit_transform(dfX[2])
dfX[5]=labelencoder_X.fit_transform(dfX[5])
dfX[7]=labelencoder_X.fit_transform(dfX[7])
dfX[7]=dfX[7].replace(9,np.nan)
dfX[8]=labelencoder_X.fit_transform(dfX[8])
imputer = imputer.fit(dfX.iloc[:,7:8])
dfX.iloc[:,7:8] = imputer.transform(dfX.iloc[:,7:8])
dfX[11]=labelencoder_X.fit_transform(dfX[11])
dfX[14]=labelencoder_X.fit_transform(dfX[14])
dfX[15]=labelencoder_X.fit_transform(dfX[15])
imputer = imputer.fit(dfX.iloc[:,11:15])
dfX.iloc[:,11:15] = imputer.transform(dfX.iloc[:,11:15])
plt.scatter(dfX[0],dfX[14])
plt.show()
