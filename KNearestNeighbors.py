import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('auto.csv', )
header = ['Symbolyng','Normalized-losses','make','Fuel-Type','Aspiration','Num-of-doors','Body-style','Drive-wheels',
         'Engine-location', 'Whell-base', 'Lenght', 'Width', 'Weight','Curb-weight','Engine-type','Num-of-cylinders',
         'Engine-size','Fuel-system','Bore','Stroke','Compression-ratio','Horsepower','Peak-rpm','City-mpg','Highway-mpg',
         'Price']
df.columns=header
df.replace("?", np.nan, inplace = True)
df = df.dropna()
x=df.select_dtypes(include=["float64","int64"])
y = (df["Fuel-Type"])

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.90, random_state=0)
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
knn = KNeighborsClassifier(n_neighbors=10, metric='minkowski')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print("%90 Test %10 Train, number of neighbors=10:",f1_score(y_test, y_pred, average='macro'))

##################################################################

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.90, random_state=0)
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print("%90 Test %10 Train, number of neighbors=5:",f1_score(y_test, y_pred, average='macro'))

#################################################################################

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.60, random_state=0)
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
knn = KNeighborsClassifier(n_neighbors=10, metric='minkowski')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print("%60 Test %40 Train, number of neighbors=10:",f1_score(y_test, y_pred, average='macro'))

################################################################

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.70, random_state=0)
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print("%70 Test %30 Train, number of neighbors=5:",f1_score(y_test, y_pred, average='macro'))