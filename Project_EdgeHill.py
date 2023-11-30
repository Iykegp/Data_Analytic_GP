#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from pandas import set_option
from sklearn import preprocessing
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
import statistics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import svm
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV


# In[2]:


df = pd.read_csv("Cost_Tracker.csv", sep=',', encoding = 'unicode_escape')


# In[3]:


#print the dataset
df.head()


# In[4]:


# to count number of rows and columns
df.shape


# In[5]:


# to know items on the columns
df.columns


# # Data Type For Each Attribute

# In[6]:


# get the counts and Datatype
df.info()


# In[ ]:





# # Descriptive Statistics

# In[7]:


# Statistical Summary
# from pandas import read_csv
# from pandas import set_option

set_option('display.width', 100)
#pd.set_option("display.max_columns", 100)
set_option('display.precision', 3)
#display.precision


description = df.describe()
print(description)


# In[ ]:





# In[8]:


#counting for each column how many null values there are and produce a new dataframe 
df.isnull().sum()


# In[9]:


df.describe()


# In[10]:


# #df['COST_320'].fillna(df_train['COST_320'].mean(),inplace=True)
# df.get('COST_320')
# df


# In[11]:


df.columns


# 

# #### DATA PREPROCESSING

# In[12]:


df.get('month')


# In[13]:


df.get('DAY')


# In[14]:


df.get('km')


# In[15]:


df.get('Pallet_Qty')


# In[16]:


df.get('HL')


# In[17]:


df.get('BIT_ALLOWANCE')


# In[18]:


df['BIT_ALLOWANCE'].fillna(df['BIT_ALLOWANCE'].mean(),inplace=True)


# In[19]:


df.isnull().sum()


# In[20]:


df.columns = df.columns.str.strip()
df['COST_820'].describe()


# In[21]:


df['COST_320'].fillna(df['COST_320'].mean(),inplace=True)
df['COST_399'].fillna(df['COST_399'].mean(),inplace=True)
df['COST_782'].fillna(df['COST_782'].mean(),inplace=True)
df['COST_795'].fillna(df['COST_795'].mean(),inplace=True)
df['COST_815'].fillna(df['COST_815'].mean(),inplace=True)
df['COST_820'].fillna(df['COST_820'].mean(),inplace=True)


# In[22]:


df.isnull().sum()


# In[23]:


df['Ship_Owner_Name'].describe()


# In[24]:


df['Ship_Owner_Name'].fillna(df['Ship_Owner_Name'].mode()[0],inplace=True)


# In[25]:


df.isnull().sum()


# In[26]:


df.get('VAT_0.075')


# 

# In[27]:


# df = df.dropna(subset=['Load_Number'])
#df = df.dropna()
df.drop(['Load_Number','Stock_Code'],axis=1,inplace=True)
df.drop(['Invoice_Date'],axis=1,inplace=True)
df.drop(['DAY'],axis=1,inplace=True)


# In[28]:


df.drop(['Ship_Register'],axis=1,inplace=True)


# In[29]:


df.isnull().sum() 


# In[30]:


# from the above we can see that the we have dropped the irrevalant columns and the missing values have been taking care of


# In[31]:


df


# In[32]:


df.get('Pallet_Qty')


# In[33]:


df.get('VAT_0.075')


# # Exploratory Data Analysis

# In[34]:


# Combined Histogram and KDE plot
plt.figure(figsize =(10,7 ))
sns.histplot(df, x="LOCATION", kde=True, bins=30)

# Rotate the x-axis labels for readability
plt.xticks(rotation=70)

plt.title("Combined Histogram and KDE Plot of Location in Nigeria")
plt.show()


# In[ ]:





# In[35]:


sns.displot(df['HL'])
plt.show()


# In[36]:


sns.distplot(df['Qty_Invoiced'])


# In[37]:


plt.figure(figsize =(10,8 ))
sns.distplot(df['COST_320'])


# In[38]:


# Combined Histogram and KDE plot
plt.figure(figsize =(10,8 ))
sns.histplot(df, x="COST_399", kde=True, bins=30)
plt.title("Combined Histogram and KDE Plot of COST OF DISEAL @ 399/LITRE in Nigeria")
plt.show()


# In[39]:


# Combined Histogram and KDE plot
plt.figure(figsize =(10,8 ))
sns.histplot(df, x="COST_782", kde=True, bins=30)
plt.title("Combined Histogram and KDE Plot of COST OF DISEAL @ 782/LITRE in Nigeria")
plt.show()


# In[40]:


# Combined Histogram and KDE plot
plt.figure(figsize =(10,8 ))
sns.histplot(df, x="COST_795", kde=True, bins=30)
plt.title("Combined Histogram and KDE Plot of COST OF DISEAL @ 795/LITRE in Nigeria")
plt.show()


# In[41]:


# Combined Histogram and KDE plot
plt.figure(figsize =(10,8 ))
sns.histplot(df, x="COST_815", kde=True, bins=30)
plt.title("Combined Histogram and KDE Plot of COST OF DISEAL @ 815/LITRE in Nigeria")
plt.show()


# In[42]:


plt.figure(figsize =(10,8 ))
sns.distplot(df['COST_820'])


# In[43]:


# sns.displot(df, x="month")


# In[44]:


month_counts = df["month"].value_counts()
fig = px.pie(month_counts, values=month_counts.values, names=month_counts.index, title="Pie Chart of Month")
fig.show()


# In[45]:


# sns.displot(df, x="Pallet_Qty")
# Pie chart
Pallet_counts = df["Pallet_Qty"].value_counts()
fig = px.pie(Pallet_counts, values=Pallet_counts.values, names=Pallet_counts.index, title="Pie Chart of Pallet_Quantity")
fig.show()


# In[46]:


# sns.kdeplot(data = df, x= "LOCATION", shade = True)
# plt.title("Density plot")
# plt.xlabel("Values")
# plt.ylabel("Density")
# plt.show()

# g = sns.catplot(data=df, x="LOCATION", y="COST_820", kind="violin", inner=None)
# sns.swarmplot(data=df, x="LOCATION", y="COST_820", color="k", size=1, ax=g.ax)


# In[47]:


# Stacked Bar Plot
plt.figure(figsize =(10,8))
class_embarked = df.groupby(["BAND", "Pallet_Qty"]).size().unstack()
class_embarked_prop = class_embarked.div(class_embarked.sum(axis=1), axis=0)
class_embarked_prop.plot(kind="bar", stacked=True)
plt.title("Stacked Bar Plot of Pallet_size vs. Band")
plt.ylabel("Proportion")
plt.show()


# In[ ]:





# In[ ]:





# In[48]:


# fig = px.scatter_3d(df, x='LOCATION', y='km', z='COST_820', color='HL')
# fig.show()
# # Helix equation
# # t = np.linspace(0, 10, 50)
# # x, y, z = np.cos(t), np.sin(t), t

# # fig = go.Figure(data=[go.Scatter3d(x='LOCATION', y='km', z='COST_820',
# #                                    mode='markers')])
# # fig.show()


# In[49]:


# # create a scatter3d chart
# fig = go.Figure(data=[go.Scatter3d(x=df['Pallet_Qty'], y=df['km'], z=df['HL'], mode='markers')])

# # show the chart
# fig.show()


# In[ ]:





# In[50]:


# Select columns for x, y, z, and color
x = df['LOCATION']
y = df['km']
z = df['COST_820']
colors = df['HL']

# Create the trace for the scatter plot
trace = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=5,
        color=colors,  # set colors
        colorscale='Viridis',  # set the colorscale
        opacity=0.8
    )
)

# Create the layout for the plot
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='STATES'),
        yaxis=dict(title='KILOMETERS'),
        zaxis=dict(title='COST')
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

# Create the figure and add the trace and layout
fig = go.Figure(data=[trace], layout=layout)

# Show the plot
fig.show()


# In[51]:


# Define the helix equation
def helix(t):
    x = np.cos(t)
    y = np.sin(t)
    z = t / (2 * np.pi)
    return x, y, z

# Calculate the color values based on the helix equation
t = np.linspace(0, 10 * np.pi, len(df))
colors = t / (2 * np.pi)

# Select columns for x, y, and z
x = df['LOCATION']
y = df['km']
z = df['COST_820']

# Create the trace for the scatter plot
trace = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=5,
        color=colors,  # set colors
        colorscale='Viridis',  # set the colorscale
        opacity=0.8
    )
)

# Create the layout for the plot
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='X Axis'),
        yaxis=dict(title='Y Axis'),
        zaxis=dict(title='Z Axis')
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

# Create the figure and add the trace and layout
fig = go.Figure(data=[trace], layout=layout)

# Show the plot
fig.show()


# In[52]:


# Create a bar chart of categorical data
counts = df['LOCATION'].value_counts()
plt.bar(counts.index, counts.values)

# # Create a horizontal bar chart of the categorical data
# plt.bar(counts.index, counts.values, color='blue')

# Set the title and labels for the chart
plt.title('Location Data Visualization')
plt.xlabel('Location')
plt.ylabel('Counts')

# Show the chart
plt.show()


# In[53]:


# Group the data by category and count the number of observations in each category
counts = df.groupby('LOCATION')['Customer_Name'].count()

# Create a bar chart of the categorical data
plt.bar(counts.index, counts.values, color='blue')

# Set the title and labels for the chart
plt.title('Categorical Data Visualization')
plt.xlabel('Categories')
plt.ylabel('Counts')

# Add a legend
plt.legend(['Counts'])

# Rotate the x-axis labels for readability
plt.xticks(rotation=70)

# Show the chart
plt.show()


# In[ ]:





# # Coorelation Matrix

# In[54]:


# Pairwise Pearson correlations

set_option('display.width', 1000)
set_option('display.precision', 2)


correlations = df.corr(method='pearson')
print(correlations)


# In[55]:


set_option('display.width', 100)

plt.figure(figsize=(20,10))
#sns.heatmap(df.corr())

sns.heatmap(df.corr(), annot = True)


# In[56]:


df.corr()


# In[57]:


# Sometimes the correlation coefficients mayhave too many floating digits. 
# As such one can Reduce the decimal places to improve readability using the argument 
# fmt = '.3g'or fmt = ‘.1g' because by default the function displays two digits after the 
# decimal (greater than zero) i.e fmt='.2g'



# In[58]:


plt.figure(figsize=(16,5))
sns.heatmap(df.corr(), annot = True, vmin=-1, vmax=1, center= 0, fmt='.2g')


# In[59]:


# To change the shape use the NumPy methods; .triu() and .tril() 
# and then specify the Seaborn heatmap argument called mask=

# .triu() is a method in NumPy that returns the lower triangle of any matrix given to it, 
# while .tril() returns the upper triangle of any matrix given to it.


# In[ ]:





# In[60]:


matrix = np.triu(df.corr())
plt.figure(figsize=(16,5))
sns.heatmap(df.corr(), annot=True, mask=matrix)


# In[61]:


mask = np.tril(df.corr())
plt.figure(figsize=(16,5))
sns.heatmap(df.corr(), annot=True, mask=mask)


# In[62]:


skew = df.skew()
print(skew)


# In[ ]:





# # Understand the Data With Visualization

# In[63]:


#Univariate Histograms


# In[64]:


df.hist()
plt.gcf().set_size_inches(20,20)
plt.show()


# In[65]:


# Univariate Density Plots


# In[66]:


df.plot(kind='density', subplots=True, layout=(3,4), sharex=False)
plt.gcf().set_size_inches(20,20)
plt.show()


# In[67]:


# Box and Whisker Plots

df.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)
plt.gcf().set_size_inches(20,20)
plt.show()


# In[68]:


from pandas.plotting import scatter_matrix
scatter_matrix(df)
plt.gcf().set_size_inches(20,20)
plt.show()


# In[ ]:





# In[69]:


# sns.set(style = 'ticks', color_codes=True)
# # sns.pairplot(df, hue='month',  vars=['km','Pallet_Qty','COST_820','COST_815', 'COST_795','COST_782','COST_320', 'COST_399'])
# sns.pairplot(df, hue='month', vars=['km','Pallet_Qty','BAND','HL', 'COST_820'])


# In[ ]:





# In[70]:


df


# In[ ]:





# In[ ]:





# In[ ]:





# 

# # Prepare Data For Machine Learning

# 

# In[71]:


df.describe()


# In[72]:


#from the data described, it is not normally distributed. this we have to correct by scalling it properly
#X = df.iloc[:,0:len(df.columns)-1]


# # LABEL ENCODING

# In[73]:


columns_to_be_encoded = ['Ship_Owner_Name','Customer_Name','Qty_Invoiced','LOCATION','BAND','VAT_0.075']  # list of column names you want encoded

# Instantiate the encoders
encoders = {column: LabelEncoder() for column in columns_to_be_encoded}

for column in columns_to_be_encoded:
    df[column] = encoders[column].fit_transform(df[column])


# In[74]:


df


# In[75]:


df.columns


# ### Rescale Data

# In[76]:


array = df.values
# separate array into input and output components
X = array[:,0:16]
Y = array[:,16] # We are predicting the cost_820
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
set_printoptions(precision=3)

#print(rescaledX[0:5,:])
# print(rescaledX[0:5,:])
print(rescaledX)


# In[ ]:





# In[77]:


X


# In[78]:


Y


# In[79]:


rescaledXDF = pd.DataFrame(rescaledX, columns = ['month','Ship_Owner_Name', 'Customer_Name', 'Qty_Invoiced', 'HL', 'LOCATION', 'BAND', 'km',
       'Pallet_Qty', 'COST_320', 'COST_399', 'COST_782', 'COST_795', 'COST_815', 'BIT_ALLOWANCE', 'VAT_0.075'])


# In[80]:


rescaledXDF.head()


# In[81]:


rescaledXDF


# In[82]:


rescaledXDF.describe()


# # Standardize Data

# In[83]:


array = rescaledXDF.values
# separate array into input and output components
X = array[:,0:15] 
#Y = array[:,15]
scaler = StandardScaler().fit(X)
reStandardX = scaler.transform(X)
# summarize transformed data
#set_printoptions(precision=3)
#print(rescaledX[0:5,:])

print(reStandardX)


# In[84]:


reStandardXDF = pd.DataFrame(reStandardX, columns = ['Ship_Owner_Name', 'Customer_Name', 'Qty_Invoiced', 'HL', 'LOCATION', 'BAND', 'km',
       'Pallet_Qty', 'COST_320', 'COST_399', 'COST_782', 'COST_795', 'COST_815', 'BIT_ALLOWANCE','VAT_0.075'])


# In[85]:


reStandardXDF


# In[86]:


reStandardXDF.describe()


# In[87]:


reStandardXDF.hist()
plt.gcf().set_size_inches(20,20)
plt.show()


# In[ ]:





# In[ ]:





# # Normalization

# In[88]:


#array = standardX_df.values
# separate array into input and output components
array = reStandardXDF.values
X = array[:,0:14]
#Y = array[:,14]
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=5)
print(normalizedX[0:5,:])  


# In[89]:


normDF = pd.DataFrame(normalizedX)


# In[90]:


normDF.columns = ['month','Ship_Owner_Name', 'Customer_Name', 'Qty_Invoiced', 'HL', 'LOCATION', 'BAND', 'km',
    'Pallet_Qty', 'COST_320', 'COST_399', 'COST_782', 'COST_795', 'COST_815']


# In[91]:


normDF


# In[ ]:





# In[92]:


normDF.describe()


# In[ ]:





# In[ ]:





# In[93]:


import matplotlib.pyplot as plt
normDF.hist()
plt.gcf().set_size_inches(20,20)
plt.show()


# In[94]:


DfModel = normDF


# 

# In[95]:


df


# In[96]:


Y


# In[97]:


X


# In[98]:


DfModel


# In[99]:


Y


# # Training the Model

# In[100]:


X=DfModel.drop(columns=['COST_320','COST_399','COST_782','COST_795','COST_815'],axis=1)
Y=df['COST_820'].astype('int')

# splitting both input and output in the train and test dataset
test_size = 0.30
seed = 4
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
random_state=42) # using a radomizer will help the model to predict accurately

# Modelling phase:invoking our algorithm and passing both training and test data
model = LinearRegression()
model.fit(X_train, Y_train) #Training the model: model is learning
result = model.score(X_test, Y_test) # Testing the model to see how much it has learnt

# Cross Validation Regression R^2
scoring = 'r2'
results = cross_val_score(model, X, Y, cv=5, scoring=scoring)

print('The LineaRegression efficiency is:',result*100)
print("R^2: {} ({})".format(results.mean()*100, results.std()*100))
print("Standard deviation :",statistics.stdev(results)*100)# The closer the standard deviation to zero is the better for the model
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")


# In[101]:


num_folds = 5 #splitting of the data into equal folders
seed = 7 # number of randomizer
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

model = LinearRegression()
# model = LogisticRegression(max_iter=10000)
model.fit(X, Y)

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring) # no need for training as they happen implicitly inside the cross val score


print(results)
print(statistics.mean(results)*100)
print("Decision Tree Accuracy: {}".format(results.mean()*100.0))
print("Standard deviation :",statistics.stdev(results)*100)# The closer the standard deviation to zero is the better for the model


# In[ ]:





# In[ ]:





# In[102]:


num_folds = 5 #splitting of the data into equal folders
seed = 7 # number of randomizer
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

model = DecisionTreeRegressor()
model.fit(X, Y)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring) # no need for training as they happen implicitly inside the cross val score

# Cross Validation Regression R^2
scoring = 'r2'
results = cross_val_score(model, X, Y, cv=5, scoring=scoring)

print('DecisionTreeRegressor efficiency:',results)
print(statistics.mean(results)*100)
print("Accuracy: {}".format(results.mean()*100.0))
print("Standard deviation :",statistics.stdev(results))# The closer the standard deviation to zero is the better for the model
print("R^2: {} ({})".format(results.mean()*100, results.std()*100))

coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Decision Tree Feature Importance")


# In[103]:


num_folds = 5 #splitting of the data into equal folders
seed = 7 # number of randomizer
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

model = RandomForestRegressor(n_estimators=1000)
model.fit(X, Y)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring) # no need for training as they happen implicitly inside the cross val score

# Cross Validation Regression R^2
scoring = 'r2'
results = cross_val_score(model, X, Y, cv=5, scoring=scoring)

print(results)
print(statistics.mean(results)*100)
print("Accuracy: {}".format(results.mean()*100.0))
print("Standard deviation :",statistics.stdev(results))# The closer the standard deviation to zero is the better for the model
print("R^2: {} ({})".format(results.mean()*100, results.std()*100))

coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="RandomForestRegressor Feature Importance")


# In[104]:


num_folds = 5 #splitting of the data into equal folders
seed = 7 # number of randomizer
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

model = ExtraTreesRegressor()
model.fit(X, Y)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring) # no need for training as they happen implicitly inside the cross val score

# Cross Validation Regression R^2
scoring = 'r2'
results = cross_val_score(model, X, Y, cv=5, scoring=scoring)

print(results)
print(statistics.mean(results)*100)
print("Accuracy: {}".format(results.mean()*100.0))
print("Standard deviation :",statistics.stdev(results)*100)# The closer the standard deviation to zero is the better for the model
print("R^2: {} ({})".format(results.mean()*100, results.std()*100))

coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="ExtraTreesRegressor Feature Importance")


# In[ ]:





# In[105]:


num_folds = 5 #splitting of the data into equal folders
seed = 7 # number of randomizer
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

regressor = SVR(kernel = 'linear')
# regressor.fit(X, Y)
results = cross_val_score(regressor, X, Y, cv=kfold, scoring=scoring) # no need for training as they happen implicitly inside the cross val score

# Cross Validation Regression R^2
scoring = 'r2'
results = cross_val_score(regressor, X, Y, cv=5, scoring=scoring)

print(results)
print(statistics.mean(results)*100)
print("Accuracy: {}".format(results.mean()*100.0))
print("Standard deviation :",statistics.stdev(results))# The closer the standard deviation to zero is the better for the model
print("R^2: {} ({})".format(results.mean()*100, results.std()*100))


# In[106]:


num_folds = 5 #splitting of the data into equal folders
seed = 7 # number of randomizer
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

model = XGBRegressor()
model.fit(X, Y)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring) # no need for training as they happen implicitly inside the cross val score

# Cross Validation Regression R^2
scoring = 'r2'
results = cross_val_score(model, X, Y, cv=5, scoring=scoring)

print(results)
print(statistics.mean(results)*100)
print("Accuracy: {}".format(results.mean()*100.0))
print("Standard deviation :",statistics.stdev(results))# The closer the standard deviation to zero is the better for the model
print("R^2: {} ({})".format(results.mean()*100, results.std()*100))

coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="XGBRegressor Feature Importance")


# # hyperparameter Tunning

# In[107]:


# Define linear regression model
linreg = LinearRegression()

# Define the hyperparameter space
param_grid = {'fit_intercept': [True, False], 'positive': [True, False]}

# Perform grid search with cross-validation
grid_search = GridSearchCV(linreg, param_grid, cv=5)
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Evaluate the performance on the test set with the best hyperparameters
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, Y_test)
print("Test score with best hyperparameters:", test_score*100)


# In[108]:


# Define Extra Trees Regression model
et = ExtraTreesRegressor(random_state=42)

# Define the hyperparameter distribution
param_dist = {
    'n_estimators': np.arange(10, 200, 10),
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform randomized search with cross-validation
random_search = RandomizedSearchCV(et, param_distributions=param_dist, cv=5, n_iter=50, random_state=42)
random_search.fit(X_train, Y_train)

# Print the best hyperparameters
print("Best hyperparameters:", random_search.best_params_)

# Evaluate the performance on the test set with the best hyperparameters
best_model = random_search.best_estimator_
test_score = best_model.score(X_test, Y_test)
print("Test score with best hyperparameters:", test_score*100)


# In[109]:


test_score


# In[110]:


# Define Decision Tree Regression model
dt = DecisionTreeRegressor(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(dt, param_grid=param_grid, cv=5)
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Evaluate the performance on the test set with the best hyperparameters
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, Y_test)

print("Test score with best hyperparameters:", test_score)


# In[111]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

# # Load the Boston housing dataset
# X, y = load_boston(return_X_y=True)


# Define the parameter grid to search over
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'max_features': ['sqrt', 'log2']
}

# Create a random forest regressor object
rf_reg = RandomForestRegressor()

# Use grid search to find the best parameters
grid_search = GridSearchCV(estimator=rf_reg, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, Y)

# Print the best parameters and best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", -grid_search.best_score_)

# Use the best model to make predictions
best_model = grid_search.best_estimator_
Y_pred = best_model.predict(X)
mse = mean_squared_error(Y, Y_pred)
print("MSE: ", mse)
# Calculate the percentage improvement compared to a baseline MSE of 50 million
baseline_mse = 50000000
improvement = (baseline_mse - mse) / baseline_mse * 100
print("Percentage improvement: ", np.round(improvement, 2), "%")


# In[112]:


from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
# import pandas as pd

# Define the Extra Trees Regressor model
model = ExtraTreesRegressor()

# Define the hyperparameter grid for tuning
param_grid = {'n_estimators': [100, 200, 300, 400, 500],
              'max_features': [1.0, 'sqrt', 'log2'],
              'max_depth': [None, 5, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}

# Define the Grid Search object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

# Fit the Grid Search object to the training data
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters found by Grid Search
print("Best hyperparameters: ", grid_search.best_params_)

# Make predictions on the test data using the best model
Y_pred = grid_search.predict(X_test)

# Print the evaluation metrics
print("MSE: ", mean_squared_error(Y_test, Y_pred))
print("R2 score: ", r2_score(Y_test, Y_pred))


# In[113]:


import xgboost as xgb
# Define the XGB Regressor model
model = xgb.XGBRegressor(objective='reg:squarederror')

# Define the hyperparameter grid for tuning
param_grid = {'n_estimators': [100, 200, 300, 400, 500],
              'max_depth': [3, 4, 5, 6, 7],
              'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
              'colsample_bytree': [0.3, 0.5, 0.7],
              'gamma': [0, 0.1, 0.2, 0.3, 0.4]}

# Define the Grid Search object
XGBOOST_grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

# Fit the Grid Search object to the training data
XGBOOST_grid_search.fit(X_train, Y_train)

# Print the best hyperparameters found by Grid Search
print("Best hyperparameters: ", XGBOOST_grid_search.best_params_)

# Make predictions on the test data using the best model
Y_pred = XGBOOST_grid_search.predict(X_test)

# Print the evaluation metrics
print("MSE: ", mean_squared_error(Y_test, Y_pred))
print("R2 score: ", r2_score(Y_test, Y_pred))


# In[114]:


import joblib


# In[115]:


new_data = {'month':0.79,'Ship_Owner_Name':0.06, 'Customer_Name':-0.19, 'Qty_Invoiced':0.27, 'HL':0.27, 'LOCATION': -2.26e-01, 'BAND':-0.04, 'km':-0.04,
    'Pallet_Qty':0.27, 'COST_320':-0.06, 'COST_399':-7.78e-02, 'COST_782':-0.12, 'COST_795':-0.12, 'COST_815':-0.12}
index=[0]
cost_df = pd.DataFrame(new_data,index)


# In[116]:


cost_df


# In[117]:


# Save the trained model to a file
joblib.dump(XGBOOST_grid_search, 'CostModel.pkl')


# In[118]:


# Define the filename to save the trained model
model_filename = 'CostModel.pkl'


# In[119]:


import tkinter as tk
import joblib

# joblib.dump = (XGBOOST_grid_search,CostModel.pki)

# Load the trained model
XGBOOST_grid_search = joblib.load('CostModel.pkl')

# Create the GUI
window = tk.Tk()
window.title("XGBOOST Regressor Model")

# Add input fields for user input
tk.Label(window, text="BAND").grid(row=0)
tk.Label(window, text="Pallet_Qty").grid(row=1)
tk.Label(window, text="Qty_Invoiced").grid(row=2)
tk.Label(window, text="LOCATION").grid(row=3)
tk.Label(window, text="km").grid(row=4)

BAND = tk.Entry(window)
BAND.grid(row=0, column=1)

Pallet_Qty = tk.Entry(window)
Pallet_Qty.grid(row=1, column=1)

Qty_Invoiced = tk.Entry(window)
Qty_Invoiced.grid(row=2, column=1)

LOCATION = tk.Entry(window)
LOCATION.grid(row=3, column=1)

km = tk.Entry(window)
km.grid(row=4, column=1)

# Define the prediction function
def predict():
    # Get the user input
    X = [float(BAND.get()), float(Pallet_Qty.get()), float(Qty_Invoiced.get()), float(LOCATION.get()), float(km.get())]

    # Make the prediction
    Y_pred = CostModel.predict([X])

    # Display the prediction
    tk.Label(window, text="Prediction: " + str(Y_pred[0])).grid(row=6)

# Add a button for making predictions
tk.Button(window, text="Predict", command=predict).grid(row=5)


# Start the GUI
window.mainloop()


# In[ ]:




