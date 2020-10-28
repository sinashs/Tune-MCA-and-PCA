#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load wrangling libraries
import numpy as np
import pandas as pd
from pathlib import Path
pd.set_option('max_columns', 30)  # Show all columns
pd.set_option('max_colwidth', 100)

#Load Machine Learning libraries
from sklearn.model_selection import train_test_split, KFold
from prince import MCA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import itertools
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')


# In[2]:


def match(X_train_cat, X_test_cat):
    
    '''
    Description: This function checks the categorical features 
    between training and testing sets. If it detects dimension mismatch, it will make each categorical variables in 
    the training and testing tests have the same number of levels.
    input : 
        X_train_cat: dataframe, categorical predictors of training set
        X_test_cat: dataframe, categorical predictors of testing set
    output:
        X_train_cat, X_test_cat = dataframes , 
            having number of levels same as each other
    
    '''
    
    keep = X_train_cat.nunique() == X_test_cat.nunique()
    X_train_cat = X_train_cat[X_train_cat.columns[keep]]
    X_test_cat = X_test_cat[X_test_cat.columns[keep]]

    # For categorical features that have same levels, make sure the classes are the same
    keep = []
    for i in range(X_train_cat.shape[1]):
        keep.append(all(np.sort(X_train_cat.iloc[:,i].unique()) == np.sort(X_test_cat.iloc[:,i].unique())))
    X_train_cat = X_train_cat[X_train_cat.columns[keep]]
    X_test_cat = X_test_cat[X_test_cat.columns[keep]]
    
    return X_train_cat, X_test_cat


# In[3]:


def seperation(X):
    """
    Description : This functions separates the features into numerical and categorical
    input :
        X : dataframe
    output :
        X_cat : dataframe , categorical features
        X_num : dataframe, numerical features
    
    """
    X_cat = X.select_dtypes(include = 'object')
    X_num = X.select_dtypes(exclude = 'object')
    
    return X_cat, X_num


# ## Data Profile

# In[4]:


# Load data set
file_path = Path('D:\SFSU\job\Data Analyst\RAPP\RAPP_car_data.csv')
data = pd.read_csv(file_path , index_col = None)
data = data.drop(['id','symboling'], axis = 1)

# Show data, its dimensions, and types
display(data.head())
print(data.shape)
display(data.dtypes)


# In[5]:


# Display the missing values within each column
data.isnull().sum()


# In[6]:


# Show the data distribution in normalized-losses
data['normalized-losses'].describe()


# In[7]:


# To avoid considering outliers in our dataset, replace the missing values in the 'normalized-losses' column with its median value
data['normalized-losses'].fillna(data['normalized-losses'].dropna().median(), inplace = True)

# Drop other missing values
data.dropna(inplace = True)


# In[8]:


# Show unique values in each categorical column
data.select_dtypes(include = 'object').apply(lambda x : set(x))


# In[9]:


# Define the Predictors and the response value
X = data.copy()

y = X['price']
del X['price']


# In[10]:


# Split the data into training and testing sest
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = .2, random_state = 862)


# In[11]:


# Tune PCA and MCA

pca_n_components = range(5,20) #set range of numerical components to iterate
mca_n_components = range(2,8) #set range of categorical components to iterate


scaler = StandardScaler () #instantiate scaler

params = list(itertools.product(pca_n_components, mca_n_components))

validation_mse = []

#Five-fold cross validation

for ind, param in enumerate(params):
    kfold = KFold(10,True, ind)
    temporary_mse = []
    
    for train_index, valid_index in kfold.split(X_train):
        X_training, Y_training = X_train.iloc[train_index], y_train.iloc[train_index] # training set
        X_validation, Y_validation = X_train.iloc[valid_index], y_train.iloc[valid_index] # validation set
        
        # Seperate categorical and numerical features
        X_training_cat, X_training_num = seperation(X_training)
        X_validation_cat, X_validation_num = seperation(X_validation)
        
        # Make same level categorical features of both training and validation 
        X_training_cat, X_validation_cat = match(X_training_cat, X_validation_cat)

        # Scale numerical data
        scaler.fit(X_training_num) #fit to training data
        X_training_num_S = pd.DataFrame(scaler.transform(X_training_num)) #transform both training and validation data
        X_validation_num_S = pd.DataFrame(scaler.transform(X_validation_num))
        X_training_num_S.columns = X_training_num.columns.values
        X_validation_num_S.columns = X_training_num.columns.values
        
        # Numerical Feature reduction
        pca = PCA(n_components = param[0])
        X_training_num_r = pca.fit_transform(X_training_num)     
        X_validation_num_r = pca.transform(X_validation_num) #transform validation set
        
        # Categorical Feature Reduction
        mca = MCA(n_components = param[1])
        X_training_cat_r = mca.fit_transform(X_training_cat)
        X_validation_cat_r = mca.transform(X_validation_cat) #transform validation set

        # Combine categorical and numerical together
        X_training = pd.concat([X_training_cat_r, X_training_num_r] , axis = 1 )
        X_validation = pd.concat([X_validation_cat_r, X_validation_num_r] , axis = 1 )
                
        # Perform regression fitting
        model = LinearRegression()
        model.fit(X_training, Y_training)
        
        temporary_mse.append(mean_squared_error(Y_validation, model.predict(X_validation)))
    
    validation_mse.append(np.mean(temporary_mse))   
        


# In[12]:


#Print best parameters,MSE, and RMSE
best_combination = params[np.argmin(validation_mse)]
print ('Best parameters:', best_combination)
print ('MSE:', (np.min(validation_mse)))
print ('RMSE:', (np.sqrt(np.min(validation_mse))))


# In[ ]:




