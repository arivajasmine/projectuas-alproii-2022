#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import pandas as pd


# In[65]:


mamikos = r"E:/FILE PERKULIAHAN/SMT 3/Algoritma Pemrograman II/Project UAS/train.csv"


# In[66]:


df=pd.read_csv(mamikos)


# In[67]:


df.describe()


# In[68]:


df.columns


# In[69]:


df=df.dropna(axis=0)


# In[70]:


y=df["harga"]


# In[71]:


fitur=['lebih_satu_orang', 'km_dalam', 'wifi', 'listrik', 'ac', 'tipe']


# In[72]:


X=df[fitur]


# In[73]:


X.describe()


# In[74]:


from sklearn.tree import DecisionTreeRegressor
kos_model=DecisionTreeRegressor(random_state=0)
kos_model.fit(X,y)


# In[76]:


print("Prediksi 5 kos:")
print(X.head())
print(y.head())
print("The predictions are")
print(kos_model.predict(X.head()))


# MEAN ABSOLUTE ERROR

# In[77]:


from sklearn.metrics import mean_absolute_error

predicted_kos = kos_model.predict(X)
mean_absolute_error(y,predicted_kos)


# In[78]:


from sklearn.model_selection import train_test_split

train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=0,test_size=0.2)

kos_model=DecisionTreeRegressor()
kos_model.fit(train_X,train_y)
val_predictions=kos_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
score=kos_model.score(val_X,val_y)
print(score)


# MAX DEPTH

# In[79]:


from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y):
    model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(train_X,train_y)
    preds_val=model.predict(val_X)
    mae=mean_absolute_error(val_y,preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# Min samples split

# In[80]:


def get_mae(min_samples_split,train_X,val_X,train_y,val_y):
    model=DecisionTreeRegressor(min_samples_split=min_samples_split,random_state=0)
    model.fit(train_X,train_y)
    preds_val=model.predict(val_X)
    mae=mean_absolute_error(val_y,preds_val)
    return(mae)

for min_samples_split in [2,3,4,5,6,7,8,9,10,20,30]:
    my_mae=get_mae(min_samples_split,train_X,val_X,train_y,val_y)
    print("Mean samples split: %d  \t\t Mean Absolute Error:  %d" %(min_samples_split, my_mae))


# Min samples leaf

# In[81]:


def get_mae(min_samples_leaf,train_X,val_X,train_y,val_y):
    model=DecisionTreeRegressor(min_samples_leaf=min_samples_leaf,random_state=0)
    model.fit(train_X,train_y)
    preds_val=model.predict(val_X)
    mae=mean_absolute_error(val_y,preds_val)
    return(mae)

for min_samples_leaf in [5,10,15,20,30,40,50,60,70]:
    my_mae=get_mae(min_samples_leaf,train_X,val_X,train_y,val_y)
    print("Mean samples leaf: %d  \t\t Mean Absolute Error:  %d" %(min_samples_leaf, my_mae))


# In[82]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


# COMBINED

# In[83]:


dt=DecisionTreeRegressor(criterion='squared_error',splitter='best',min_samples_split=3,max_depth=50,min_samples_leaf=7)
dt.fit(train_X,train_y)
y_pred=dt.predict(val_X)

dt_regressor=evaluate(dt,val_X,val_y)


# In[84]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


# In[85]:


from scipy.stats import randint 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import RandomizedSearchCV 
  
# Creating the hyperparameter grid  
param_dist = {"max_depth": randint(1,100), 
              "min_samples_split": randint(2, 40), 
              "min_samples_leaf": randint(1, 50),
             "max_leaf_nodes":randint(1,200)} 
  
# Instantiating Decision Tree classifier 
tree = DecisionTreeRegressor() 
  
# Instantiating RandomizedSearchCV object 
tree_cv = RandomizedSearchCV(tree, param_dist, cv = 5) 
  
tree_cv.fit(train_X, train_y) 
  
tree_accuracy=evaluate(tree_cv,val_X,val_y)


# In[89]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print(accuracy)
    
    return accuracy


# In[90]:


from sklearn.ensemble import RandomForestRegressor
rt_model=RandomForestRegressor(n_estimators=50,random_state=0)
rt_model.fit(train_X,train_y)
random_forest_accuracy=evaluate(rt_model,val_X,val_y)


# In[91]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
               
#pprint(random_grid)
#{'bootstrap': [True, False],
# 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
# 'max_features': ['auto', 'sqrt'],
# 'min_samples_leaf': [1, 2, 4],
# 'min_samples_split': [2, 5, 10],
# 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}


# In[92]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_X, train_y)


# In[93]:


rf_random.best_params_


# In[94]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


# In[95]:


best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, val_X, val_y)

basemodel_accuracy=evaluate(rt_model,val_X,val_y)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - basemodel_accuracy) / basemodel_accuracy))


# In[96]:


predicted = rt_model.predict(X)
predicted.round()


# In[97]:


import pickle
pickle.dump(rt_model, open('model.pkl', 'wb'))


# In[ ]:




