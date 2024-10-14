#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[39]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]


# In[6]:


data  = pd.read_csv('creditcard.csv',sep=',')
data.head()


# In[4]:


creditcard_data .info()


# In[7]:


data.tail()


# In[8]:


#checking the number of missing values in each column
data.isnull().sum()


# In[9]:


data.isnull().values.any()


# In[40]:


#distribution of legit and fraudulent transaction
data['Class'].value_counts()


# In[11]:


## Get the Fraud and the normal dataset 

fraud = data[data['Class']==1]

normal = data[data['Class']==0]


# In[12]:


print(fraud.shape,normal.shape)


# In[13]:


## We need to analyze more amount of information from the transaction data
#How different are the amount of money used in different transaction classes?
fraud.Amount.describe()


# In[14]:


normal.Amount.describe()


# In[41]:


# compare the values for both transaction
data.groupby('Class').mean()


# In[15]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();


# In[45]:


#undersampling
#build a sample dataset containing similar distribution of legit anf fraudulent transaction
#number of fraudulent transaction --> 492

normal_sample = normal.sample(n=492)


# In[47]:


#concatenating two dataframes
new_dataset = pd.concat([normal_sample,fraud],axis=0)


# In[48]:


new_dataset.head()


# In[49]:


new_dataset.tail()


# In[50]:


new_dataset['Class'].value_counts()


# In[51]:


new_dataset.groupby('Class').mean()


# In[52]:


#splitting the data into features and targets
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[53]:


print(X)


# In[54]:


print(Y)


# In[55]:


#split the data into training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[56]:


print(X.shape, X_train.shape, X_test.shape)


# In[57]:


#model training
#Logistic Regression Model

model = LogisticRegression()


# In[59]:


#training logistic regression model with training model
model.fit(X_train, Y_train)


# In[61]:


#model evaluation
#accuracy score

#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[62]:


print('Accuracy on training data: ', training_data_accuracy)


# In[63]:


#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[64]:


print('Accuracy on test data: ', test_data_accuracy)


# In[44]:


# We Will check Do fraudulent transactions occur more often during certain time frame ? Let us find out with a visual representation.

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()


# In[20]:


## Take some sample of the data

data1= data.sample(frac = 0.1,random_state=1)

data1.shape


# In[21]:


data.shape


# In[22]:


#Determine the number of fraud and valid transactions in the dataset

fraud = data1[data1['Class']==1]

Valid = data1[data1['Class']==0]

outlier_fraction = len(fraud)/float(len(Valid))


# In[26]:


print(outlier_fraction)

print("fraud Cases : {}".format(len(fraud)))

print("Valid Cases : {}".format(len(Valid)))


# In[27]:


## Correlation
import seaborn as sns
#get correlations of each features in dataset
corrmat = data1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[28]:


#Create independent and Dependent Features
columns = data1.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Class"]]
# Store the variable we are predicting 
target = "Class"
# Define a random state 
state = np.random.RandomState(42)
X = data1[columns]
Y = data1[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)


# In[34]:


##Define the outlier detection methods

classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                       contamination=outlier_fraction,random_state=state, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
                                         max_iter=-1)
   
}


# In[35]:


type(classifiers)


# In[38]:


n_outliers = len(fraud)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    #Fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
    else:    
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
    #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != Y).sum()
    # Run Classification Metrics
    print("{}: {}".format(clf_name,n_errors))
    print("Accuracy Score :")
    print(accuracy_score(Y,y_pred))
    print("Classification Report :")
    print(classification_report(Y,y_pred))


# In[ ]:




