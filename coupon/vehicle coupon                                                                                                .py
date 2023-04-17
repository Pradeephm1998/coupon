#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('in-vehicle-coupon-recommendation.csv')


# In[3]:


df.head(30)


# In[4]:


df.coupon.value_counts()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df.isnull().sum().sort_values(ascending=False) * 100 /len(df)


# Observations :
# 
# 1.The car veriable has null values more than 95% so we drop te column.  

# In[9]:


#Dropping the car veriable  
df=df.drop(["car"],axis=1)


# In[10]:


# Selecting all object type columns
cat_cols = df.select_dtypes(include='object').columns.to_list()
cat_cols


# In[11]:


# Exploring Object Type Columns
for col in cat_cols:
    print("Unique Value count : ", col, '\n')
    print('Count: ', len(df[col].unique()),'\n\n')


# In[12]:


## Checking the value counts of Categorical Columns
## Exploring Object Type Columns
for col in cat_cols[:]:
    print("Value count : ", col, '\n')
    print(df[col].value_counts(),'\n\n')


# In[13]:


#non_cat_cols = df.select_dtypes(include!='object').columns.to_list()
#non_cat_cols
non_cat_cols = [i for i in df.columns if df[i].dtype != 'O']
non_cat_cols


# In[14]:


for col in non_cat_cols:
    print("Unique Value count : ", col, '\n')
    print('Count: ', len(df[col].unique()),'\n\n')


# In[15]:


for col in non_cat_cols[1:]:
    print("Value count : ", col, '\n')
    print(df[col].value_counts(),'\n\n')


# In[16]:


## Missing values
miss_values = df.isnull().sum()
print(type(miss_values))
miss_values[miss_values > 0]


# In[17]:


for col in df.columns:
    if df[col].isnull().sum() and df[col].dtype == 'object':
        df[col].loc[(df[col].isnull())] = df[col].mode().max()


# In[18]:


df.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


## Getting the indexes of missing values form Outlet_size column
#mv_idx = df[df.Bar.isnull()].index.to_list()
#mv_idx[:7]


# In[20]:


#df.groupby(['passanger','Bar'])['Bar'].count()


# In[21]:


#df.groupby('gender')['Bar'].count()


# In[22]:


#df.loc[mv_idx, 'Bar'] = 'never'


# In[23]:


#df.groupby(['direction_same','gender','CoffeeHouse'])['CoffeeHouse'].count()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


#for i in df:
    #print(df[i].value_counts())


# In[25]:


df.head()


# In[26]:


null110=[]
for i in df[df.columns]:
    if df[i].isnull().sum()<110 and df[i].isnull().sum()>0:
        null110.append(i)
print(null110)


# In[27]:


df.dropna(subset = null110, inplace = True)


# In[28]:


df.isnull().sum()


# In[29]:


df.shape


# In[30]:


null110=[]
for i in df[df.columns]:
    if df[i].isnull().sum()<110 and df[i].isnull().sum()>0:
        null110.append(i)
print(null110)


# In[31]:


df.dropna(subset = null110, inplace = True)


# In[32]:


df.isnull().sum()


# In[33]:


df.shape


# In[34]:


df_nonobjcolumns = [i for i in df.columns if df[i].dtype == 'O']
df_nonobjcolumns


# In[35]:


for col in df.columns:
    if df[col].isnull().sum() and df[col].dtype == 'object':
        df[col].loc[(df[col].isnull())] = df[col].mode().max()


# In[36]:


df.isnull().sum()


# In[37]:


df.info()


# In[38]:


df.columns


# In[39]:


df.dtypes


# In[40]:


df_obj = df.select_dtypes(include=['object']).copy()

for col in df_obj.columns:
    df[col]=df[col].astype('category')


# In[41]:


df.dtypes


# In[42]:


df.head()


# In[43]:


df.describe


# In[44]:


df.head()


# In[45]:


from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


# In[46]:


x = df.drop('Y', axis=1)
y = df['Y']


# In[47]:


ohe = OneHotEncoder(sparse=False)
x = pd.DataFrame(ohe.fit_transform(x), index=y.index, columns=ohe.get_feature_names(x.columns))
x


# In[ ]:





# In[48]:


pd.set_option('display.max_columns',25)
df.sample(5)


# In[49]:


df.sample(30)


# In[50]:


df["occupation"].value_counts()


# In[51]:



df["education"].value_counts()


# In[ ]:





# In[52]:


df.dtypes


# In[53]:


df.describe()


# In[54]:


import scipy.stats
from scipy.stats import chi2_contingency


# In[55]:


for i in df:             
    print(df[i].value_counts())


# In[56]:


ct_dest = pd.crosstab(df.Y, df.destination)
chi2_contingency(ct_dest, correction = False)


# In[57]:


cat_cols


# In[101]:


for i in cat_cols:
    ct_pass = pd.crosstab(df.Y, df[i])    
    print(chi2_contingency(ct_pass, correction = False))
   
    
        
    
    
    


# In[ ]:





# In[59]:


ct_pass = pd.crosstab(df.Y, df.passanger)
chi2_contingency(ct_pass, correction = False)


# In[60]:


ct_weather = pd.crosstab(df.Y, df.weather)
chi2_contingency(ct_weather, correction = False)


# In[61]:


ct_temp = pd.crosstab(df.Y, df.temperature)
chi2_contingency(ct_temp, correction = False)


# In[62]:


ct_time = pd.crosstab(df.Y, df.time)
chi2_contingency(ct_time, correction = False)


# In[63]:


ct_coupon = pd.crosstab(df.Y, df.time)
chi2_contingency(ct_coupon, correction = False)


# In[64]:


ct_expiration = pd.crosstab(df.Y, df.expiration)
chi2_contingency(ct_expiration, correction = False)


# In[65]:


ct_gender = pd.crosstab(df.Y, df.gender)
chi2_contingency(ct_gender, correction = False)


# In[66]:


ct_age = pd.crosstab(df.Y, df.age)
chi2_contingency(ct_age, correction = False)


# In[67]:


ct_maritalStatus = pd.crosstab(df.Y, df.maritalStatus)
chi2_contingency(ct_maritalStatus, correction = False)


# In[68]:


ct_has_children = pd.crosstab(df.Y, df.has_children)
chi2_contingency(ct_has_children, correction = False)


# In[69]:


ct_education = pd.crosstab(df.Y, df.education)
chi2_contingency(ct_education, correction = False)


# In[70]:


ct_occupation = pd.crosstab(df.Y, df.occupation)
chi2_contingency(ct_occupation, correction = False)


# In[71]:


ct_income = pd.crosstab(df.Y, df.income)
chi2_contingency(ct_income, correction = False)


# In[72]:


ct_bar = pd.crosstab(df.Y, df.Bar)
chi2_contingency(ct_bar, correction = False)


# In[73]:


df.columns


# In[74]:


ct_CoffeeHouse = pd.crosstab(df.Y, df.CoffeeHouse)
chi2_contingency(ct_CoffeeHouse, correction = False)


# In[75]:


ct_CarryAway = pd.crosstab(df.Y, df.CarryAway)
chi2_contingency(ct_CarryAway, correction = False)


# In[76]:


ct_RestaurantLessThan20 = pd.crosstab(df.Y, df.RestaurantLessThan20)
chi2_contingency(ct_RestaurantLessThan20, correction = False)


# In[77]:


ct_Restaurant20To50 = pd.crosstab(df.Y, df.Restaurant20To50)
chi2_contingency(ct_Restaurant20To50, correction = False)


# In[78]:


ct_toCoupon_GEQ5min = pd.crosstab(df.Y, df.toCoupon_GEQ5min)
chi2_contingency(ct_toCoupon_GEQ5min, correction = False)


# In[79]:


ct_toCoupon_GEQ15min = pd.crosstab(df.Y, df.toCoupon_GEQ15min)
chi2_contingency(ct_toCoupon_GEQ15min, correction = False)


# In[80]:


ct_toCoupon_GEQ25min = pd.crosstab(df.Y, df.toCoupon_GEQ25min)
chi2_contingency(ct_toCoupon_GEQ25min, correction = False)


# In[81]:


ct_direction_same = pd.crosstab(df.Y, df.direction_same)
chi2_contingency(ct_direction_same, correction = False)


# In[82]:


ct_direction_opp = pd.crosstab(df.Y, df.direction_opp)
chi2_contingency(ct_direction_opp, correction = False)


# In[83]:


df1=df.drop(["direction_opp","direction_same"],axis=1)


# In[84]:


cont=['destination', 'passanger', 'weather', 'temperature', 'time', 'coupon','expiration','gender','age','maritalStatus','has_children','education',
     'occupation','income','Bar','CoffeeHouse','CarryAway','RestaurantLessThan20','Restaurant20To50','toCoupon_GEQ5min','toCoupon_GEQ15min','toCoupon_GEQ25min']
plt.figure(figsize=(10,80))
n=1
for i in cont:
    ax= plt.subplot(12,2,n)
    sns.countplot(df[i],hue=df['Y'])
    plt.xlabel(i,fontsize=8)
    plt.xticks(rotation=90)
    n+=1
plt.show()


# In[85]:


df1.columns


# In[86]:


df1.info()


# In[87]:


df1.Bar.value_counts()


# Decision Tree
# 

# In[88]:



x = df1.drop('Y', axis=1)
y = df1['Y']

ohe = OneHotEncoder(sparse=False)
x = pd.DataFrame(ohe.fit_transform(x), index=y.index, columns=ohe.get_feature_names(x.columns))
x


# In[ ]:





# In[89]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=2)


# In[90]:


from sklearn.metrics import classification_report, confusion_matrix      # confusion_matrix

from sklearn import metrics                              # evaluation

#cross validation

from sklearn.model_selection import cross_val_score


# In[91]:



from sklearn.linear_model import LogisticRegression      # Loggistic Regression

from sklearn.naive_bayes import ComplementNB             # Naive Bayes

from sklearn.neighbors import KNeighborsClassifier 


# In[92]:


logreg = LogisticRegression()
logreg.fit(x_train,y_train.ravel())
y_pred = logreg.predict(x_test)


# In[93]:


# plot

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
plt.plot(fpr,tpr,label = "data 1")
plt.legend(loc = 4)
plt.show()


# In[94]:


y_pred_proba = logreg.predict_proba(x_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
plt.plot(fpr,tpr,label="data 1")
plt.legend(loc=4)
plt.show()


# In[95]:


confusion_matrix(y_test, y_pred)


# In[96]:


# score model

print('Accuracy:', metrics.accuracy_score(y_test, y_pred))


# In[97]:


print('f1 score is:', metrics.f1_score(y_test, y_pred))


# In[98]:


print(classification_report(y_test, y_pred))


# In[99]:


Complement = ComplementNB()
Complement.fit(x_train,y_train.ravel())
y_pred = Complement.predict(x_test)


# In[ ]:




