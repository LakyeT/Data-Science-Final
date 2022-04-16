#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Data management libraries
import pandas as pd
import numpy as np

#Visualization library
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#Chart appearance
get_ipython().run_line_magic('matplotlib', 'inline')

import scipy.stats as stats

from IPython.display import Image


# In[2]:


df = pd.read_csv('archive/googleplaystore.csv')
df.head()


# In[3]:


print(df.shape)


# In[4]:


df.isnull().sum()


# In[5]:


#Remove null values w/ sanity check
df = df.dropna()
print(df.shape)

#reprint for null values
df.isnull().sum()


# In[6]:


df.head()


# In[7]:


print(df.duplicated().sum())


# In[8]:


df[df.duplicated(keep=False)].sort_values(by="App")


# In[9]:


df=df.drop_duplicates()


# In[10]:


df[df.duplicated(subset="App", keep=False)].sort_values(by="App")


# In[11]:


df=df.drop_duplicates(subset="App")


# In[12]:


df


# In[13]:


sns.pairplot(data=df)


# In[14]:


df.Rating.hist(bins=40, figsize=(21,13), color='maroon');


# In[15]:


#cond1=df.Reviews <= 1000
#df[cond1].Reviews.hist(bins=40, figsize=(21,13), color='maroon');
sns.displot(df['Rating'])
plt.show()
print('Knew of Distribution'.df['Rating'].skew())
print('The Median of this distribution {} is greater than mean {} of this distribution'.format(df.Rating.median(),df.Rating.mean()))


# In[16]:


plt.figure(figsize=(12,8))
fig=sns.countplot(df['Category'],palette='Reds')
fig.set_xlabel("Category",size=15)
fig.set_ylabel("Count",size=15)
plt.xticks(size=10,rotation=90)
plt.yticks(size=10)
plt.title('Number of apps per Category',size = 20)
plt.tight_layout()


# In[17]:


"""
Data Visualizataion
"""

def compute_app_types(df):
    """
    Given a dataframe, compute the number 
    of free and paid apps respectively
    """
    return sum(df.Type == "Free"), sum(df.Type == 'Paid')

def plot_app_types(df):
    """
    Plot app type distributions across categories
    """
    vc_rating = df.Category.value_counts()
    cat_free_apps = []
    cat_paid_apps = []
    for cat in vc_rating.index:
        n_free, n_paid = compute_app_types(df.query("Category == '{}'".format(cat)))
        cat_free_apps.append(n_free)
        cat_paid_apps.append(n_paid)

    f, ax = plt.subplots(2,1)
    ax[0].bar(range(1, len(cat_free_apps)+1), cat_free_apps)
    ax[1].bar(range(1, len(cat_free_apps)+1), cat_paid_apps)

def plot_target_by_group(df, target_col, group_col, figsize=(6,4), title=""):
    """
    Plot the mean of a target column (Numeric) groupped by the group column (categorical)
    """
    order = sorted(list(set(df[group_col])))
    stats = df.groupby(group_col).mean()[target_col]
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=group_col, y=target_col, data=df, ax=ax, order=order).set_title(title)
    ax.set(ylim=(3.8, 4.5))  
    ax.tick_params(labelrotation=90)
    return stats


# In[18]:


fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(25,10))
plt.suptitle('Count plots')
sns.countplot(y='Category',data=df,ax=ax1)
sns.countplot('Type',data=df,ax=ax2)
sns.countplot('Content Rating',data=df,ax=ax3)
plt.show()


# In[19]:


df.Category.unique()


# In[20]:


category_list = list(df.Category.unique())
ratings = []

for category in category_list:
    x = df[df.Category == category]
    rating_rate = x.Rating.sum()/len(x)
    ratings.append(rating_rate)
data = pd.DataFrame({'Category':category_list, 'Rating':ratings})
new_index = (data['Rating'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

sorted_data


# In[21]:


#CONTENT RATINGS BY CATEGORY
cat_list = list(df.Category.unique())

# content rating lists
everyone = []
teen = []
everyone_10 = []
mature_17 = []
adults_only_18 = []
unrated = []

# the function that fills category's content rating counts into lists
def insert_counts(everyone, teen, everyone_10, mature_17, adults_only_18, unrated, temp):
    
    # everyone
    try:
        everyone.append(temp.groupby('Content Rating').size()['Everyone'])
    except:
        everyone.append(0)
    
    # teen
    try:
        teen.append(temp.groupby('Content Rating').size()['Teen'])
    except:
        teen.append(0)
    
    # everyone 10+
    try:
        everyone_10.append(temp.groupby('Content Rating').size()['Everyone 10+'])
    except:
        everyone_10.append(0)
        
    # mature 17+
    try:
        mature_17.append(temp.groupby('Content Rating').size()['Mature 17+'])
    except:
        mature_17.append(0)
        
    # adults only 18+
    try:
        adults_only_18.append(temp.groupby('Content Rating').size()['Adults only 18+'])
    except:
        adults_only_18.append(0)
        
    # unrated
    try:
        unrated.append(temp.groupby('Content Rating').size()['Unrated'])
    except:
        unrated.append(0)

# fill lists iteratively via function
for cat in cat_list:
    temp = df[df.Category == cat]
    insert_counts(everyone, teen, everyone_10, mature_17, adults_only_18, unrated, temp)
    


# In[22]:


f,ax = plt.subplots(figsize = (25,25))
sns.barplot(x=everyone,y=cat_list,color='Orange',alpha = 0.5,label='Everyone')
sns.barplot(x=teen,y=cat_list,color='blue',alpha = 0.7,label='Teen')
sns.barplot(x=everyone_10,y=cat_list,color='pink',alpha = 0.6,label='Everyone 10+')
sns.barplot(x=mature_17,y=cat_list,color='red',alpha = 0.6,label='Mature 17+')
sns.barplot(x=adults_only_18,y=cat_list,color='black',alpha = 0.6,label='Adults Only 18+')
sns.barplot(x=unrated,y=cat_list,color='aqua',alpha = 0.6,label='Unrated')

ax.legend(loc='lower right',frameon = True)
ax.set(xlabel='Percentage of Content Ratings', ylabel='Categories',title = "Categories by Content Ratings ")


# In[23]:


df['Category'] = df.index


# In[24]:


plt.figure(figsize=(22,8))
plt.title('Number of Apps on the basis of Android version')
sns.countplot(x='Android Ver',data = df.sort_values(by = 'Android Ver'),palette='hls')
plt.xticks(rotation = 75)

plt.show()


# In[25]:


sns.distplot(df['Rating'], color ='blue')
plt.show()
print('The skewness of this distribution is',df['Rating'].skew())
print('The Median of this distribution {} is greater than mean {} of this distribution'.format(df.Rating.median(),df.Rating.mean()))


# In[26]:


plt.figure(figsize=(22,8))
plt.title('Number of Apps on the basis of Genre')
sns.countplot(x='Genres',data = df,palette='hls')
plt.xticks(rotation = 90)
plt.show()


# In[27]:


type(df['Reviews'][0])


# In[28]:


df['Reviews'] = df['Reviews'].map(lambda x: int(x))


# In[29]:


df['Reviews'].describe()


# In[30]:


df[df['Reviews']>20000000]['App']


# In[31]:


df.loc[df[['Reviews']].idxmax()]


# In[32]:


order = ['0','0+','1+','5+','10+','50+','100+','500+','1,000+','5,000+','10,000+','50,000+','100,000+','500,000+','1,000,000+',
         '5,000,000+','10,000,000+',
         '50,000,000+','100,000,000+','500,000,000+','1,000,000,000+']
sns.set_style('whitegrid')
plt.figure(figsize=(22,8))
plt.title('Number of apps on the basis of Installs')
sns.countplot(x='Installs',data = df,palette='hls',order = order)
plt.xticks(rotation = 90)

plt.show()


# In[ ]:





# In[33]:


plt.figure(figsize=(12,8.27))
sns.boxplot(df['Content Rating'],df.Rating)


# In[34]:


plt.figure(figsize=(20,6))
sns.boxplot(x='Genres',y='Rating',data = df,palette='hls')
plt.xticks(rotation=90)
plt.title('App ratings across different genres')
plt.show()


# In[35]:


plt.figure(figsize=(12,8.27))
sns.boxplot(df['Type'],df.Rating)


# In[36]:


col = df['Category'] == "Dating"
df[col]


# In[37]:


plt.figure(figsize=(22,8))
plt.title('Number of Apps per Genres')
sns.countplot(x='Genres',data = df.sort_values(by = 'Android Ver'),palette='hls')
plt.xticks(rotation = 90)

plt.show()


# In[ ]:




