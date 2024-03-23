#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[6]:


ipl = pd.read_csv('/Users/utkarsh/Desktop/Coding/IPl/Book_ipl22_ver_33.csv')


# In[7]:


ipl.head()


# In[8]:


ipl = pd.read_csv('/Users/utkarsh/Downloads/ipl_2022_dataset.csv')


# In[9]:


ipl.head()ipl.head()ipl


# In[10]:


ipl.shape


# In[11]:


ipl.info()


# In[16]:


ipl.columns


# In[29]:


ipl.drop('Unnamed: 0', axis=1, inplace=True)


# In[30]:


ipl.head()


# In[32]:


ipl.isnull().sum()


# In[34]:


ipl[ipl['Cost IN $ (000)'].isnull()]


# In[41]:


ipl['COST IN ₹ (CR.)'] = ipl['COST IN ₹ (CR.)'].fillna(0)
ipl['Cost IN $ (000)']= ipl['Cost IN $ (000)'].fillna(0)    


# In[42]:


ipl[ipl['2021 Squad'].isnull()]


# In[43]:


ipl['2021 Squad)'] = ipl['2021 Squad'].fillna('Not Participated')


# In[47]:


ipl.isnull().count()


# In[48]:


ipl.isnull().sum()


# In[49]:


teams = ipl [ipl['COST IN ₹ (CR.)']>0]['Team'].unique()


# In[50]:


teams = ipl [ipl['COST IN ₹ (CR.)']>0]['Team'].unique()
teams


# In[53]:


ipl['status']=ipl['Team'].replace(teams,'sold')


# In[54]:


ipl.head()


# In[55]:


ipl


# In[57]:


ipl[ipl ['Player'].duplicated(keep=False)]


# In[58]:


# How many player participated in 2022
ipl.shape[0]


# In[61]:


# How many types of player participated in 2022
types =ipl['TYPE'].value_counts()
types.reset_index()


# In[66]:


plt.pie(types.values,labels=types.index,labeldistance=1.2,autopct='%1.2f%%' , shadow=True, startangle=60)
plt.title('Role of Players participated' , fontsize=15)
plt.plot()


# In[77]:


#player sold and unsold using a bar garph
plt.figure(figsize=(10,5))
fig = sns.countplot(x='status', data=ipl, palette=['Orange', 'Pink'])
plt.xlabel('Sold or Unsold')
plt.ylabel('Number of Players')
plt.title('Sold or Unsold', fontsize=15)

for p in fig.patches:
    fig.annotate(format(p.get_height(), '0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 4), textcoords='offset points')


# In[85]:


ipl.groupby('status')['Player'].count()


# In[96]:


ipl['Team'] = ipl['Team'].astype('category')

plt.figure(figsize=(20,10))
fig = sns.countplot(data=ipl[ipl['Team'] != 'Unsold'], x='Team')
plt.xlabel('Team Names')
plt.ylabel('Number of players')
plt.title('Players Bought by Each Team', fontsize=12)
plt.xticks(rotation=70)

# Add annotations to each bar
for p in fig.patches:
    fig.annotate(format(p.get_height(), '.0f'), 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='center', xytext=(0, 4), 
                 textcoords='offset points')

plt.show()


# In[101]:


ipl


# In[103]:


ipl['retention'] = ipl['Base Price']


# In[105]:


ipl['retention'].replace(['2 Cr', '40 Lakh' , '20 Lakh', '1 Cr', '75 Lakh','50 Lakh','30 Lakh','1.5 Cr'],
                         'From Auction' , inplace= True )


# In[109]:


#Treating Base Price 
ipl['Base Price'].replace('Draft Pick', 0 , inplace = True)


# In[119]:


ipl['base_price_unit'] = ipl['Base Price'].apply (lambda x : str(x).split (' ')[-1])
ipl['base_price'] = ipl['Base Price'].apply (lambda x : str(x).split (' ')[0])


# In[120]:


ipl ['base_price'].replace ('Retained', 0,inplace=True)


# In[121]:


ipl.head()


# In[135]:


#Total Players retained and bought
ipl.groupby(['Team','retention'])['retention'].count ()[:-1]


# In[146]:


plt.figure(figsize=(20,10))
sns.countplot(data=ipl[ipl['Team']!='Unsold'], x='Team',hue='TYPE')
plt.title('Players in Each Team')
plt.xlabel('Team Names')
plt.ylabel('Number of Players')
plt.xticks(rotation=50)


# In[156]:


# Filtering the DataFrame to include only entries where retention is 'From Auction'
from_auction_ipl = ipl[ipl['retention'] == 'From Auction']

# Grouping by 'Team' and finding the maximum cost for each team
max_cost_by_team = from_auction_ipl.groupby('Team')['max()

# Sorting the values in descending order
sorted_max_cost_by_team = max_cost_by_team.sort_values(ascending=False)

# Displaying the result
print(sorted_max_cost_by_team)


# In[159]:


# Remove the extra space from the column name 'retention'
ipl[ipl['retention']=='Retained'].sort_values(by='COST IN ₹ (CR.)', ascending=False).head()


# In[167]:


# Top 5 Bowlers from Auction
ipl[(ipl['retention'] == 'From Auction') & (ipl['TYPE'] == 'ALL-ROUNDER')].sort_values(by='COST IN ₹ (CR.)', ascending=False).head()


# In[184]:


ipl=ipl.rename(columns={'2021 Squad':'Prev_team'})


# In[180]:


ipl


# In[185]:


unsold_players = ipl[(ipl.Prev_team != 'Not Participated')
                    & (ipl.Team == 'Unsold')] [['Player', 'Prev_team']]


# In[186]:


print ( unsold_players)


# In[ ]:




