#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# set random seed to ensure reproducible runs
RSEED = 50


# In[33]:


# create features X
X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [2, 3], [3, 3]])

# create labels y
y = np.array([0, 1, 1, 1, 0, 1])

# Data Visualization
# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18
plt.figure(figsize=(8, 8))

# Plot each point as the label
for x1, x2, label in zip(X[:, 0], X[:, 1], y):
    plt.text(x1, x2, str(label), fontsize=40, color='g',
             ha='center', va='center')
# Plot formatting
plt.grid(None)
plt.xlim((0, 3.5))
plt.ylim((0, 3.5))
plt.xlabel('x1', size=20)
plt.ylabel('x2', size=20)
plt.title('Data', size=24)

# plt.show()


# In[34]:


from sklearn.tree import DecisionTreeClassifier

# Make a decision tree and train
tree = DecisionTreeClassifier(random_state=RSEED)
tree.fit(X, y)


# In[35]:


print(
    f'Decision tree has {tree.tree_.node_count} nodes with maximum depth {tree.tree_.max_depth}.')
print(f'Model Accuracy: {tree.score(X, y)}')


# In[36]:


from IPython.display import Image
from subprocess import call
from sklearn.tree import export_graphviz

# Visualize decision tree

# Export as dot
export_graphviz(tree, 'tree.dot', rounded=True, feature_names=[
                'x1', 'x2'], class_names=['0', '1'], filled=True)

# Convert to png
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=400'])

Image('tree.png')


# In[37]:


# Limit maximum depth and train
short_tree = DecisionTreeClassifier(max_depth = 2, random_state=RSEED)
short_tree.fit(X, y)

print(f'Model Accuracy: {short_tree.score(X, y)}')


# In[38]:


# Export as dot
export_graphviz(short_tree, 'shorttree.dot', rounded = True, 
                feature_names = ['x1', 'x2'], 
                class_names = ['0', '1'], filled = True)

call(['dot', '-Tpng', 'shorttree.dot', '-o', 'shorttree.png', '-Gdpi=400']);
Image('shorttree.png')


# In[39]:


# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
df = pd.read_csv("../data/2015.csv") 
df.head(5)
df = df.select_dtypes('number')

# Label distribution
df['_RFHLTH'] = df['_RFHLTH'].replace({2: 0})
df = df.loc[df['_RFHLTH'].isin([0, 1])].copy()
df = df.rename(columns={'_RFHLTH': 'label'})
df['label'].value_counts()


# In[40]:


# Remove columns with missing values
df = df.drop(columns = ['POORHLTH', 'PHYSHLTH', 'GENHLTH', 'PAINACT2', 
                        'QLMENTL2', 'QLSTRES2', 'QLHLTH2', 'HLTHPLN1', 'MENTHLTH'])


# In[41]:


# Split the data in to training and test set

from sklearn.model_selection import train_test_split

# Extract the labels
labels = np.array(df.pop('label'))

# 30% examples in test data
train, test, train_labels, test_labels = train_test_split(df, labels, stratify = labels, test_size = 0.3, random_state = RSEED)


# In[42]:


# Fill the missing values in the test set with the mean of columns in the training data

train = train.fillna(train.mean())
test = test.fillna(test.mean())

features = list(train.columns)


# In[43]:


train.shape


# In[44]:


test.shape


# In[ ]:


# Train tree
tree.fit(train, train_labels)
print(f'Decision tree has {tree.tree_.node_count} nodes with maximum depth {tree.tree_.max_depth}.')


# In[ ]:





# In[ ]:




