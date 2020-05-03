# Decision tree to solve binary classification problem

from IPython.display import Image
from subprocess import call
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# set random seed to ensure reproducible runs
RSEED = 50

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


# Make a decision tree and train
tree = DecisionTreeClassifier(random_state=RSEED)
tree.fit(X, y)

print(
    f'Decision tree has {tree.tree_.node_count} nodes with maximum depth {tree.tree_.max_depth}.')
print(f'Model Accuracy: {tree.score(X, y)}')

# Visualize decision tree

# Export as dot
export_graphviz(tree, 'tree.dot', rounded=True, feature_names=[
                'x1', 'x2'], class_names=['0', '1'], filled=True)

# Convert to png
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=400'])

Image('tree.png')
