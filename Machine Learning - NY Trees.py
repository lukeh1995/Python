#Import required packages
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
plt.show()

#Load tree dataset
trees = pd.read_csv(r'C:\Users\60234651\Downloads\NYTrees.csv')

#Initial preview of data - view all columns
print(trees.head(10))
trees.columns

#Remove irellevant columns from analysis
trees = trees[['tree_id',  'tree_dbh', 'stump_diam', 'curb_loc', 'status', 'health', 'spc_latin', 'spc_common', 'steward', 'guards', 'sidewalk', 'problems', 'root_stone','root_grate', 'root_other', 'trunk_wire', 'trnk_light', 'trnk_other','brch_light', 'brch_shoe', 'brch_other']]

#Check for null values
trees.isna().sum()

#Explore field types
trees.dtypes

#Inspect numerical fields
trees.describe()

#Explore latin name field
pd.DataFrame(trees['spc_latin'].value_counts()).plot(kind='bar', figsize = (20,15))

#Replace missing latin name field values
trees['spc_latic'].fillna('None')

#Plot distribution of numerical fields
trees.hist(bins = 100, figsize=(15,10))

#Explore outliers in tree diameter variable
tree_dbh_outliers = trees[trees['tree_dbh'] > 60 ]
tree_dbh_outliers[['tree_id', 'tree_dbh']].plot(kind = 'scatter', x = 'tree_id', y = 'tree_dbh')

#Remove outliers
trees = trees[(trees['tree_dbh'] <= 60) & (trees)['stump_diam'] <= 60]

#Explore tree status variable
trees['status'].value_counts()

#Explore tree health variable
trees['health'].value_counts()

#Explore trees that are dead
dead_trees = trees[trees['status'] == 'Dead']

#Explore distributions of binary variables - problems with tree
trees_problems = trees[['root_stone','root_grate', 'root_other', 'trunk_wire', 'trnk_light', 'trnk_other','brch_light', 'brch_shoe', 'brch_other']]
trees_problems.apply(pd.Series.value_counts)

#Dead trees have NaN for the above variables - change to "Not Applicable"
dead_mask = ((trees['status'] == 'Stump') | (trees['status'] == 'Dead'))
trees.loc[dead_mask] = trees.loc[dead_mask].fillna('Not Applicable')
trees[trees['status'] == 'Dead']

#Explore other missing values
trees['problems'].value_counts()
trees['problems'].isna().sum()

#Replace values for Problems field
trees['problems'].fillna('None', inplace=True)

alive_trees = trees[trees['status'] == 'Alive']
alive_trees = alive_trees.drop('stump_diam', axis= 1)
#Explore tree diameter across species
alive_trees.groupby('spc_latin')['tree_dbh'].describe()

#Drop trees with diameter of 0
alive_trees = alive_trees[alive_trees['tree_dbh'] > 0]

alive_trees.describe()
#Explore health variable
alive_trees['health'].value_counts()

#Seperate into X and Y variables
col = alive_trees.pop("health")
alive_trees.insert(0, col.name, col)
X = alive_trees.iloc[:,0]
Y = alive_trees.iloc[:,2:19 ]

#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)