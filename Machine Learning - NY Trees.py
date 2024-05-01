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


#Plot distribution of numerical fields
trees.hist(bins = 100, figsize=(15,10))

#Explore outliers in tree diameter variable
tree_dbh_outliers = trees[trees['tree_dbh'] > 60 ]
tree_dbh_outliers[['tree_id', 'tree_dbh']].plot(kind = 'scatter', x = 'tree_id', y = 'tree_dbh')

#Explore tree status variable
trees['status'].value_counts()

#Explore tree health variable
trees['health'].value_counts()

#Explore trees that are dead
dead_trees = trees[trees['status'] == 'Dead']

#Explore distributions of binary variables - problems with tree
trees_problems = trees[['root_stone','root_grate', 'root_other', 'trunk_wire', 'trnk_light', 'trnk_other','brch_light', 'brch_shoe', 'brch_other']]
trees_problems.apply(pd.Series.value_counts)
