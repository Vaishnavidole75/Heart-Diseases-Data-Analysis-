# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:33:19 2023

@author: Vaishnavi Dole
"""
#======================================
#Name:-Vaishnavi Dole
#DataSet:- heart disease.csv

#======================================

#######################################################

          #DataSet - heart disease.csv
         
           #BUSINESS UNDERSTANDING

"""Maximise:-Daily exercise which lead to achieve
max. heart rate,Blood Circulation and good health"""



"""Minimise:-Minimise trestbps,cholesterol,thal. 
            Reduce the no. Of Heartdiseases patients.
"""


"""Business Constraints:-To maintain the level
of chol(cholesterol),if it is beyond
 its limit consider risky
"""

###########################################################

#           EDA=Exploratory Data Analysis

#    DATA DICTIONARY
"""
Name of             Description                                             Type                                         Relevance   
Feature
                                                                    Qualitative(Nominal,Ordinal)                      [Relevant,Irrelevant]
                                                                    Quantitative(Discrete,Continuous)             (i.e presents col.provide useful info or not??? )

age       Age of the patient                                       Continuous(Quantitative)                        Relevant                                                                                                  
sex       Gender of the patient (1 = male,0 = female).             Nominal (Qualitative)                           Relevant
cp        Chest pain type.                                         Ordinal (Qualitative)                           Relevant
trestbps  Resting blood pressure (in mm Hg).                       Continuous(Quantitative)                        Relevant
chol      Serum cholesterol level (in mg/dL).                      Quantitative (Continuous)                       Relevant
fbs       Fasting blood sugar level. If fbs>120                    Nominal (Qualitative)                           Relevant
         it is represented as 1. Otherwise, it is 0.              
restecg   Resting electrocardiographic results.                    Nominal (Qualitative)                           Relevant
thalach   Maximum heart rate achieved.                             Quantitative (Continuous)                       Relevant 
exang     Exercise-induced angina (1 = yes,0 = no).                Nominal (Qualitative)                           Relevant
oldpeak   ST depression induced by exercise relative to rest.      Quantitative (Continuous)                       Relevant
slope     The slope of the peak exercise ST segment.               ordinal (Qualitative)                           Relevant
ca        Number of major vessels                                  Quantitative (Discrete)                         Relevant 
thal      Thalassemia. It is a blood disorder.                     Nominal (Qualitative)                           Relevant
target    The presence of heart disease(1=presence,0=absence).     Nominal (Qualitative                            Relevant

"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df=pd.read_csv("C:/2-datasets/heart disease.csv")
df1=pd.DataFrame(df)
df1
###############################################

##              DATA PRE-PROCESSING

#     Data Cleaning and Feature Engineering

df1.columns
#(total 14 columns are there.)

######
df1.shape
"""(rows-303,col-14)"""
###############################
#              Univarient Analysis
"""PROCESS:-
Check Data Types
Display the First Few Rows
Checking for duplicates 
Handle Missing Values
Check Outliers"""
#

#check the data type of columns
df1.dtypes
"""all col. have int64 data type ,only oldpeak col. have data type float."""
#

# Summary
s=df1.describe()
print(s)

"""
count 303	
mean	54.366336633663366	0.6831683168316832	0.966996699669967	131.62376237623764	246.26402640264027	0.1485148514851485	0.528052805280528	149.64686468646866	0.32673267326732675	1.0396039603960396	1.3993399339933994	0.7293729372937293	2.3135313531353137	0.5445544554455446
std	  9.082100989837858	    0.4660108233396251	1.0320524894832992	17.53814281351709	51.830750987930045	0.35619787492797594	0.525859596359298	22.905161114914087	0.46979446452231716	1.1610750220686343	0.6162261453459631	1.0226063649693276	0.6122765072781412	0.4988347841643926

min	29.0	0.0	0.0	94.0	126.0	0.0	0.0	71.0	0.0	0.0	0.0	0.0	0.0	0.0
25%	47.5	0.0	0.0	120.0	211.0	0.0	0.0	133.5	0.0	0.0	1.0	0.0	2.0	0.0
50%	55.0	1.0	1.0	130.0	240.0	0.0	1.0	153.0	0.0	0.8	1.0	0.0	2.0	1.0
75%	61.0	1.0	2.0	140.0	274.5	0.0	1.0	166.0	1.0	1.6	2.0	1.0	3.0	1.0
max	77.0	1.0	3.0	200.0	564.0	1.0	2.0	202.0	1.0	6.2	2.0	4.0	3.0	1.0
"""
################################
#Display the first few rows of the dataset
y=df1.head()
"""we display intial 5 rows data"""

############################
#check whether null value is present or not
df1.isnull().sum()
"""there is not a single null value present"""
################################3

#(No need to write this code bcz we don't have any missing value)
# Handle missing values
m=df1.fillna(df1.mean(), inplace=True)
m
"""
as we check there is no missing value is present in this data
so need to write this code
Fills missing values with the mean of each column.
"""


######################################################
# Check for duplicates
df1.duplicated().sum()
#o/p 1
""" 1 duplicate row is avaliable
here in this dataset duplicate values are present.
That duplicate value introduce some error in our model 
and also we get less accuracy.
we need remove that duplicate data,To get more accuracy

before we have (rows=303 & col=14)
"""
#####################################################

#Remove duplicate values
d=df1.drop_duplicates()
d
"""Removed duplicate row
now data have 302 rows and 14 column"""

####################################################

     #Check Whether Outlier present or not
"""
affect the mean value(inaccurate mean value) of the
data but have little effect on the median or mode.
Outliers can have a big impact on
analyses test if they are inaccurate. """

#various method we have to check outliers
#    scatter plot
#    Box plot
#    Z-score
#    IQR (Interquartile Range Q1=25%,Q2=50%median,Q3=75%)

# Identify & handel outliers (using IQR method)
Q1 = df1.quantile(0.25)
Q3 = df1.quantile(0.75)
IQR = Q3 - Q1
#removing outlier from data
no_outliers = df1[~((df1 < (Q1 - 1.5 * IQR)) | (df1 > (Q3 + 1.5 * IQR))).any(axis=1)]
"""Here we remove the outlier which is greater than 1.5 and
less than 1.5"""
#we know 
#If data value  < Q1 - 1.5*IQR  =then outlier
#If data value  > Q3 + 1.5*IQR  =then outlier
no_outliers.shape

"""(228, 14)
some outlier rows is removed from data"""


####################################################3
#Technique-2 for finding outlier

#               Bivariate Analysis
#       Process
"""
Correlation Analysis:[ use df1.corr()]
Box Plot
Standardization
One-Hot Encoding
"""

#1] Correlation Analysis
# Correlation Analysis

"""
How to read a correlation matrix:
 Coefficent of Correlation is measured from -1 to 1
 
1)Look at the number in each cell to see the
strength and direction of the correlation.
2)Positive numbers indicate positive correlations,
while negative numbers indicate negative correlations.
3)The closer the number is to 1 (or -1), 
the stronger the correlation.
4)A number of 0 means there is no correlation 
  between the two variables.

"""
correlation_matrix = df1.corr()

# Plotting the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

##################################################

#2]Box Plot

sns.boxplot(df1)
"""trestbps,chol,thalach have outliers
other dont have outlier
"""
#(automatic take number column which have variation in no.)

#same in above code
# box plot numerical columns
#plt.figure(figsize=(12, 8))
#sns.boxplot(data=df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])
#plt.title('Box Plots for Numerical Columns')
#plt.show()


###############################################################################
       # Standardization numerical
"""-To bring all features to a similar scale
   -To make effective Regularization techniques, such as L1 or L2 regularization
   
   L1=Lasso Regularization
   term is the absolute sum of the model's coefficients. 
   
   L2=Ridge Regularization
   term is the squared sum of the model's coefficients
"""

from sklearn.preprocessing import StandardScaler

#columns_to_standardize is a list of column names to standardize
columns_to_standardize = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the selected columns
df1[columns_to_standardize] = scaler.fit_transform(df1[columns_to_standardize])

###################################################################################

############################################################
"""
#There is no need of normalisation
#It's only practice purpose
#Normalization
from sklearn.preprocessing import MinMaxScaler


# Assuming 'data' is your DataFrame and 'columns_to_standardize' is a list of column names to standardize
columns_to_normalize = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Initialize the StandardScaler
scaler = MinMaxScaler()

# Fit and transform the selected columns
df1[columns_to_normalize] = scaler.fit_transform(df1[columns_to_normalize])
"""

#####################################################
#4]ONE HOT ENCODING
#           ONE HOT ENCODING (Feature Engineering)
#One hot Encoding is used :-
    #1] Algorithms Requiring Numerical Input
    #In ML algorithms, such as linear models and
    #distance-based models, require numerical input
   #2]Preventing Multicollinearity

# Identify categorical columns
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# Perform one-hot encoding
df1 = pd.get_dummies(df1, columns=categorical_columns, drop_first=True)

############################################################333333333

###                 Clustering
#Process
"""
1]Scatter pLot-
Every time Before clustering we need 
to plot scatter plot for clustering operation

 
                                                                          
"""
# Visualization using Scatter Plot Before K-Means Clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(x='thalach', y='chol', data=df1, palette='Set1', s=80, alpha=0.7)
plt.title('Scatter Plot Before K-Means Clustering')
plt.xlabel('Maximum Heart Rate Achieved (thalach)')
plt.ylabel('Serum Cholesterol Level (chol)')
plt.show()


###############################################

# Model Building

##Hierarchical Clusering

#For visualzing the cluster of  the above dataframe we  have to draw
# Dendodron first then we cluster the datapoints

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

# linkage function give the hierarchical and Agglomotive clustering
 

z=linkage(df1,method='complete',metric='euclidean')

plt.figure(figsize=(15,8))
plt.title('Hierarchical Clustering')
plt.xlabel('Index')
plt.ylabel('Distance')
#sch is help to draw 
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

#appying agglomerative clustering choose 1 as a cluster from dendogram

# In dedrogram is not show the clustering it only shows how many clusters are there

from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=2,linkage='complete',affinity='euclidean').fit(df1)

#apply labels to the cluster
h_complete.labels_
# so these all are in the form of array we have to convert the Series
cluster_labels=pd.Series(h_complete.labels_)
# so these all are in the form of array we have to convert the Series
cluster_labels=pd.Series(h_complete.labels_)

df['clust']=cluster_labels
df

####### K-Means Clustering ###############
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df1)
    
    TWSS.append(kmeans.inertia_)
    
    '''
    kmeans inertia also known as sum odf sqares methos
    .It measures all the datapoints from the centroid of the point.
    it differentiate between observed value and predicted value
    '''
    
TWSS
# Plot a elbow curve
plt.plot(k,TWSS,'ro-')
plt.xlabel('No of clusers')
plt.ylabel('Total within SS')

model=KMeans(n_clusters=3)
model.fit(df1)
model.labels_
mb=pd.Series(model.labels_)
type(mb)
df['clust']=mb
df.head()
d=df.iloc[:,[5,0,1,2,3,4]]
d


#################### PCA #####################

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

df=pd.read_csv('C:/2-datasets/heart disease.csv')
df
#Normalize the numeric data
uni_normal=scale(df)
uni_normal

pca=PCA(n_components=3)
pca_values=pca.fit_transform(uni_normal)

#The amount of variance that each PCA explain

var=pca.explained_variance_ratio_
var

#Commulative Variance
var1=np.cumsum(np.round(var,decimals=4)*100)
var1
#Variance plot for PCA component obtained
plt.plot(var1,color='red')
#PCA Scores
pca_values

pca_data=pd.DataFrame(pca_values)
pca_data.columns='comp0','comp1','comp2'

final=pd.concat([df.clust,pca_data],axis=1)

#Visualize the dataframe
ax=final.plot(x='comp0',y='comp1',kind='scatter',figsize=(12,8))
final[['comp0','comp1','clust']].apply(lambda x:ax.text(*x),axis=1)


