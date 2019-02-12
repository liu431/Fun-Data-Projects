#!/usr/bin/env python
# coding: utf-8

# Author: Li Liu
# 
# Date: Jan 30th - Feb 1st, 2019

# ### Import packages

# In[390]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import plotly.plotly as py
plotly.tools.set_credentials_file(username='lliu95877', api_key='YFckxswGtCktzNhNV6Jk')


# ### Import data 

# In[391]:


df=pd.read_excel("RA Test Data Task 1.xlsx")
df.head(2)


# ### Exploratory Data Analysis

# ##### Status

# In[437]:


#"Status" is the binary outcome variable
sns.countplot(df['Status'],label="Count")
plt.title("The number of students for each status is almost balanced.")
plt.show()
print("Not attending students:",sum(df['Status']==0))
print("Attending students:",sum(df['Status']==1))
print("The ratio of two status is around 52:48.")


# ##### Preprocessing 

# In[393]:


#Summary statistics for continuous variables in the data
df.describe()


# There are 141 observations with 19 input variables and 1 output variable.
# 
# The youngest, average and oldest ages are 21, 28 47 years old, respectively. I am suspicious about the maximum values for `College Degree GPA`, `GRE Verbal1`, `GRE Quantitative1` as they seem to be measured on different scales compared with other values. 

# In[394]:


#Find and fix the GPAs on different scales
print(df[df['College Degree GPA']>4.0]['College Degree GPA'])
df['College Degree GPA'].replace(79.71,2.7,inplace=True)
df['College Degree GPA'].replace(94.0,4.0,inplace=True)
df['College Degree GPA'].replace(4.3,4.0,inplace=True)


# The three GPAs larger than 4.0 could be measured by percent grades or weighted GPA. In this exerciese, most of the GPAs are on the 4.0 scale. So I converted the three GPAs to the 4.0 scale based on the conversion table (available from https://gpacalculator.net/gpa-scale/).

# In[395]:


##Find the GREs on different scales
df[df['GRE Verbal1']>170][['GRE Verbal1','GRE Quantitative1']]


# The average values of GRE verbal and quantitative parts are 176, which is problematic as the total score is 170 for each section. This is because there are four GRE scores on the prior 800 scale (before August 2011). 
# 
# The percentiles don't depend on the grading scale and show the percentage of test-takers the students scored better than. `GRE Verbal Percentile1` and `GRE Quantitative Percentile1` are two variables that are useful for the predictive models.

# ##### Race v.s. Status

# In[396]:


df.groupby('Race').agg({'Ref':'size','Status':'mean'})


# Surprisingly, the Korean and Chinese have the highest attending rates. 61% of the students are white and around half of them accepted the offers.

# ##### Area of focus v.s. Status

# In[397]:


print("There are",len(df['Area of Focus'].unique()),"areas of focus chosen by the students.")
df.groupby('Area of Focus').agg({'Ref':'size','Status':'mean'})


# Education policy is the most popular area of focus, while students interested in international policy are very likely to pursue the study at Harris.

# ##### Programs v.s. Status

# In[398]:


df.groupby('Masters Program').agg({'Ref':'size','Status':'mean'})


# There are seven programs and the 61% of the students got offer from the MPP program. Students admitted to Professional Option and MA Program (1 Year) are most likely to attend, while MSCAPP offer holders might have many alternative options.

# In[399]:


sns.countplot(y=df['Masters Program'],label="Count")
plt.title("Counts of students with offers from different programs")
plt.show()


# ##### Reader 1 v.s. Status

# In[400]:


df.groupby('Reader 1 Recommendations').agg({'Ref':'size','Status':'mean'})          


# Better recommendations don't lead to higher attending rates. Instead, students given "C" recommendation grades all choose to attend, while around half of the students given "A" or "B" recommendation grades accept the offers. I will create a binary variable in the following session for whether given "C" grade or not.

# In[401]:


df.groupby('Reader 1 Leadership').agg({'Ref':'size','Status':'mean'})


# Better leadership scores are associated with higher attending rates. Students given "A" leadership grades all choose to attend, while less than half of the students given "B" or "C" leadership grades accept the offers. I will create a binary variable in the following session for whether given "A" grade or not.
# 

# ##### Status Table by States

# In[402]:


#Group by identified U.S. states
df['Region'].replace('LA','CA',inplace=True)
states=df.groupby('Region').agg({'Ref':'size','Status':'mean','Age':'mean',
                                 'College Degree GPA':'mean',
                                 'GRE Verbal Percentile1':'mean',
                                 'GRE Quantitative Percentile1':'mean'})
states=states.drop(['Eastern','Kyonggi-do']) #'Kyonggi-do' is in Korea and 'Eastern' is ambiguous
states['Region']=states.index
states=states.round(2)
states.head(3)


# This aggregated table shows 34% of the offer holders are from Illinois and 65% of them decided to attend Harris. The number of students from neighboring states (Indiana, Kentucky, Missouri, Iowa, Wisconsin) is relatively small. Instead, there are 14, 8, and 8 students from California, DC, and New York, respectively. The attending rates for them are 36%, 25%, and 25%, which are quite low. These states have great universities offering public policy programs, so students might have many options to choose from.

# #####  Status Map by States

# In[403]:


#Plot a choropleth map using plotly package
for col in states.columns:
    states[col] = states[col].astype(str)

states['text'] = 'States  '+states['Region'] + '<br>' +                 'Counts  '+states['Ref']+ '<br>' +                  'Average Age  '+states['Age']+ '<br>' +                 'College Degree GPA  '+states['College Degree GPA']+ '<br>' +                 'GRE Verbal Percentile1  '+states['GRE Verbal Percentile1']+ '<br>' +                 'GRE Quantitative Percentile1  '+states['GRE Quantitative Percentile1']

data = [ dict(type='choropleth', autocolorscale = True,
        locations = states['Region'], z = states['Status'].astype(float),
        locationmode = 'USA-states', text = states['text'],
        marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
        colorbar = dict(title = "Percentage")) ]

layout = dict(title = 'Information on Harris Offer-holders by State',
        geo = dict(scope='usa',projection=dict( type='albers usa' ),
            showlakes = True,lakecolor = 'rgb(255, 255, 255)'),)
    
fig = dict( data=data, layout=layout )
py.iplot(fig, filename='students-states-map')


# The choropleth map displays various characteristics of students from different states. Users can hover around the plot for detailed information on attending rates, students counts, age, GPA, and GRE percentiles for the specific state.

# ### Handling Missing Data

# In[404]:


#Number of missing values in each column
df.isnull().sum()


# 15.6% of the observations don't have values for `Reader 1 Academic`,`Reader 1 Leadership`,`Reader 1 GPA`, and `Reader 1 Recommendations`. The four variables either exist or miss together. These 22 students might be evaluated by other readers. 
# 
# 10.6% of the students don't have the GRE scores. 2 observations have GRE scores but miss the GRE Quantitative Percentile1.

# ##### Impute GPA

# In[405]:


#Table with valid 'Reading 1 GPA' and valid 'College Degree GPA'
GPAValid=df[(df['Reader 1 GPA'].isnull()==False)& (df['College Degree GPA'].isnull()==False)]


# In[406]:


GPAValid.groupby('Reader 1 GPA')[['Status','College Degree GPA',
                                  'GRE Quantitative Percentile1','GRE Verbal Percentile1']].mean()


# In[407]:


#Table with valid 'Reading 1 GPA' and missing 'College Degree GPA'
GPAmiss=df[(df['Reader 1 GPA'].isnull()==False)& (df['College Degree GPA'].isnull()==True)]
GPAmiss[['Reader 1 GPA','College Degree GPA']]


# In[408]:


for i in GPAmiss.index:
    if df.iloc[i]['Reader 1 GPA']=='A':
        df=df.set_value(i,'College Degree GPA',3.81) 
    elif df.iloc[i]['Reader 1 GPA']=='B':
        df=df.set_value(i,'College Degree GPA',3.58)    
    elif df.iloc[i]['Reader 1 GPA']=='C':
        df=df.set_value(i,'College Degree GPA',3.27)
    else:
        df=df.set_value(i,'College Degree GPA',2.70)


# In[409]:


#The remaining missing GPA. Fill it with the GPA of the student who has the exactly same GRE scores.
df[df['College Degree GPA'].isnull()==True]


# In[410]:


guess=df[(df['GRE Quantitative1']==158) & (df['GRE Verbal1']==156)]['College Degree GPA'].iloc[0]
df=df.set_value(16, 'College Degree GPA', guess) #Quant Score: 162


# The `Reader 1 GPA` is highly correlated with the `College Degree GPA`. So it does not add much information. However, it's useful to infer 11 out of 12 missing GPA values with the average GPA associated with the different `Reader 1 GPA`. For the only missing GPA left, I used the GPA of the student who has the same GRE scores with her.

# ##### Impute GRE scores

# The missing values in four columns related with GRE indicate these students don't have the GRE scores. They might take GMAT or LSAT in lieu of GRE.

# In[411]:


#Locate and fix the observations with GRE scores but missing GRE Quantitative Percentile1
df[(df['GRE Quantitative1'].isnull()==False) & 
   (df['GRE Quantitative Percentile1'].isnull()==True)]
p1=round(np.mean(df[df['GRE Quantitative1']==155]['GRE Quantitative Percentile1']),0)
p2=round(np.mean(df[df['GRE Quantitative1']==162]['GRE Quantitative Percentile1']),0)
df=df.set_value(84, 'GRE Quantitative Percentile1', p1) #Quant Score: 155
df=df.set_value(128, 'GRE Quantitative Percentile1', p2) #Quant Score: 162


# In[412]:


#Table with valid GRE scores
GREValid=df[(df['GRE Quantitative Percentile1'].isnull()==False)]

plt.scatter(GREValid['College Degree GPA'],GREValid['GRE Verbal Percentile1'])
plt.title("No linear relationship between GPA and GRE Verbal Percentile")
plt.xlabel('College Degree GPA')
plt.ylabel('GRE Verbal Percentile')
plt.show()

plt.scatter(GREValid['College Degree GPA'],GREValid['GRE Quantitative Percentile1'])
plt.title("No linear relationship between GPA and GRE Quant Percentile")
plt.xlabel('College Degree GPA')
plt.ylabel('GRE Quant Percentile')
plt.show()


# The scatterplots suggest it's not a good idea to impute missing GREs from GPAs. So I turned to use the `Reader 1 Academic`.

# In[413]:


GPAValid.groupby('Reader 1 Academic')[['Status','College Degree GPA',
                                  'GRE Quantitative Percentile1','GRE Verbal Percentile1']].mean()


#    There are significant gaps between GRE scores among the three groups. So `Reader 1 Academic` is useful for inferring missing GRE scores.

# In[414]:


#Table with valid 'Reader 1 Academic' and missing 'GRE'
GREmiss=df[(df['Reader 1 Academic'].isnull()==False) &
           (df['GRE Quantitative Percentile1'].isnull()==True)]
GREmiss[['Reader 1 Academic','GRE Quantitative Percentile1']]


# In[415]:


for i in GREmiss.index:
    if df.iloc[i]['Reader 1 Academic']=='A':
        df=df.set_value(i,'GRE Quantitative Percentile1',82.07)
        df=df.set_value(i,'GRE Verbal Percentile1',88.07)

    elif df.iloc[i]['Reader 1 Academic']=='B':
        df=df.set_value(i,'GRE Quantitative Percentile1',72.50)
        df=df.set_value(i,'GRE Verbal Percentile1',86.79)
    else:
        pass


# In[416]:


df['GRE Quantitative Percentile1']=df['GRE Quantitative Percentile1'].fillna(df['GRE Quantitative Percentile1'].mean())
df['GRE Verbal Percentile1']=df['GRE Verbal Percentile1'].fillna(df['GRE Verbal Percentile1'].mean())


# Using the same logic as before, I imputed the 11 missing `GRE Quantitative Percentile1` and `GRE Verbal Percentile1` from the average scores associated with three `Reader 1 Academic` groups. I filled the remaining 4 missing GRE scores with the average values.

# ### Feature Engineering

# In[438]:


#Create a new dataframe to contain the y and X for predictive models

#Outcome variable
tb=pd.DataFrame({'Status':df['Status']}) 

#Academic measures
tb['GPA']=df['College Degree GPA']
tb['Quant']=df['GRE Quantitative Percentile1']
tb['Verbal']=df['GRE Verbal Percentile1']
tb['Fin']=df['App - Financial - Amount Total']/5000

#Reader 1
#Binary variable for whether leadership grade is C or not
tb['LeaderC']=np.where(df['Reader 1 Leadership']=='C', 1,0)
#Binary variable for whether recommendations grade is A or not
tb['RecomA']=np.where(df['Reader 1 Recommendations']=='A', 1,0)

#Demographics
#Assumption: All missing values in 'Region','Sex','Hispanic' are treated as False
tb['Age']=df['Age']
#Binary variable for whether from ILLINOIS or not
tb['Local']=np.where(df['Region']=='IL', 1,0) 
#Binary variable for whether the student is male or not
tb['Gender']=np.where(df['Sex']=='M', 1,0)
#Binary variable for whether the student is Hispanic or not
tb['Hisp']=np.where(df['Hispanic']=='Yes', 1,0)
#Binary variable for whether the student is Asian or not
df['Race1']=df['Race'].map(lambda x: str(x).split(" ")[0])
tb['Asian']=np.where(df['Race1']=='Asian,',1,0)


#Professional experience
tb['Work']=df['Post-Bac Work'].map(lambda x: str(x).split(" ")[0])
#Take average for the range of values.
tb['Work'] = tb.apply(lambda row: 0.5*(int(row['Work'][0])+int(row['Work'][2]))
                    if '-' in row['Work'] else row['Work'], axis=1) 

work7=tb[tb['Work']=='7+']
work10=tb[tb['Work']=='10+']
tb=tb.set_value(work7.index, 'Work',8) #Replace "7+" with 8
tb=tb.set_value(work10.index, 'Work',11) #Replace "10+" with 11
#The missing 'Post-Bac Work' are associated with students with ages 22 or 23
#This group clearly skip the question as don't have any working experience
tb=tb.astype(float)
tb['Work']=tb['Work'].fillna(0,inplace=False) 


# In[418]:


#Make sure all variables are float type
tb.info()


# In[419]:


#Make sure there is no missing values
tb.isnull().sum()


# In[420]:


#Names of independent variables
tb.columns[1:]


# ### Predictive models

# In[421]:


#Impact Packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# ##### Feature Selection using Random Forest

# In[422]:


from sklearn.ensemble import RandomForestClassifier
feature_names = [i for i in tb.columns[1:]]
X = tb[feature_names]
y = tb['Status']
rfc= RandomForestClassifier(n_estimators=500,n_jobs=-1).fit(X, y)
for name,score in zip(feature_names,rfc.feature_importances_):
    print(name,score)


# Random Forest measures the relative importance of each predictor by counting the number of associated training samples in the 500 trees. The five most importance features are `GPA` (19%), `Verbal` (16%), `Quant` (14%), `Age` (13%), `Work` (12%).

# ##### Training and testing sets

# In[423]:


feature_names = ['GPA','Verbal','Quant', 'Age', 'Work']
X = tb[feature_names]
y = tb['Status']

#Divide the data into training and set test, with ratio 3:1.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ##### Logistic Regression

# In[424]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(X_train, y_train)


# ##### Random Forest

# In[425]:


from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier(n_estimators=500,n_jobs=-1).fit(X_train, y_train)


# ##### Support Vector Machine

# In[426]:


from sklearn.svm import SVC
svm = SVC().fit(X_train, y_train)


# ##### K-Nearest Neighbors

# In[427]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier().fit(X_train, y_train)


# ##### Linear Discriminant Analysis

# In[428]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis().fit(X_train, y_train)


# ### Evaluate models

# Cross-validation is a good way to evaluate the performance of classification model. This method splits the training data into K folds and evaluates the classification accuracy on each of the fold with the models trainined on K-1 folds. 

# In[429]:


model=[logreg,rfc,svm,knn,lda]
num=len(model)
modelnames=["Logistic Regression","Random Forest","Support Vector Machine",
            "K-Nearest Neighbors","Linear Discriminant Analysis"]


# In[431]:


#CV scores
#I choose K=3 to make sure there are enough observations in each fold.
CVScore=[np.mean(cross_val_score(i,X_train,y_train,cv=3,scoring="accuracy")) for i in model]

#Predictions of all models
pred=[cross_val_predict(i,X_train,y_train,cv=3) for i in model]

#Confusion Matrix
mat=[confusion_matrix(y_train,pred[i]) for i in range(num)]

#True Negatives
TN=[mat[i][0,0] for i in range(num)]

#False Positives
FP=[mat[i][0,1] for i in range(num)]

#False Negatives
FN=[mat[i][1,0] for i in range(num)]

#True Positives
TP=[mat[i][1,1] for i in range(num)]

#Precision: TP/(TP+FP)
perc=[precision_score(y_train,pred[i]) for i in range(num)]

#Recall: TP/(TP+FN)
recall=[recall_score(y_train, pred[i]) for i in range(num)]

#F1=2*(precision*recall)/(precision+recall)
f1=[f1_score(y_train,pred[i]) for i in range(num)]

#The area under the ROC curv
auc=[roc_auc_score(y_train,pred[i]) for i in range(num)]


# In[432]:


#Performance Table
perf=pd.DataFrame({"CV Scores":CVScore},index=modelnames)
perf["True Negatives"]=TN
perf["False Positives"]=FP
perf["False Negatives"]=FN
perf["True Positives"]=TP
perf["Percision"]=perc
perf["Recall"]=recall
perf["F1-Score"]=f1
perf["AUC"]=auc
perf.round(2)


# The precision measures how many times the predictions are correct when the model predicts a student to attend. The logistic regression model has the highest precision of 60%.
# 
# The recall measures how many students who attend are correctly identified. KNN model identifies 52% of the students who attend, which slightly outperforms the score from logistic regression.
# 
# The logistic regression also has the highest cross-validation scores, F-1 score and area under the curve.

# In[433]:


for i in range(num):
    plt.plot(roc_curve(y_train,pred[i])[0],roc_curve(y_train,pred[i])[1],linewidth=2,label=modelnames[i])
plt.plot([0,1],[0,1],'k--')
plt.axis([0,1,0,1])
plt.title("ROC Curves")
plt.xlabel("False Positive Rate (1-Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.legend(loc='lower right')
plt.show()


# ### Conclusion

# In[434]:


#Coefficients for logistic regression
coeff=pd.DataFrame({'Coefficients':np.concatenate([logreg.intercept_,logreg.coef_[0]])},
                   index=['Intercept','GPA','Verbal','Quant', 'Age', 'Work'])
coeff


# The best predictive model current analysis is a logistic regression model with five predictors:
# \begin{equation}
#   Y=
#    \begin{cases}
#      1=\text{attend}, & \text{if}\ p(X)>=0.5 \\
#      0=\text{not attend}, & \text{if}\ p(X)<0.5 \\
#    \end{cases}
# \end{equation}
# 
# where
# \begin{equation}
#   p(X) = \frac{\text{exp}(0.16-0.01*\text{GPA}+0.28*\text{Verbal}-0.04*\text{Quant}-0.59*\text{Age}-0.59*\text{Work})}{1 + \text{exp}(0.16-0.01*\text{GPA}+0.28*\text{Verbal}-0.04*\text{Quant}-0.59*\text{Age}-0.59*\text{Work})} 
# \end{equation}

# The coefficients of `GPA`, `Quant`, `Age`, `Work` are negative, which indicate increasing these variables will decrease the probability of attending. 
# 
# The coefficients of `Verbal` is positive, which indicate increasing these variables will increase the probability of attending. 

# In[439]:


print('Accuracy of Logistic regression classifier on the test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))

print('Accuracy of Logistic regression classifier on the full dataset: {:.2f}'
     .format(logreg.score(X, y)))


# In[436]:


bestpred=cross_val_predict(logreg,X,y,cv=2)
mat=confusion_matrix(y,bestpred)
print(mat)
print('True negatives: {:.0f} non-attending students are predicted to not attend.'.format(mat[0,0]))
print('False positives: {:.0f} non-attending students are predicted to attend.'.format(mat[0,1]))
print('False negatives: {:.0f} attending students are predicted to not attend.'.format(mat[1,0]))
print('True positives: {:.0f} attending students are predicted to attend.'.format(mat[1,1]))


# In summary, I analyzed the demographic information of students who get offers from Harris school and performed machine learning models to predict students' attending decision. There are many interesting findings in the exploratory data analysis. Since there is missing data, I applied appropriate data imputing methods to guess the missing values. I also transformed categorical variables into binary variable. I selected five most important features by the weighted average from Random Forest. After fitting and evaluating five classification models, I found the logistic regression outperforms the other models using five variables. 
