‎df = pd.read_csv('/content/covid.csv',encoding='ISO-8859-1',parse_dates=[3,4])‎

‎df.head()‎
‎df.info()‎

‎#Preprocessing‎
‎df.describe()‎

‎#Number of unique values by columns‎
‎for i in df.columns‎:
    ‎print(i,"=>\t",len(df[i].unique()))‎

‎#Find columns with high level of NaN or missing values‎
‎df.isnull().sum()‎

‎#total de columnas‎
‎cols = df.columns.values‎


‎#total de columnas‎


‎for col in cols‎:
    ‎print(df[col].value_counts().to_frame())‎
    ‎print("----------------")‎


‎import pandas as pd‎


‎# Assuming df is your DataFrame‎


‎cols = df.columns.values‎


‎for col in cols‎:
    ‎# Compute value counts‎
    ‎counts = df[col].value_counts()‎


    ‎# Compute percentages‎
    ‎percentages = counts‎ / ‎len(df[col]) * 100‎


    ‎# Create a DataFrame with counts and percentages‎
    ‎result_df = pd.concat([counts‎, ‎percentages]‎, ‎axis=1)‎
    ‎result_df.columns = ['Counts'‎, ‎'Percentage']‎


    ‎# Print column name‎
    ‎print("Column:"‎, ‎col)‎


    ‎# Print counts and percentages‎
    ‎print(result_df)‎
    ‎print("----------------")‎


‎print('Percentage of missing values')‎
‎for col in df.columns‎ :
    ‎print('{:<20} => {:>10.2f}‎
    %'.format(col, ‎len(df[(df[col]==98) | (df[col]==99) | (df[col]==97)])/len(df)*100))‎



‎df.drop(columns={'id'},axis=1,inplace=True)‎
‎df1 = df.copy()‎


‎df1['death'] = df1['date_died'].apply(lambda x‎: ‎0 if x == '9999-99-99' else 1)‎
‎df1.loc[df1['sex']==2,'sex']=0‎
‎df1.loc[df1['patient_type']==2,'patient_type']=0‎
‎df1.loc[df1['inmsupr']==2,'inmsupr']=0‎
‎df1.loc[df1['pneumonia']==2,'pneumonia']=0‎
‎df1.loc[df1['diabetes']==2,'diabetes']=0‎
‎df1.loc[df1['asthma']==2,'asthma']=0‎
‎df1.loc[df1['copd']==2,'copd']=0‎
‎df1.loc[df1['hypertension']==2,'hypertension']=0‎
‎df1.loc[df1['cardiovascular']==2,'cardiovascular']=0‎
‎df1.loc[df1['renal_chronic']==2,'renal_chronic']=0‎
‎df1.loc[df1['obesity']==2,'obesity']=0‎
‎df1.loc[df1['tobacco']==2,'tobacco']=0‎
‎# df1.loc[df1['icu']==2,'icu']=0‎
‎df1.loc[df1['covid_res']==3,'covid_res']=0‎
‎df1.loc[df1['covid_res']==2,'covid_res']=1‎
‎df1.loc[df1['covid_res']==1,'covid_res']=2‎



‎df1['date_died'].replace('9999-99-99','30-06-2020',inplace=True)‎



‎# Define a function to convert the date format‎
‎from datetime import datetime‎
‎def convert_date(date_str)‎:
    ‎date_obj = datetime.strptime(date_str‎, ‎'%d-%m-%Y')‎
    ‎new_date_str = date_obj.strftime('%Y-%m-%d')‎
    ‎return new_date_str‎


‎# Apply the function to the 'date-died' column‎
‎df1['date_died'] = df1['date_died'].apply(convert_date)‎


‎date_fields=['entry_date','date_died']‎
‎for dates in date_fields‎:
    ‎df1[dates]=pd.to_datetime(df1[dates]‎, ‎dayfirst=True)‎


‎df1['days_prior_to_treatment'] = abs(df1['date_died']‎ - ‎df1['entry_date'])‎
‎df1['days_prior_to_treatment'] = abs(df1['date_died']‎ - ‎df1['entry_date']).dt.days‎


‎df1.rename(columns={"death"‎: ‎"status",‎ ‎"days_prior_to_treatment":"time"},‎
   ‎inplace=True) # rename duration/event cols‎
‎df1.drop(['other_disease','icu','intubed','pregnancy','contact_other_covid'‎,
   ‎'entry_date','date_symptoms','date_died'],inplace=True,axis=1)‎


‎df1.reset_index(inplace=True)‎
‎df1.drop(columns=['index'],inplace=True)‎
‎df1 = df1.astype('float32')‎

‎#Find outliers‎
‎import seaborn as sns‎
‎import matplotlib.pyplot as plt‎
‎import numpy as np‎


‎# Setting up figure size‎
‎plt.figure(figsize=(4,3))‎


‎# Plot before median imputation‎
‎sns.boxplot(df1['age'])‎
‎plt.title("Box Plot before median AGE")‎
‎plt.show()‎


‎# Calculating quartiles and median‎
‎q1 = df1['age'].quantile(0.25)‎
‎q3 = df1['age'].quantile(0.75)‎
‎iqr = q3‎ - ‎q1‎
‎Lower_tail = q1‎ - ‎1.5 * iqr‎
‎Upper_tail = q3‎ + ‎1.5 * iqr‎
‎med = np.median(df1['age'])‎


‎# Imputation‎
‎for i in df1['age']‎:
    ‎if i > Upper_tail or i < Lower_tail‎:
        ‎df1['age'] = df1['age'].replace(i‎, ‎med)‎


‎# Setting up figure size‎
‎plt.figure(figsize=(4,3))‎


‎# Plot after median imputation‎
‎sns.boxplot(df1['age'])‎
‎plt.title("Box Plot after median AGE")‎
‎plt.show()‎

‎#Normalization‎
‎from sklearn.preprocessing import Normalizer‎
‎from sklearn.preprocessing import StandardScaler‎
‎from sklearn.preprocessing import MinMaxScaler‎
‎from sklearn.preprocessing import RobustScaler‎
‎from sklearn.preprocessing import MaxAbsScaler‎
‎# scaler = MaxAbsScaler()‎
‎#Rescaling features age‎, ‎trestbps‎, ‎chol‎, ‎thalach‎, ‎oldpeak‎.
‎scaler = StandardScaler()‎
‎# scaler = MinMaxScaler(feature_range=(0,1))‎
‎# scaler =  Normalizer()‎
‎df1.age = scaler.fit_transform(df1.age.values.reshape(-1,1))‎
‎df1.head()‎


‎#Data Visualization‎
‎plt.figure(figsize=(4,3))‎
‎ax = sns.countplot(x='status'‎, ‎hue = 'status',data=df1,palette=sns.cubehelix_palette(2))‎
‎plt.bar_label(ax.containers[0])‎
‎plt.title("Death Distribution"‎, ‎fontsize=18,color="red");‎


‎plt.figure(figsize=(4,3))‎
‎sns.histplot(x=df1.age)‎
‎plt.title("Age Distribution"‎, ‎color="red"‎, ‎fontsize=18);‎


‎plt.figure(figsize=(4,3))‎
‎sns.boxplot(x="status"‎, ‎y="age",data=df1,palette=sns.color_palette(["#2f4f4f","#eedd82"]))‎
‎plt.xlabel("Death")‎
‎plt.title("Age-death",fontsize=18‎, ‎color="red");‎


‎plt.figure(figsize=(4,3))‎
‎sns.boxplot(x="status"‎, ‎y="age",hue="sex",data=df1‎,
              ‎palette=sns.color_palette(["#2f4f4f","#eedd82"]))‎
‎plt.title("age-death-sex",fontsize=18‎, ‎color="red")‎
‎plt.xlabel("Death")‎
‎plt.legend(loc="best");‎


‎plt.figure(figsize=(4,3))‎
‎sns.countplot(x='sex',hue='status'‎, ‎data=df1‎, ‎palette=sns.cubehelix_palette(2))‎
‎plt.title("Sex-Death",fontsize=18‎, ‎color="red")‎
‎plt.legend(loc="best");‎


‎plt.figure(figsize=(4,3))‎
‎ax=sns.countplot(x='obesity',hue='status'‎, ‎data=df1‎,
                 ‎palette=sns.color_palette(["#7fffd4","#a52a2a"]))‎
‎plt.title("Obesity-Death",fontsize=18‎, ‎color="red")‎
‎plt.bar_label(ax.containers[0])‎
‎plt.bar_label(ax.containers[1])‎
‎plt.legend(loc="best");‎


‎plt.figure(figsize=(4,3))‎


‎import seaborn as sns‎
‎import matplotlib.pyplot as plt‎

‎sns.countplot(x='sex'‎, ‎data=df1‎, ‎palette=sns.color_palette(["#00cc99","#b2beb5"])‎, ‎width=0.6)‎
‎# Adding title‎
‎plt.title("Counts of Sex")‎
‎# Showing the plot‎
‎plt.show()‎
‎plt.figure(figsize=(4,3))‎



‎sns.countplot(x='patient_type'‎, ‎data=df1‎,
        ‎palette=sns.color_palette(["#00cc99","#b2beb5"])‎, ‎width=0.6)‎
‎# Adding title‎
‎plt.title("Counts of patient_type")‎
‎# Showing the plot‎
‎plt.show()‎
‎plt.figure(figsize=(4,3))‎


‎sns.countplot(x='pneumonia'‎, ‎data=df1‎ 
      , ‎palette=sns.color_palette(["#00cc99","#b2beb5"])‎, ‎width=0.6)‎
‎# Adding title‎
‎plt.title("Counts of pneumonia")‎
‎# Showing the plot‎
‎plt.show()‎
‎plt.figure(figsize=(4,3))‎

‎sns.countplot(x='diabetes'‎, ‎data=df1‎
      , ‎palette=sns.color_palette(["#00cc99","#b2beb5"])‎, ‎width=0.6)‎
‎# Adding title‎
‎plt.title("Counts of diabetes")‎
‎# Showing the plot‎
‎plt.show()‎
‎plt.figure(figsize=(4,3))‎


‎sns.countplot(x='copd'‎, ‎data=df1‎
        , ‎palette=sns.color_palette(["#00cc99","#b2beb5"])‎, ‎width=0.6)‎
‎# Adding title‎
‎plt.title("Counts of copd")‎
‎# Showing the plot‎
‎plt.show()‎
‎plt.figure(figsize=(4,3))‎


‎sns.countplot(x='asthma'‎, ‎data=df1‎
        , ‎palette=sns.color_palette(["#00cc99","#b2beb5"])‎, ‎width=0.6)‎
‎# Adding title‎
‎plt.title("Counts of asthma")‎
‎# Showing the plot‎
‎plt.show()‎
‎plt.figure(figsize=(4,3))‎


‎sns.countplot(x='inmsupr'‎, ‎data=df1‎,
          ‎palette=sns.color_palette(["#00cc99","#b2beb5"])‎, ‎width=0.6)‎
‎# Adding title‎
‎plt.title("Counts of inmsupr")‎
‎# Showing the plot‎
‎plt.show()‎
‎plt.figure(figsize=(4,3))‎



‎sns.countplot(x='hypertension'‎, ‎data=df1‎,
         ‎palette=sns.color_palette(["#00cc99","#b2beb5"])‎, ‎width=0.6)‎
‎# Adding title‎
‎plt.title("Counts of hypertension")‎
‎# Showing the plot‎
‎plt.show()‎
‎plt.figure(figsize=(4,3))‎

‎sns.countplot(x='cardiovascular'‎, ‎data=df1‎,
         ‎palette=sns.color_palette(["#00cc99","#b2beb5"])‎, ‎width=0.6)‎
‎# Adding title‎
‎plt.title("Counts of cardiovascular")‎
‎# Showing the plot‎
‎plt.show()‎
‎plt.figure(figsize=(4,3))‎

‎sns.countplot(x='obesity'‎, ‎data=df1‎,
         ‎palette=sns.color_palette(["#00cc99","#b2beb5"])‎, ‎width=0.6)‎
‎# Adding title‎
‎plt.title("Counts of obesity")‎
‎# Showing the plot‎
‎plt.show()‎
‎plt.figure(figsize=(4,3))‎


‎sns.countplot(x='renal_chronic'‎,
         ‎data=df1‎, ‎palette=sns.color_palette(["#00cc99","#b2beb5"])‎, ‎width=0.6)‎
‎# Adding title‎
‎plt.title("Counts of renal_chronic")‎
‎# Showing the plot‎
‎plt.show()‎
‎plt.figure(figsize=(4,3))‎


‎sns.countplot(x='tobacco'‎, ‎data=df1‎,
         ‎palette=sns.color_palette(["#00cc99","#b2beb5"])‎, ‎width=0.6)‎
‎# Adding title‎
‎plt.title("Counts of tobacco")‎
‎# Showing the plot‎
‎plt.show()‎
‎plt.figure(figsize=(4,3))‎


‎sns.countplot(x='covid_res'‎, ‎data=df1‎,
         ‎palette=sns.color_palette(["#00cc99","#b2beb5"])‎, ‎width=0.6)‎
‎# Adding title‎
‎plt.title("Counts of covid_res")‎
‎# Showing the plot‎
‎plt.show()‎
