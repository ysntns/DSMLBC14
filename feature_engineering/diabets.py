#############################################
# DIABET FEATURA ENGINEERING
#############################################

'''
Work Problem

Predicting whether people have diabetes when their characteristics are specified
to develop a machine learning model that is capable of learning the data. Model
the data analysis and feature engineering steps required before development.
you are expected to realize
'''

'''
The Dataset Story

The dataset is part of a larger dataset held at the National Institutes of Diabetes-Digestive-Kidney Diseases in the US. In the US
Pima Indian women aged 21 years and older living in Phoenix, the 5th largest city in the State of Arizona
are the data used for diabetes research.
The target variable is specified as "outcome", where 1 indicates a positive diabetes test result and 0 indicates a negative result.
'''

'''
9 Variables 768 Observations 24 KB
Pregnancies                      :Number of pregnancies
Glucose                            :2-hour plasma glucose concentration in oral glucose tolerance test
Blood Pressure                   :Blood Pressure (small blood pressure) (mm Hg)
SkinThickness                    :Skin Thickness
Insulin                              :2-hour serum insulin (mu U/ml)
DiabetesPedigreeFunction    :Function (2-hour plasma glucose concentration in oral glucose tolerance test)
BMI                                 :Body mass index
Age                                 :Age (years)
Outcome                          :Have the disease (1) or not (0)
'''

#############################################
# Task 1 : Exploratory Data Analytics
#############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter(action="ignore")


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#Step 1: Examine the overall picture.

df = pd.read_csv(r'diabetes.csv')

def check_df(dataframe, head=5):
    print("############## Shape #############")
    print(dataframe.shape)
    print("############## Types #############")
    print(dataframe.dtypes)
    print("############## Tail #############")
    print(dataframe.tail(head))
    print("############## NA #############")
    print(dataframe.isnull().sum())
    print("############## Quantiles #############")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)
#Step 2: Capture numeric and categorical variables.
#Step 3: Analyze the numeric and categorical variables.
#Step 4: Analyze the target variable (according to categorical variables mean of the target variable, relative to the target variable average of numeric variables)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car, num_but_cat
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car


#Step 5: Analyze outlier observations.
sns.boxplot(x=df["Outcome"])
plt.show()

'''
q1 = df["Outcome"].quantile(0.25)
q3 = df["Outcome"].quantile(0.75)
iqr = q3 - q1 ### IQR CALCULATION
up = q3 + 1.5* iqr ## FOR UPPER LIMIT
low = q1 - 1.5 * iqr ## LOWER LIMIT

df[(df["Outcome"] < low) | (df["Outcome"] > up)]

df[(df["Outcome"] < low) | (df["Outcome"] > up)].index ## WRITTEN FOR INDEX OF INVALID VALUES

df[(df["Outcome"] < low) | (df["Outcome"] > up)].any(axis=None)
df[(df["Outcome"] < low)].any(axis=None)
'''

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "Outcome")
outlier_thresholds(df, "Glucose")

#Step 6: Perform missing observation analysis.
df.isnull().values.any()


#Step 7: Perform correlation analysis.

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

#############################################
# Task 2 : FEATURA ENGINEERING
#############################################

#Step 1: Take necessary actions for missing and outliers. There are no missing observations in the data set, but observation units with 0 values in Glucose,
# Insulin, etc. observation units containing a value of 0 in variables such as Glucose, Insulin, etc. may represent a missing value.
# For example; glucose or insulin value of a person is 0 will not be possible. Taking this into account, we assign the zero values as
# NaN in the relevant values and then assign the missing You can apply the operations to the values.
df[['Glucose', 'Insulin']] = df[['Glucose', 'Insulin']].replace(0, np.nan)
df.isnull().sum()
df.fillna(df.mean(), inplace=True)

#Step 2: Create new variables.
df['BMI_Age_Interact'] = df['BMI'] * df['Age']

#Step 3: Perform encoding operations.


#Step 4: Standardize for numeric variables.
rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.head()
df.describe().T

#Step 5: Create a model.
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# How about the new variables we just created?
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)