from pyspark.sql.types import *

from pyspark.sql import Row
from pyspark import SparkContext, SparkConf, HiveContext
get_ipython().magic(u'matplotlib inline')

import matplotlib.pyplot as plt
import seaborn as sb

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier

conf = SparkConf().setAppName("Telco Churn IRL")
sc = SparkContext(conf=conf)
sqlContext = HiveContext(sc)

data_set = sqlContext.sql("select * from jfletcher.churn_test_3")
data_set.printSchema()
data_set = data_set.na.fill(0) #filling the NA values with 0





import os
import numpy as np
import pandas as pd

from churnexplainer.utils import data_dir
from churnexplainer.data import utils

from pyspark.sql import SparkSession, SQLContext


ibmxlsxpath = os.path.join(data_dir, 'raw', 'ibm.xlsx')

idcol = 'customerID'
labelcol = 'Churn'
cols = (('gender', True),
        ('SeniorCitizen', True),
        ('Partner', True),
        ('Dependents', True),
        ('tenure', False),
        ('PhoneService', True),
        ('MultipleLines', True),
        ('InternetService', True),
        ('OnlineSecurity', True),
        ('OnlineBackup', True),
        ('DeviceProtection', True),
        ('TechSupport', True),
        ('StreamingTV', True),
        ('StreamingMovies', True),
        ('Contract', True),
        ('PaperlessBilling', True),
        ('PaymentMethod', True),
        ('MonthlyCharges', False),
        ('TotalCharges', False))


def drop_missing(df):
    '''Remove rows with missing values'''
    return df.replace(r'^\s$', np.nan, regex=True).dropna()


def clean(df):
    # Make target variable a true boolean column
    # Drop unpredictive column
    df.drop(['customerID'], axis=1)


def booleanize_senior_citizen(df):
    '''Make SeniorCitizen 'Yes'/'No' like other columns in this dataset.'''
    return df.replace({'SeniorCitizen': {1: 'Yes', 0: 'No'}})


def load_dataset():
    '''Return Real Telco customers and labels.'''
    #df = pd.read_excel(ibmxlsxpath)
    
    conf = SparkConf().setAppName("Telco Churn IRL")
    sc = SparkContext(conf=conf)
    sqlContext = HiveContext(sc)
    df = sqlContext.sql("select * from jfletcher.churn_test_3").toPandas()


    df = drop_missing(df).reset_index()
    df.index.name = 'id'
    features, labels = utils.splitdf(df, labelcol)
    features = booleanize_senior_citizen(features)
    features = utils.drop_non_features(features, cols)
    features = utils.categorize(features, cols)
    labels = (labels == 'Yes')
    return features, labels

