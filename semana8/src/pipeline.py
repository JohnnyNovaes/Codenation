from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

'''
Pipeline class. Manage all (categorical and numerical) features
in the dataset Enem_2016. 
'''

class FullPipeline():

    def __init__(self,df,feat_num,feat_cat):
        self.feat_num = feat_num
        self.feat_cat = feat_cat
        self.df = df

    def full_pipeline(self):
        num_pipeline = Pipeline([
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('std_scaler', StandardScaler())])
        
        pipeline = ColumnTransformer(
            [('num',num_pipeline,list(self.feat_num.columns)),
             ('cat_onehot',OneHotEncoder(),list(self.feat_cat.columns))
            ])
        return pipeline.fit_transform(self.df)