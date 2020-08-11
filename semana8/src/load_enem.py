import pandas as pd

class LoadEnem:
    
    def __init__(self):
        self.path_train = '../data/train.csv'
        self.path_test = '../data/test.csv'
    
    def load(self):
        
        train = pd.read_csv(self.path_train)
        test = pd.read_csv(self.path_test)
        return (train,test)
        