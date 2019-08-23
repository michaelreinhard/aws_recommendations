import pandas as pd
import numpy as np
import os 


cd = os.getcwd()
#import the data 
df = pd.read_csv(cd + '/data/reviews.csv')

print(df.head())