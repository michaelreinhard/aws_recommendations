import pandas as pd
import numpy as np
import os 


cd = os.getcwd()
#import the data 
df = pd.read_csv(cd + '/data/reviews.csv')

#combine all the text into one variable
df["text_all"] = df.Summary.str.cat(df.Text, sep = ' . ')

#confirm that they are all stings
# print(type(df.text_all[0]))

print(df.columns)
#get rid of all the variables we will not be using
df = df.drop(['Text', 'Id', 'ProductId', 'UserId',\
              'ProfileName', 'Time'], axis=1)

print(df.columns)

