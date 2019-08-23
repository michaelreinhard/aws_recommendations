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

#get rid of all the variables we will not be using
df = df.drop(['Text', 'Id', 'ProductId', 'UserId',\
              'ProfileName', 'Time'], axis=1)

#create a small function that sets the threshold
def postitive_threshold(df, threshold):
    '''
    Input: 
        df: a Pandas DataFrame
        threshold: a number between 2 and 5 
    Sets the threshold at which a review will be coded as 
    positive or negative. For example, if set at 4 the reviews 
    that gave 4 or 5 stars will be positive and 3 and below negative.
    '''
    df.loc[df.Score >=threshold,'positive'] = int(1)
    df.loc[df.Score <threshold,'positive'] = int(0)
    return df

#set positive threshold to 4
df = postitive_threshold(df, 4)

#change data type of positive to int
df.positive = df.positive.apply(int)




