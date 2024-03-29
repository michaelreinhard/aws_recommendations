import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


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


def load_data(food_reviews_filepath):
    '''
    Inputs: (
        food_reviews_filepath: path to disaster_categories.csv
    Output: 
        df: a pandas DataFrame
        )
    '''
    df = pd.read_csv(food_reviews_filepath)
    
    #combine all the text into one variable
    df["text_all"] = df.Summary.str.cat(df.Text, sep = ' . ')
    
    #create the dependent variable
    df = postitive_threshold(df, threshold=4)
    
    #drop variables not used in the analysis
    df = df.drop(['Text', 'Id', 'ProductId', 'UserId', 'ProfileName', 'Time'], axis=1)
    
    return df


# def clean_data(df):
#     '''
#     Input: 
#         df: Pandas DataFrame
#     Output: 
#         df: a cleaned Pandas DataFrame
#     Makes the columns names the list of categories and makes 
#     the values of the columns into 1s and 0s. 
#     '''
#     categories = df['categories'].str.split(';', expand=True)
#     row = categories.iloc[:1,:]
#     my_list = []
#     for i in row:
#         my_list.append(row[i][0][:-2])
#     category_colnames = my_list
#     categories.columns = category_colnames
#     for column in categories:
#         # set each value to be the last character of the string
#         categories[column] = categories[column].str[-1:]
#         # convert column from string to numeric
#         categories[column] = categories[column].astype(int)
#     df.drop('categories', axis=1, inplace=True)
#     df = pd.concat([df, categories], axis=1)
#     df = df.drop_duplicates(subset='message', keep='last')
#     return df
    


#changing second argument to database_filepath to match how it is
#named below when the function is called. Originally 'database_filename'
def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('reviews', engine, index=False) 


def main():
    if len(sys.argv) == 3:

        food_reviews_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    reviews: {}\n'
              .format(food_reviews_filepath))
        df = load_data(food_reviews_filepath)

        # print('Cleaning data...')
        # df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepath the reviews data as the  '\
              'dataset as the first argument, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python data/process_data.py '\
              'data/reviews.csv ' \
              'data/Reviews.db')


if __name__ == '__main__':
    main()