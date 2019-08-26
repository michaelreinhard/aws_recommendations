import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Inputs: (
        messages_filepath: path to disaster_messages.csv
        categories_filepath: path to disaster_categories.csv
    Output: 
        df: a pandas DataFrame
        )
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    '''
    Input: 
        df: Pandas DataFrame
    Output: 
        df: a cleaned Pandas DataFrame
    Makes the columns names the list of categories and makes 
    the values of the columns into 1s and 0s. 
    '''
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[:1,:]
    my_list = []
    for i in row:
        my_list.append(row[i][0][:-2])
    category_colnames = my_list
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates(subset='message', keep='last')
    return df
    


#changing second argument to database_filepath to match how it is
#named below when the function is called. Originally 'database_filename'
def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()