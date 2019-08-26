import sys
import numpy as np
import pandas as pd
import re
import sqlalchemy 
import nltk
import string
import spacy
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle

nlp = spacy.load('en')



def text_cleaner(df, variable='text_all', lemma_stopword=True, only_stopword=False):
    '''
    Takes in a dataframe as an argument and goes through spaCy procedures to 
    tokenize, lemmatize and remove stop_words. Returns a cleaned string for each 
    review in the data set.
    '''
    cleaned_text = []
    
    regex = re.compile(r'<span.*\/span>|<br.\/>|<\/a >|<a href=.+?\s>')
    
    
    #takes avariable specified in arguments to iterate through
    for text in df[variable]:
        
        text = str(text) #just to make sure it is a string
        
        text = re.sub(regex,'',text)

        text = nlp(text)
        if lemma_stopword==lemma_stopword:
            cleaned = [token.lemma_ for token in text if token.is_punct==False and token.is_stop==False]
            cleaned_text.append(' '.join(cleaned))
        elif only_stopword==only_stopword:
            cleaned = [token.text for token in text if token.is_punct==False and token.is_stop==False]
            cleaned_text.append(' '.join(cleaned))
        else: 
            cleaned_text = text
    # print(len(cleaned_text))
    new_variable = "{}_cleaned".format(variable)
    df[new_variable] = cleaned_text
    return df



def load_data(database_filepath):
    '''
    input: (
        database_filepath: path to database
            )
    Loads data from sqlite database 
    output: (
        X: features dataframe
        y: target dataframe
        category_names: list of target names
        )
    '''
    engine = sqlalchemy.create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('reviews', engine) 

    df = df.sample(1000)

    df = text_cleaner(df, variable='text_all', lemma_stopword=True, only_stopword=False)
    
    X = df.loc[:,'text_all_cleaned']
    y = df.loc[:,'positive']


    # category_names = list(y.columns.values)
    return X, y
    


# def tokenize(text):
#     '''
#     input: (
#         text: raw text data
#             )
#     output: (
#         returns cleaned tokens in list 
#             )
#     Function normalizes, tokenizes, and lemmatizes the text and
#     removes stopwords.
#     '''
#     stop_words = stopwords.words("english")
#     lemmatizer = WordNetLemmatizer()
#     # normalize case and remove punctuation
#     text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
#     # tokenize text
#     tokens = word_tokenize(text)
#     # lemmatize andremove stop words
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
#     return tokens

def build_model():
    '''
    Searches via GridSearch for the best model.
    Input: (
        none, 
        )
    Output: (
        cv: GridSearchCV object) 
    '''
    pipeline = Pipeline([
        # ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.5)),
        # ('tfidf', TfidfTransformer(use_idf=True)),
        # ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=250,\
            # max_features='log2')))
        ('tfidf', TfidfVectorizer(max_features=400, ngram_range=(1,3), max_df=0.5)), 
        ('reduce_dim', TruncatedSVD(n_components=140)), 
        ('clf', SVC(probability=True, gamma=0.0001, C=1000))
        
    ])
    
    return pipeline 
    
def evaluate_model(model, X_test, y_test):
    '''
    Input: (
        model: a pipeline as defined by build_model(),
        X_test: values of a dataframe defined by train_test_split() below,
        y_test: values of a dataframe,
        category names: list defined in load_data() function
        )
    Output: (
        y_pred: predicted values for X_test,
        a confusion matrix of the results
        )
        
    '''
    y_pred = model.predict(X_test)
    # for i, label in enumerate(category_names):
    #     print(label)
    #     print(confusion_matrix(y_pred[:,i] ,y_test.values[:,i]))
    print(accuracy_score(y_test, y_pred))


def save_model(model, model_filepath):
    '''
    input: (
        model: trained model 
        model_filepath: filepath to save model in flattened, serialized form 
            )
    Saves the model to a Python pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()