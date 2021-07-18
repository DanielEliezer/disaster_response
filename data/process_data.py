import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    """
    Load Data: Receives the path to the messages and categories files, and return a dataframe, with after joining the data.    
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df 

def clean_data(df):
    """
    Clean Data Function: Split the categories column into multiple columns, remove duplicates and drop nulls.
    """
    # get the name of the categories, to rename the dataframe columns
    aux_categories= df['categories'].str.split(';')[0]
    lista = []
    for item in aux_categories:
        lista.append(item.split('-')[0])

    # create a separated dataframe, with the categories results 
    categories = df['categories'].str.split(';', expand = True)
    categories.columns = lista

    # set each value to be the last character of the string
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)

    # join the new dataframe with the old one, drop nulls and duplicates
    df = df.drop('categories', axis = 1)
    df = pd.concat([df, categories], axis = 1)
    df = df.drop_duplicates()
    df = df[df['offer'].isnull()==False]
        
    # fix the 'related' field
    df.loc[df['related']== 2, 'related'] = 1
    return df

def save_data(df, database_filename):
    """
    Save the dataframe in a convenient path
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Disasters', engine, index=False, if_exists='replace')
      

def main():
    """
    Load the data, split the categories column into multiple columns, remove duplicates, drop nulls, and save the result in a database
    """


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