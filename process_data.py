import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''load two cvs files and return a merge dataframe'''
    message = pd.read_csv(messages_filepath,low_memory=False)
    categories = pd.read_csv(categories_filepath,low_memory=False)
    df = pd.merge(message,categories,on='id')
    
    return df


def clean_data(df):
    '''clean dataframe'''
    # create a df of 36 categories
    cat =  df.categories.str.split(';',expand=True)
    # extract categories names from first row
    cat_cols = cat.loc[0,:].tolist()
    # rename the columns of categories
    cat.columns = cat_cols
    # convert category values to [0,1]
    for col in cat:
        cat[col] = cat[col].str[-1].astype('int')
    # drop original categories colmn from df
    df.drop('categories',axis=1,inplace=True)
    # concate cate and df
    df = pd.concat([df,cat],axis=1)
    # drop duplicatges
    df.drop_duplicates(inplace=True)
    
    return df                  
                   
    


def save_data(df, database_filename):
    engine = create_engine('sqlite:///../data/clean_data.db')
    df.to_sql('clean_data',engine,index=False)


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