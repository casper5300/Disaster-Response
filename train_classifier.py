import sys
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



def load_data(database_filepath):
    '''load data from sql table and return X, y'''
    engine = create_engine('sqlite:///clean_data.db')
    df = pd.read_sql_table('clean_data',engine)
    cat_name = df.columns[4:].tolist()
    X = df['message'].values  
    y = df[cat_name].values
    
    return X, y, cat_name
    


def tokenize(text):
    '''Preprocess text data and return clean token'''
    # initial lemmatize and stopwords
    clean_token = token = []
    stop_word = stopwords.words('english')
    le = WordNetLemmatizer().lemmatize
    # normalize, remove punctuation and tokenize
    for t in text:
        token = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " " ,t.lower()))
        
        for tok in token:
            if tok not in stop_word:
                clean_token.append(le(tok).lower())
                
    return clean_token
                            
                             


def build_model():
    '''create a model pipeline'''
    model = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(MLPClassifier(),n_jobs=-1))
    ])
    
    return model
    


def evaluate_model(model, X_test, Y_test, category_names):
    '''Predict X_test use trained model and evaluate by some metrics'''
    # get prediction
    y_pred  = model.predict(X_test)
    # evaluate category by category
    accuracy_avg = 0
    for i in range(y_pred.shape[1]):
        accuracy = accuracy_score(Y_test[:,i],y_pred[:,i])
        accuracy_avg += accuracy
        print('Accuracy of {} is {}'.format(i+1,accuracy))
        print(classification_report(Y_test[:,i],y_pred[:,i]))
              
    print('Average Accuracy for 36 categories is {}'.format(accuracy_avg/36))
             
    


def save_model(model, model_filepath):
    joblib.dump(model,model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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