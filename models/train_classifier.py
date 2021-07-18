# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import os
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger') # download for lemmatization
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator,TransformerMixin
import pickle

################################################################################################################
# A lot of the classed and functions bellow were based on the classes from the Udacity Data Sciente Nanodegree #
################################################################################################################
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Create Verb Extractor class: Creates a new feature, bases on the starting verb of the sentence
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    """cd app
    Get the data from the 'Disasters' table, fix the 'related' field, and split the data into X and Y to feed to the ML model 
    """
    # import data from database
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = 'Disasters'
    df = pd.read_sql_table(table_name,engine)

    
    # create X,Y and category_names

    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    ''' 
    Normalize, lemmatize, and tokenize a text received.
    '''

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Return a grid search model, with a pipeline that normalize, lemmatize, tokenize, and apply TF-IDF"""

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', RandomForestClassifier())
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__min_samples_split': [2, 4, 6],
        'clf__max_features': ['auto', 'sqrt', 'log2']
        
    }

    model = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_macro', cv=2, n_jobs=-1,verbose=10)

    return model



def evaluate_model(model, X_test, y_test, category_names):
    """
    Applies a ML pipeline to the test set and reports the performance (accuracy, test, recall, f1)
    """
    y_pred = model.predict(X_test)
    class_report = classification_report(y_test, y_pred, target_names=category_names)
    print(class_report)  


def save_model(model, model_filepath):    
    """Save the model to a picke file """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """
    Load the data from the database, build, train, evaluate and save the model
    """



    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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