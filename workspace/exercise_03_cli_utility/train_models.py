"""<DOCSTRING>
"""

__author__ = 'Charlie Harrison <charliejharrison@gmail.com>'

from os.path import join as path_join

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
import pickle


if __name__ == '__main__':

    #######################
    # Language classifier #
    #######################
    print("Language classifer:")

    # Load data
    print("\tloading data...")
    lang_data_folder = '../../data/languages/paragraphs'
    lang_data = load_files(lang_data_folder)

    # Train on whole dataset
    print("\ttraining...")
    X_lang, y_lang = lang_data.data, lang_data.target
    lang_pipeline = Pipeline([('vect', TfidfVectorizer(analyzer='char',
                                                       ngram_range=(1, 3))),
                              ('esti', Perceptron())])
    lang_pipeline.fit(X_lang, y_lang)

    # Pickle result
    print("\tsaving result...")
    pickle_folder = 'serialised/'
    with open(path_join(pickle_folder, 'language_clf.p'), 'wb') as fi:
        pickle.dump(lang_pipeline, fi)

    ######################
    # Sentiment detector #
    ######################
    print("Sentiment detector:")

    # Load data
    print("\tloading data...")
    sent_data_folder = '../../data/movie_reviews/txt_sentoken'
    sent_data = load_files(sent_data_folder)

    # Train on whole dataset
    print("\ttraining...")
    X_sent, y_sent = sent_data.data, sent_data.target
    sent_vect = TfidfVectorizer
    sent_pipeline = Pipeline([('vect', TfidfVectorizer(min_df=1, max_df=0.9,
                                                       ngram_range=(1, 3))),
                              ('esti', LinearSVC())])
    sent_pipeline.fit(X_sent, y_sent)

    # Pickle result
    print("\tsaving result...")
    with open(path_join(pickle_folder, 'sentiment_clf.p'), 'wb') as fi:
        pickle.dump(sent_pipeline, fi)


    print("All done!")