"""<DOCSTRING>
"""

__author__ = 'wah'

import os.path
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
import pickle

if __name__ == '__main__':

    LONG_TARGETS = ['Arabic',
                    'German',
                    'English',
                    'Spanish',
                    'French',
                    'Italian',
                    'Japanese',
                    'Dutch',
                    'Polish',
                    'Portuguese',
                    'Russian']

    # Load language classifier
    serialised_path = 'serialised'
    if not os.path.isdir(serialised_path):
        print("Run train_models.py first!  Models should be stored in {"
              "}".format(serialised_path))
        sys.exit()

    with open(os.path.join(serialised_path, 'language_clf.p'), 'rb') as fi:
        lang_clf = pickle.load(fi)

    print("Hello there.  Is there anything you'd like to say?")
    doc = input('...')

    lang_probs = lang_clf.decision_function([doc])[0]

    lang_idx, lang_conf = max(enumerate(lang_probs), key=lambda x: x[1])
    lang = LONG_TARGETS[lang_idx]

    print("I'm {} confident that you're speaking {}...".format(lang_conf, lang))

    if lang == 'English':
        with open(os.path.join(serialised_path, 'sentiment_clf.p'), 'rb') as fi:
            sent_clf = pickle.load(fi)

        sent = sent_clf.predict([doc])
        if sent == 0:
            print("and I think you sound upset.")
        elif sent == 1:
            print("and I think you sound happy!")
        else:
            print("and I can't read you at all. ({})".format(sent))
    else:
        print("I'm sorry, I don't speak {}".format(lang))