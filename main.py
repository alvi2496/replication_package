import warnings
import pandas as pd
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from progressbar import progressbar
from sklearn import tree, naive_bayes, ensemble, svm, linear_model
from sklearn.model_selection import cross_val_score
from prettytable import PrettyTable
from imblearn.over_sampling import SMOTE
from sklearn import metrics
import pdb
from gensim.models.keyedvectors import KeyedVectors


warnings.filterwarnings("ignore")
brunet_data_location = "files/datasets/Brunet2014.csv"
cc_data_location = "files/datasets/code_comments.csv" # cc = code_comments
td_data_location = "files/datasets/tech_debt.csv"


def structure(data_file_path):
    data = pd.read_csv(data_file_path, sep=",", header=None, names=['text', 'label'])
    return data


data = structure(brunet_data_location)
# print(data.head())


STOPSET_WORDS = ['might', 'may', 'would', 'must', 'lgtm', 'could', 'can', 'good', 'great', 'nice', 'well', \
                 'better', 'worse', 'worst', 'should', 'i', "i'll", "ill", "it's", "its", "im", "i'm", \
                 "they're", "theyre", "you're", "youre", "that's", 'btw', "thats", "theres", "shouldnt", \
                 "shouldn't", "didn't", "didnt", "dont", "don't", "doesn't", "doesnt", "wasnt", "wasn't", \
                 'sense', "mon", 'monday', 'tue', 'wed', 'wednesday', 'thursday', 'lgtm', 'pinging', 'thu', 'friday', 'fri', \
                 'sat', 'saturday', 'sun', 'sunday', 'jan', 'january', 'feb', 'february', 'mar', 'march', \
                 'apr', 'april', 'may', 'jun', 'june', 'july', 'jul', 'aug', 'august', 'sep', 'september', \
                 'oct', 'october', 'nov', 'novenber', 'dec', 'december', 'pm', 'am', '//'
]


def remove_stopwords(data):
    stopset = set(stopwords.words('english'))
    for word in STOPSET_WORDS:
        stopset.add(word)

    data['text'] = data['text'].apply(lambda sentence: ' '.join([word for word in sentence.lower().split() \
                                                                 if word not in (stopset)]))
    return data


data = remove_stopwords(data)
# print(data.head())


def count_vectorizer(data):
    count_vect = CountVectorizer()
    data_count = count_vect.fit_transform(data['text'])

    return data_count


def tf_idf_vectorizer(data):
    tf_idf_vector = TfidfVectorizer()
    data_tf_idf = tf_idf_vector.fit_transform(data['text'])

    return data_tf_idf


WIKI_WORDS = 'files/pre_trained_word_vectors/wiki-news-300d-1M.vec'
GLOVE_WORDS = 'files/pre_trained_word_vectors/glove.42B.300d.txt'
WORD_DICTIONARY = {}


def make_word_dictionary():
    print('Loading word-embedding file and making dictionary...')
    for line in progressbar(open(WIKI_WORDS)):
        values = line.split()
        WORD_DICTIONARY[values[0]] = np.array(values[1:], dtype='float32')


def word_embed(data):
    # load the pre-trained word-embedding vectors
    if len(WORD_DICTIONARY) == 0:
        make_word_dictionary()
    word_vector = np.zeros(np.array((data.shape[0], 300)))
    i = 0
    print('embedding word and converting to vector...')
    for sentence in data['text']:
        words = sentence.split()
        for word in words:
                if word in WORD_DICTIONARY:
                        word_vector[i] = np.add(word_vector[i], WORD_DICTIONARY[word])
        i += 1
    print("Word Embedding is completed")
    return word_vector


def inject_similar_words(data):
    # load the pre-trained similar word vector
    print("Loading similar word vector....")
    word_vect = KeyedVectors.load_word2vec_format("files/pre_trained_word_vectors/SO_vectors_200.bin", binary=True)
    print('injecting similar words......')
    sentences = []
    for sentence in progressbar(data['text']):
        injected_words = ''
        words = sentence.split()
        for word in words:
            try:
                similar_pairs = word_vect.most_similar(word)
                for similar_word in similar_pairs:
                    if similar_word[1] >= 0.7 and similar_word[0] not in injected_words and similar_word[0] not in words:
                        injected_words = injected_words + ' ' + similar_word[0]
            except KeyError as e:
                continue
        sentences.append(sentence + injected_words)
    data['text'] = sentences
    print("similar word injection completed.")
    return data


data = inject_similar_words(data)

feature_vectors = {}

feature_vectors['count'] = count_vectorizer(data)
feature_vectors['tf-idf'] = tf_idf_vectorizer(data)
feature_vectors['word-embedd'] = word_embed(data)


# our cross verifier method
def cross_verifier(classifier, text, label, n):
    result_array = cross_val_score(classifier, text, label, cv=n)
    result = sum(result_array) / n
    return result_array, round(result, 4)


# cv -> cross validation folds
def cross_verification(label, feature_vectors, classifier, cv, nb):
    count_resut_array, count_result = cross_verifier(classifier, \
        feature_vectors['count'], label, cv)
    tf_idf_result_array, tf_idf_result = cross_verifier(classifier, \
        feature_vectors['tf-idf'], label, cv)
    if nb:
        word_embedd_result = 0
    else:
        word_embedd_result_array, word_embedd_result = cross_verifier(classifier, \
            feature_vectors['word-embedd'], label, cv)
    return count_result, tf_idf_result, word_embedd_result


def cross_verification_result(feature_vectors, label):
    result = {}

    # cv -> cross validation, dt -> decision tree, rf -> random forest, nb -> naive bayes
    # support vector machine -> svm, logistic regression -> lr, word embedd -> we

    result['cv_dt_count'], result['cv_dt_tfidf'], result['cv_dt_we'] = \
        cross_verification(label, feature_vectors, tree.DecisionTreeClassifier(), 10, False)
    result['cv_rf_count'], result['cv_rf_tfidf'], result['cv_rf_we'] = \
        cross_verification(label, feature_vectors, ensemble.RandomForestClassifier(), 10, False)
    result['cv_nb_count'], result['cv_nb_tfidf'], result['cv_nb_we'] = \
        cross_verification(label, feature_vectors, naive_bayes.MultinomialNB(), 10, True)
    result['cv_svm_count'], result['cv_svm_tfidf'], result['cv_svm_we'] = \
        cross_verification(label, feature_vectors, svm.SVC(), 10, False)
    result['cv_lr_count'], result['cv_lr_tfidf'], result['cv_lr_we'] = \
        cross_verification(label, feature_vectors, linear_model.LogisticRegression(), 10, False)

    # table representation
    table = PrettyTable()
    table.field_names = ['', 'Count', 'TF-IDF', 'Word Embedding']
    table.add_row(['Decision Tree', result['cv_dt_count'], result['cv_dt_tfidf'], result['cv_dt_we']])
    table.add_row(['Random Forest', result['cv_rf_count'], result['cv_rf_tfidf'], result['cv_rf_we']])
    table.add_row(['Naive Bayes', result['cv_nb_count'], result['cv_nb_tfidf'], result['cv_nb_we']])
    table.add_row(['SVM', result['cv_svm_count'], result['cv_svm_tfidf'], result['cv_svm_we']])
    table.add_row(['Logistic Regression', result['cv_lr_count'], result['cv_lr_tfidf'], result['cv_lr_we']])
    print(table)


print("Before Oversample Classifying and performing cross verification...")
cross_verification_result(feature_vectors, data['label'])


def oversample(X, Y):
    sm = SMOTE(random_state=42)
    return sm.fit_resample(X, Y)


os_feature_vectors = {}
os_data = {}
os_feature_vectors['count'], os_data['label'] = oversample(feature_vectors['count'], data['label'])
os_feature_vectors['tf-idf'], os_data['label'] = oversample(feature_vectors['tf-idf'], data['label'])
os_feature_vectors['word-embedd'], os_data['label'] = oversample(feature_vectors['word-embedd'], \
                                                                          data['label'])

print("After Oversample Classifying and performing cross verification...")
cross_verification_result(os_feature_vectors, os_data['label'])


def train(classifier, train_data, train_label, is_neural_net=False):

    # fit the training dataset on the classifier
    trained_classifier = classifier.fit(train_data, train_label)

    return trained_classifier


def verifier(trained_classifier, test_data, test_label):
    predictions = trained_classifier.predict(test_data)
    return round(metrics.accuracy_score(predictions, test_label), 4)


def classify_and_verify(classifier, train_data, train_label, test_data, test_label, nb):
    count = verifier(train(classifier, train_data['count'], train_label), test_data['count'], test_label)
    tf_idf = verifier(train(classifier, train_data['tf-idf'], train_label), test_data['tf-idf'], test_label)
    if nb:
        word_vector = 0
    else:
        word_vector = verifier(train(classifier, train_data['word-embedd'], train_label), \
                               test_data['word-embedd'], test_label)

    return count, tf_idf, word_vector


test_data_location = td_data_location
test_data = structure(test_data_location)


# remove stopwords
train_data = data

test_data = remove_stopwords(test_data)

whole_data = pd.concat([train_data, test_data])


def count_vectorizer(data, train_data, test_data):
    count_vector = CountVectorizer(analyzer = 'word', token_pattern = r'\w{1,}')
    count_vector.fit(data['text'])
    train_data_count = count_vector.transform(train_data['text'])
    test_data_count = count_vector.transform(test_data['text'])

    return train_data_count,test_data_count


def tf_idf_vectorizer(data, train_data, test_data):
    tf_idf_vector = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    tf_idf_vector.fit(data['text'])
    train_data_tf_idf = tf_idf_vector.transform(train_data['text'])
    test_data_tf_idf = tf_idf_vector.transform(test_data['text'])

    return train_data_tf_idf, test_data_tf_idf

# Convert to vector
train_feature_vectors = {}
test_feature_vectors = {}
train_feature_vectors['count'], test_feature_vectors['count'] = \
count_vectorizer(pd.concat([train_data, test_data]), train_data, test_data)
train_feature_vectors['tf-idf'], test_feature_vectors['tf-idf'] = \
tf_idf_vectorizer(pd.concat([train_data, test_data]), train_data, test_data)
train_feature_vectors['word-embedd'], test_feature_vectors['word-embedd'] = \
train_feature_vectors['word-embedd'], test_feature_vectors['word-embedd'] = word_embed(train_data), word_embed(test_data)

# Oversampling

os_train_feature_vectors = {}
os_train_data = {}
os_test_feature_vectors = {}
os_test_data = {}

os_train_feature_vectors['count'], os_train_data['label'] = oversample(train_feature_vectors['count'], train_data['label'])
os_test_feature_vectors['count'], os_test_data['label'] = oversample(test_feature_vectors['count'], test_data['label'])

os_train_feature_vectors['tf-idf'], os_train_data['label'] = oversample(train_feature_vectors['tf-idf'],train_data['label'])
os_test_feature_vectors['tf-idf'], os_test_data['label'] = oversample(test_feature_vectors['tf-idf'], test_data['label'])

os_train_feature_vectors['word-embedd'], os_train_data['label'] = oversample(train_feature_vectors['word-embedd'],train_data['label'])
os_test_feature_vectors['word-embedd'], os_test_data['label'] = oversample(test_feature_vectors['word-embedd'], test_data['label'])


def test_data_verification_result(train_data, train_label, test_data, test_label):
    result = {}

    # cv -> cross validation, dt -> decision tree, rf -> random forest, nb -> naive bayes
    # support vector machine -> svm, logistic regression -> lr, word embedd -> we

    result['test_dt_count'], result['test_dt_tfidf'], result['test_dt_we'] = \
        classify_and_verify(tree.DecisionTreeClassifier(), train_data, train_label, \
                            test_data, test_label, False)

    result['test_rf_count'], result['test_rf_tfidf'], result['test_rf_we'] = \
        classify_and_verify(ensemble.RandomForestClassifier(), train_data, train_label, \
                            test_data, test_label, False)

    result['test_nb_count'], result['test_nb_tfidf'], result['test_nb_we'] = \
        classify_and_verify(naive_bayes.MultinomialNB(), train_data, train_label, \
                            test_data, test_label, True)

    result['test_svm_count'], result['test_svm_tfidf'], result['test_svm_we'] = \
        classify_and_verify(svm.SVC(), train_data, train_label, \
                            test_data, test_label, False)

    result['test_lr_count'], result['test_lr_tfidf'], result['test_lr_we'] = \
        classify_and_verify(linear_model.LogisticRegression(), train_data, train_label, \
                            test_data, test_label, False)

    # table representation
    table = PrettyTable()
    table.field_names = ['', 'Count', 'TF-IDF', 'Word Embedding']
    table.add_row(['Decision Tree', result['test_dt_count'], result['test_dt_tfidf'], result['test_dt_we']])
    table.add_row(['Random Forest', result['test_rf_count'], result['test_rf_tfidf'], result['test_rf_we']])
    table.add_row(['Naive Bayes', result['test_nb_count'], result['test_nb_tfidf'], result['test_nb_we']])
    table.add_row(['SVM', result['test_svm_count'], result['test_svm_tfidf'], result['test_svm_we']])
    table.add_row(['Logistic Regression', result['test_lr_count'], result['test_lr_tfidf'], result['test_lr_we']])
    print(table)


print("Before Oversample Classifying and performing verification...")
test_data_verification_result(train_feature_vectors, train_data['label'], test_feature_vectors, \
                              test_data['label'])


print("After Oversample Classifying and performing verification...")
test_data_verification_result(os_train_feature_vectors, os_train_data['label'], os_test_feature_vectors, \
                              os_test_data['label'])


# implementation of LOO
def loo_validation(data):
    result = {}
    result['test_dt_count'], result['test_dt_tfidf'], result['test_dt_we'] = 0, 0, 0
    result['test_rf_count'], result['test_rf_tfidf'], result['test_rf_we'] = 0, 0, 0
    result['test_nb_count'], result['test_nb_tfidf'], result['test_nb_we'] = 0, 0, 0
    result['test_svm_count'], result['test_svm_tfidf'], result['test_svm_we'] = 0, 0, 0
    result['test_lr_count'], result['test_lr_tfidf'], result['test_lr_we'] = 0, 0, 0
    for index, row in progressbar(data.iterrows()):
        d = {'text': row['text'], 'label': row['label']}
        test_data = pd.DataFrame(data=d, index=[0])
        train_data = data.drop(data.index[index])

        train_feature_vectors = {}
        test_feature_vectors = {}
        train_feature_vectors['count'], test_feature_vectors['count'] = \
        count_vectorizer(pd.concat([train_data, test_data]), train_data, test_data)
        train_feature_vectors['tf-idf'], test_feature_vectors['tf-idf'] = \
        tf_idf_vectorizer(pd.concat([train_data, test_data]), train_data, test_data)
        train_feature_vectors['word-embedd'], test_feature_vectors['word-embedd'] = \
        train_feature_vectors['word-embedd'], test_feature_vectors['word-embedd'] = word_embed(train_data), word_embed(test_data)

        os_train_feature_vectors = {}
        os_train_data = {}

        os_train_feature_vectors['count'], os_train_data['label'] = oversample(train_feature_vectors['count'], train_data['label'])

        os_train_feature_vectors['tf-idf'], os_train_data['label'] = oversample(train_feature_vectors['tf-idf'],train_data['label'])

        os_train_feature_vectors['word-embedd'], os_train_data['label'] = oversample(train_feature_vectors['word-embedd'],train_data['label'])

        cr = test_data_verification_result(os_train_feature_vectors, os_train_data['label'], test_feature_vectors, \
                                           test_data['label'])

        result['test_dt_count'] += cr['test_dt_count']
        result['test_dt_tfidf'] += cr['test_dt_tfidf']
        result['test_dt_we'] += cr['test_dt_we']
        result['test_rf_count'] += cr['test_rf_count']
        result['test_rf_tfidf'] += cr['test_rf_tfidf']
        result['test_rf_we'] += cr['test_rf_we']
        result['test_nb_count'] += cr['test_nb_count']
        result['test_nb_tfidf'] += cr['test_nb_tfidf']
        result['test_nb_we'] += cr['test_nb_we']
        result['test_svm_count'] += cr['test_svm_count']
        result['test_svm_tfidf'] += cr['test_svm_tfidf']
        result['test_svm_we'] += cr['test_svm_we']
        result['test_lr_count'] += cr['test_lr_count']
        result['test_lr_tfidf'] += cr['test_lr_tfidf']
        result['test_lr_we'] += cr['test_lr_we']

    result['test_dt_count'] /= 1000
    result['test_dt_tfidf'] /= 1000
    result['test_dt_we'] /= 1000
    result['test_rf_count'] /= 1000
    result['test_rf_tfidf'] /= 1000
    result['test_rf_we'] /= 1000
    result['test_nb_count'] /= 1000
    result['test_nb_tfidf'] /= 1000
    result['test_nb_we'] /= 1000
    result['test_svm_count'] /= 1000
    result['test_svm_tfidf'] /= 1000
    result['test_svm_we'] /= 1000
    result['test_lr_count'] /= 1000
    result['test_lr_tfidf'] /= 1000
    result['test_lr_we'] /= 1000

    # table representation
    table = PrettyTable()
    table.field_names = ['', 'Count', 'TF-IDF', 'Word Embedding']
    table.add_row(['Decision Tree', result['test_dt_count'], result['test_dt_tfidf'], result['test_dt_we']])
    table.add_row(['Random Forest', result['test_rf_count'], result['test_rf_tfidf'], result['test_rf_we']])
    table.add_row(['Naive Bayes', result['test_nb_count'], result['test_nb_tfidf'], result['test_nb_we']])
    table.add_row(['SVM', result['test_svm_count'], result['test_svm_tfidf'], result['test_svm_we']])
    table.add_row(['Logistic Regression', result['test_lr_count'], result['test_lr_tfidf'], result['test_lr_we']])
    print(table)


# uncomment the return value of test_data_verification_result
print("LOO validation...")
loo_validation(data)
