import nltk
import re
import word_category_counter
import data_helper
import pickle
import sys
import zipfile
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
#import movie_reviews
from word2vec_extractor import Word2vecExtractor
#selected_features = None
w2vecextractor =  Word2vecExtractor("GoogleNews-vectors-negative300.bin")

def bin(count):
    # Just a wild guess on the cutoff
    return count if count < 4 else 5

def normalize(token, should_normalize=True):
    """
    This function performs text normalization.

    If should_normalize is False then we return the original token unchanged.
    Otherwise, we return a normalized version of the token, or None.

    For some tokens (like stopwords) we might not want to keep the token. In
    this case we return None.

    :param token: str: the word to normalize
    :param should_normalize: bool
    :return: None or str
    """
    if not should_normalize:
        normalized_token = token

    else:

        ###     YOUR CODE GOES HERE
        stopwords = nltk.corpus.stopwords.words('english')
        final = [w.lower() for w in token if w not in stopwords]

        normalized_token = []
        r = re.compile('\w')
        for item in final:
            if r.match(item):
                normalized_token.append(item)

        #raise NotImplemented

    return normalized_token



def get_words_tags(text, should_normalize=True):
    """
    This function performs part of speech tagging and extracts the words
    from the review text.

    You need to :
        - tokenize the text into sentences
        - word tokenize each sentence
        - part of speech tag the words of each sentence

    Return a list containing all the words of the review and another list
    containing all the part-of-speech tags for those words.

    :param text:
    :param should_normalize:
    :return:
    """
    words = []
    tags = []

    # tokenization for each sentence

    ###     YOUR CODE GOES HERE
    #tokenize the text into sentences
    sents = nltk.sent_tokenize(text)
    #print(sents)

    #word tokenize each sentences
    for sent in sents:
        words_in_sent = nltk.word_tokenize(sent)
        for word in words_in_sent:
            words.append(word)

    #POS tag the words of each sentences
    for sent in sents:
        word_in_sent = nltk.word_tokenize(sent)
        tag = nltk.pos_tag(word_in_sent)
        for item in tag:
            tags.append(item)

    return words, tags

def get_unigram_features(tokens):
    """
    Customized function that only gets unigram only.
    """
    feature_vectors = {}


    ###     YOUR CODE GOES HERE

    #gets unigram in text and returns the unigram to the feature
    unigram = list(nltk.ngrams(tokens, 1))
    fd = nltk.FreqDist(unigram)

    for w in unigram:
        rel_freq = fd[w]/fd.N()
        key = 'UNI_'+str(w[0])
        feature_vectors[key] = rel_freq
        #l.append((key, feature_vectors[key]))

    return feature_vectors

def get_ngram_features(tokens):
    """
    This function creates the unigram and bigram features as described in
    the assignment3 handout.

    :param tokens:
    :return: feature_vectors: a dictionary values for each ngram feature
    """
    feature_vectors = {}


    ###     YOUR CODE GOES HERE


    #gets unigram in text and returns the unigram to the feature
    unigram = list(nltk.ngrams(tokens, 1))
    fd = nltk.FreqDist(unigram)

    #gets bigram in text and returns the bigram to the feature
    bigram = list(nltk.bigrams(tokens))
    bfd = nltk.FreqDist(bigram)

    for w in unigram:
        rel_freq = fd[w]/fd.N()
        key = 'UNI_'+str(w[0])
        feature_vectors[key] = rel_freq
        #l.append((key, feature_vectors[key]))

    for w in bigram:
        w1 = w[0]
        w2 = w[1]
        rel_freq = bfd[w]/bfd.N()
        key = 'BIGRAM_'+str(w1)+"_"+str(w2)
        feature_vectors[key] = rel_freq

    return feature_vectors


def get_pos_features(tags):
    """
    This function creates the unigram and bigram part-of-speech features
    as described in the assignment3 handout.

    :param tags: list of POS tags
    :return: feature_vectors: a dictionary values for each ngram-pos feature
    """
    feature_vectors = {}

    ###     YOUR CODE GOES HERE
    #print(tags)
    only_tags = [item[1] for item in tags]

    #create a unigrams
    unigram = list(nltk.ngrams(only_tags,1))
    fd = nltk.FreqDist(unigram)

    #create a bigrams
    bigram = list(nltk.bigrams(only_tags))
    bfd = nltk.FreqDist(bigram)

    for w in unigram:
        #print(w)
        rel_freq = fd[w]/fd.N()
        key = 'UNI_POS_'+str(w[0])
        #print(key)
        feature_vectors[key] = rel_freq

    for w in bigram:
        rel_freq = bfd[w]/bfd.N()
        w1 = w[0]
        w2 = w[1]
        key = 'BI_POS_'+str(w1)+"_"+str(w2)
        feature_vectors[key] = rel_freq


    #raise NotImplemented

    return feature_vectors


def get_liwc_features(words):
    """
    Adds a simple LIWC derived feature

    :param words:
    :return:
    """
    feature_vectors = {}
    text = " ".join(words)
    liwc_scores = word_category_counter.score_text(text)

    # All possible keys to the scores start on line 269
    # of the word_category_counter.py script
    negative_score = liwc_scores["Negative Emotion"]
    positive_score = liwc_scores["Positive Emotion"]
    feature_vectors["Negative Emotion"] = negative_score
    feature_vectors["Positive Emotion"] = positive_score

    if positive_score > negative_score:
        feature_vectors["liwc:positive"] = 1
    else:
        feature_vectors["liwc:negative"] = 1

    return feature_vectors

FEATURE_SETS = {"word_pos_features", "word_features", "word_pos_liwc_features", "word_liwc_features", "bin", "embedding"}

def get_features_category_tuples(category_text_dict, feature_set):
    """

    You will might want to update the code here for the competition part.

    :param category_text_dict:
    :param feature_set:
    :return:
    """
    features_category_tuples = []
    texts = []

    assert feature_set in FEATURE_SETS, "unrecognized feature set:{}, Accepted values:{}".format(feature_set, FEATURE_SETS)

    for category in category_text_dict:
        #print(category)
        for text in category_text_dict[category]:

            words, tags = get_words_tags(text)
            words = normalize(words, True)
            feature_vectors = {}

            ###     YOUR CODE GOES HERE
            if feature_set == "word_features":
                feature_vectors.update(get_ngram_features(words))

            if feature_set == "word_pos_features":
                feature_vectors.update(get_ngram_features(words))
                feature_vectors.update(get_pos_features(tags))

            if feature_set == "word_pos_liwc_features":
                feature_vectors.update(get_liwc_features(words))
                feature_vectors.update(get_ngram_features(words))
                feature_vectors.update(get_pos_features(tags))

            #if feature_set == "word_liwc_features":
            #    feature_vectors.update(get_liwc_features(words))
            #    feature_vectors.update(get_unigram_features(words))

            features_category_tuples.append((feature_vectors, category))
            texts.append(words)
            #texts.append(text)

    if feature_set == 'bin':
        return texts
    if feature_set == 'embedding':
        return texts

    return features_category_tuples, texts

def binning(category_text_dict, selected_features):
    features_category_tuples = []
    for category in category_text_dict:
        #print(category)
        review_words = []
        for text in category_text_dict[category]:

            feature_vectors = {}
            words, tags = get_words_tags(text)
            words = normalize(words, True)

            fdist = nltk.FreqDist(words)

            for word, freq in fdist.items():
                fname = "unigram:{0}_{1}".format(word, bin(freq))
                if selected_features == None or fname in selected_features:
                    feature_vectors["unigram:{0}_{1}".format(word, bin(freq))] = 1

            features_category_tuples.append((feature_vectors, category))

    #print(features_category_tuples)
    #for item in features_category_tuples:
    #    print(item[1])
    return features_category_tuples

def word_embedding(category_text_dict, selected_features):
    feature_category_tuples = []
    for category in category_text_dict:
        for text in category_text_dict[category]:

            #words, tags = get_words_tags(text)
            #words = normalize(words, True)

            feature_dict = w2vecextractor.get_doc2vec_feature_dict(text)
            feature_vectors = {}
            for item in feature_dict:
                if selected_features == None or item in selected_features:
                    feature_vectors[item] = feature_dict[item]

            feature_category_tuples.append((feature_vectors, category))

    return feature_category_tuples

def all_liwc(category_text_dict, selected_features):
    feature_category_tuples = []
    for category in category_text_dict:
        for text in category_text_dict[category]:
            feature_dict = {}
            feature_vector = {}

            #words, tags = get_words_tags(text)
            #words = normalize(words, True)

            liwc_scores = word_category_counter.score_text(text)

            negative_score = liwc_scores["Negative Emotion"]
            positive_score = liwc_scores["Positive Emotion"]

            if positive_score > negative_score:
                feature_dict["liwc:positive"] = 1
            else:
                feature_dict["liwc:negative"] = 1

            for item in feature_dict:
                if selected_features == None or item in selected_features:
                    feature_vector[item] = feature_dict[item]

            feature_category_tuples.append((feature_vector, category))

    return feature_category_tuples


def classifier_with_bin(features_category_tuples):
    #nb_classifier = nltk.classify.NaiveBayesClassifier.train(features_category_tuples)
    dt_classifier = nltk.classify.DecisionTreeClassifier.train(features_category_tuples, entropy_cutoff = 0.05,
        depth_cutoff = 100, support_cutoff = 10)

    f = open('dt_bin_word.pickle', 'wb')
    pickle.dump(dt_classifier, f)
    f.close()

    #f = open('nb_bin_word.pickle', 'wb')
    #pickle.dump(nb_classifier, f)
    #f.close()

def classifier_with_embedding(features_category_tuples):
    #nb_classifier = nltk.classify.NaiveBayesClassifier.train(features_category_tuples)
    dt_classifier = nltk.classify.DecisionTreeClassifier.train(features_category_tuples, entropy_cutoff = 0.05,
        depth_cutoff = 100, support_cutoff = 10)

    f = open('dt_embedding_word.pickle', 'wb')
    pickle.dump(dt_classifier, f)
    f.close()

    #f = open('nb_embedding_word.pickle', 'wb')
    #pickle.dump(nb_classifier, f)
    #f.close()

def create_pickle(classifier, filename):
    f = open(filename, 'wb')
    pickle.dump(classifier, f)
    f.close()

def write_features_category(features_category_tuples, outfile_name):
    """
    Save the feature values to file.

    :param features_category_tuples:
    :param outfile_name:
    :return:
    """
    with open(outfile_name, "w", encoding="utf-8") as fout:
        for (features, category) in features_category_tuples:
            fout.write("{0:<10s}\t{1}\n".format(category, features))

def training_the_data(features_category_tuples, outfile_name):
    classifier = nltk.classify.NaiveBayesClassifier.train(features_category_tuples)
    create_pickle(classifier, outfile_name)
    return classifier

def dt_training(features_category_tuples, outfile_name):
    classifier = nltk.classify.DecisionTreeClassifier.train(features_category_tuples, entropy_cutoff = 0.05,
        depth_cutoff = 100, support_cutoff = 10)
    create_pickle(classifier, outfile_name)
    return classifier

def getting_most_informative(features_category_tuples, outfile_name):
    training_data = []
    for (features, category) in features_category_tuples:
        tup = (features, category)
        training_data.append(tup)

    classifier = nltk.classify.NaiveBayesClassifier.train(training_data)

    stdout = sys.stdout
    sys.stdout = open(outfile_name, 'w')
    classifier.show_most_informative_features()

    sys.stdout.close()
    sys.stdout = stdout

def training_through_informative(tr_data, d_data, t_data, method):
    best = (0.0, 0)
    train_data = None
    develop_data = None
    test_data = None

    if method == 'binning':
        train_data = binning(tr_data, None)
    if method == 'embedding':
        train_data = word_embedding(tr_data, None)
    if method == 'liwc':
        train_data = all_liwc(tr_data, None)

    classifier = nltk.NaiveBayesClassifier.train(train_data)
    #returns classifier and has to be NaiveBayesClassifier
    best_features = classifier.most_informative_features(10000)
    for i in [2**i for i in range(5, 15)]:
        selected_features = set([fname for fname, value in best_features[:i]])
        if method == 'binning':
            train_data = binning(tr_data, selected_features)
            develop_data = binning(d_data, selected_features)
            test_data = binning(t_data, selected_features)

        if method == 'embedding':
            train_data = word_embedding(tr_data, selected_features)
            develop_data = word_embedding(d_data, selected_features)
            test_data = word_embedding(t_data, selected_features)

        if method == 'liwc':
            train_data = all_liwc(tr_data, selected_features)
            develop_data = all_liwc(tr_data, selected_features)
            test_data = all_liwc(tr_data, selected_features)

        classifier = nltk.NaiveBayesClassifier.train(train_data)
        accuracy = nltk.classify.accuracy(classifier, develop_data)
        print("{0:6d} {1:8.5f}".format(i, accuracy))

        if accuracy > best[0]:
            best = (accuracy, i)

    selected_features = set([fname for fname, value in best_features[:best[1]]])

    classifier = nltk.NaiveBayesClassifier.train(train_data)
    accuracy = nltk.classify.accuracy(classifier, test_data)
    print("{0:6s} {1:8.5f}".format("Test", accuracy))
    print(len(selected_features))

    return classifier
    #evaluate(classifier, test_data, reviews, "output.txt")

def evaluate(classifier, features_category_tuples, data_set_name, reference_text):
    accuracy_results_file = open("dt-{}-output-dev.txt".format(data_set_name), 'w', encoding='utf-8')
    accuracy_results_file.write('Results of {}:\n\n'.format(data_set_name))
    # test on the data
    accuracy = nltk.classify.accuracy(classifier, features_category_tuples)
    print("The accuracy of {0} is ".format(data_set_name))
    print(accuracy)

    accuracy_results_file.write("{0:10s} {1:8.5f}\n\n".format("Accuracy", accuracy))

    features_only = []
    reference_labels = []
    for feature_vectors, category in features_category_tuples:
        features_only.append(feature_vectors)
        reference_labels.append(category)

    predicted_labels = classifier.classify_many(features_only)

    confusion_matrix = nltk.ConfusionMatrix(reference_labels, predicted_labels)
    print(confusion_matrix)

    accuracy_results_file.write(str(confusion_matrix))
    accuracy_results_file.write('\n\n')
    accuracy_results_file.close()

    predict_results_file = open("{}_output.txt".format(data_set_name), 'w', encoding='utf-8')
    for reference, predicted, text in zip(
                                          reference_labels,
                                          predicted_labels,
                                          reference_text
                                          ):
        if reference != predicted:
            predict_results_file.write("{0} {1}\n{2}\n\n".format(reference, predicted, text))
    predict_results_file.close()

def features_stub():
    #needs to change the datafile later.
    datafiles = {"restaurant-training.data", "restaurant-development.data", "restaurant-testing.data"}
    #datafile = "restaurant-training.data"
    train_data = data_helper.read_file('restaurant-training.data')
    develop_data = data_helper.read_file('restaurant-development.data')
    testing_data = data_helper.read_file('restaurant-testing.data')

    #create category_texts for each data
    positive_texts, negative_texts = data_helper.get_reviews(train_data)
    train_category_texts = {"positive": positive_texts, "negative": negative_texts}

    positive_texts, negative_texts = data_helper.get_reviews(develop_data)
    develop_category_texts = {"positive": positive_texts, "negative": negative_texts}

    positive_texts, negative_texts = data_helper.get_reviews(testing_data)
    testing_category_texts = {"positive": positive_texts, "negative": negative_texts}


    #look for the best classifier
    classifier = training_through_informative(train_category_texts, develop_category_texts,
        testing_category_texts, 'binning')
    create_pickle(classifier,'nb_word_bin.pickle')

    #look for the best classifier
    classifier = training_through_informative(train_category_texts, develop_category_texts,
        testing_category_texts, 'embedding')
    create_pickle(classifier,'nb_word_embedding.pickle')

    #look for the best classifier
    classifier = training_through_informative(train_category_texts, develop_category_texts,
        testing_category_texts, 'liwc')
    create_pickle(classifier,'nb_ALL_LIWC.pickle')



    #classifier for DecisionTreeClassifier for binning
    feature_category_tuples = word_embedding(train_category_texts, None)
    classifier = dt_training(feature_category_tuples, 'dt_word_bin.pickle')

    #comparing to testing
    testing_category_tuples = word_embedding(testing_category_texts, None)
    texts = get_features_category_tuples(testing_category_texts, 'bin')

    evaluate(classifier, testing_category_tuples, 'result.txt', texts)

    #comparing to development
    develop_category_tuples = word_embedding(develop_category_texts, None)
    texts = get_features_category_tuples(develop_category_texts, 'bin')

    evaluate(classifier, develop_category_tuples, 'result.txt', texts)


    #classifier for DecistionTreeClassifier for word_embedding
    feature_category_tuples = word_embedding(train_category_texts, None)
    classifier = dt_training(feature_category_tuples, 'dt_word_embedding.pickle')

    #comparing to testing
    testing_category_tuples = word_embedding(testing_category_texts, None)
    texts = get_features_category_tuples(testing_category_texts, 'embedding')

    evaluate(classifier, testing_category_tuples, 'result.txt', texts)

    #comparing to development
    develop_category_tuples = word_embedding(develop_category_texts, None)
    texts = get_features_category_tuples(develop_category_texts, 'embedding')

    evaluate(classifier, develop_category_tuples, 'result.txt', texts)

    #classifier for all_liwc
    feature_category_tuples = all_liwc(train_category_texts, None)
    classifier = dt_training(feature_category_tuples, 'dt_all_liwc.pickle')

    #comparing to testing
    testing_category_tuples = all_liwc(testing_category_texts, None)
    texts = get_features_category_tuples(testing_category_texts, 'embedding')

    evaluate(classifier, testing_category_tuples, 'result.txt', texts)

    #comparing to development
    develop_category_tuples = all_liwc(develop_category_texts, None)
    texts = get_features_category_tuples(develop_category_texts, 'embedding')

    evaluate(classifier, develop_category_tuples, 'result.txt', texts)
    """
    for datafile in datafiles:
        raw_data = data_helper.read_file(datafile)
        positive_texts, negative_texts = data_helper.get_reviews(raw_data)

        category_texts = {"positive": positive_texts, "negative": negative_texts}
        feature_set = FEATURE_SETS

    #for filename in feature_set:
    #print(filename)
        if datafile == "restaurant-training.data":
            feature_category_tuples = binning(category_texts)
            classifier_with_bin(feature_category_tuples)

            feature_category_tuples = word_embedding(category_texts)
            classifier_with_embedding(feature_category_tuples)

        for filename in feature_set:
            key = ""
            pickle_key = ""
            dt_key = ""
            if datafile == "restaurant-training.data":
                key = filename+"-training-features.txt"
                pickle_key = "restaurant-"+filename+"-model-P1.pickle"
                dt_key = "dt-"+filename+"-model-P2.pickle"

            if datafile == "restaurant-development.data":
                key = filename+"-development-features.txt"

            if datafile == "restaurant-testing.data":
                key = filename+"-testing-features.txt"

            #features_category_tuples, texts = get_features_category_tuples(category_texts, filename)
            #write_features_category(features_category_tuples, key)


            #if datafile == 'restaurant-training.data':
                #new_key = filename+"-training-informative-features.txt"
                #getting_most_informative(features_category_tuples, new_key)

    """

def unzip_the_file(input_file):
    contents = []
    zip_archive = zipfile.ZipFile(input_file)
    #print(zip_archive.namelist())
    for fn in zip_archive.namelist():
        if fn.endswith(".txt"):
            txt_file = zip_archive.open(fn, 'r').read().decode('latin1')
            if 'NEG' in fn:
                contents.append((txt_file, 'negative'))
            else:
                contents.append((txt_file, 'positive'))
    return contents

def turn_into_category(contents):
    positive_texts = []
    negative_texts = []

    for text, label in contents:
        if label == 'negative':
            negative_texts.append(text)
        else:
            positive_texts.append(text)

    return positive_texts, negative_texts

def combine_features(feature1, feature2, feature3):
    feature_category_tuples = []
    for (vector, label) in feature1:
        feature_category_tuples.append((vector, label))

    for (vector, label) in feature2:
        feature_category_tuples.append((vector, label))

    for (vector, label) in feature3:
        feature_category_tuples.append((vector, label))

    return feature_category_tuples

def features_stub_2():
    train_contents = unzip_the_file('stories-train.zip')
    develop_contents = unzip_the_file('stories-development.zip')
    test1_contents = unzip_the_file('stories-test1.zip')
    test2_contents = unzip_the_file('stories-test2.zip')

    negative_texts, positive_texts = turn_into_category(train_contents)
    train_category_texts = {"positive": positive_texts, "negative": negative_texts}

    negative_texts, positive_texts = turn_into_category(develop_contents)
    develop_category_texts = {"positive": positive_texts, "negative": negative_texts}

    negative_texts, positive_texts = turn_into_category(test1_contents)
    test1_category_texts = {"positive": positive_texts, "negative": negative_texts}

    negative_texts, positive_texts = turn_into_category(test2_contents)
    test2_category_texts = {"positive": positive_texts, "negative": negative_texts}
    """
#word_features-----------------------
    feature_category_tuples, a = get_features_category_tuples(train_category_texts, 'word_features')
    #print(feature_category_tuples)

    #nb
    #classifier = SklearnClassifier(BernoulliNB()).train(feature_category_tuples)
    #dt
    #classifier = SklearnClassifier(DecisionTreeClassifier()).train(feature_category_tuples)
    #svm
    classifier = SklearnClassifier(SVC()).train(feature_category_tuples)

    develop_category_tuples, a = get_features_category_tuples(develop_category_texts, 'word_features')
    texts = get_features_category_tuples(develop_category_texts, 'bin')

    evaluate(classifier, develop_category_tuples, 'result.txt', texts)

    test1_category_tuples, a = get_features_category_tuples(test1_category_texts, 'word_features')
    texts = get_features_category_tuples(test1_category_texts, 'bin')

    evaluate(classifier, test1_category_tuples, 'result.txt', texts)

    test2_category_tuples, a = get_features_category_tuples(test2_category_texts, 'word_features')
    texts = get_features_category_tuples(test2_category_texts, 'bin')

    evaluate(classifier, test2_category_tuples, 'result.txt', texts)

#word_pos_features------------------
    feature_category_tuples, a = get_features_category_tuples(train_category_texts, 'word_pos_features')

    #nb
    #classifier = SklearnClassifier(BernoulliNB()).train(feature_category_tuples)
    #dt
    #classifier = SklearnClassifier(DecisionTreeClassifier()).train(feature_category_tuples)
    #svm
    classifier = SklearnClassifier(SVC()).train(feature_category_tuples)

    develop_category_tuples, a = get_features_category_tuples(develop_category_texts, 'word_features')
    texts = get_features_category_tuples(develop_category_texts, 'bin')

    evaluate(classifier, develop_category_tuples, 'result.txt', texts)

    test1_category_tuples, a = get_features_category_tuples(test1_category_texts, 'word_features')
    texts = get_features_category_tuples(test1_category_texts, 'bin')

    evaluate(classifier, test1_category_tuples, 'result.txt', texts)

    test2_category_tuples, a = get_features_category_tuples(test2_category_texts, 'word_features')
    texts = get_features_category_tuples(test2_category_texts, 'bin')

    evaluate(classifier, test2_category_tuples, 'result.txt', texts)


#word_binning-----------------------
    feature_category_tuples = binning(train_category_texts, None)

    #nb
    #classifier = SklearnClassifier(BernoulliNB()).train(feature_category_tuples)
    #dt
    #classifier = SklearnClassifier(DecisionTreeClassifier()).train(feature_category_tuples)
    #svm
    classifier = SklearnClassifier(SVC()).train(feature_category_tuples)

    develop_category_tuples = binning(develop_category_texts, None)
    texts = get_features_category_tuples(develop_category_texts, 'bin')

    evaluate(classifier, develop_category_tuples, 'result.txt', texts)

    test1_category_tuples = binning(test1_category_texts, None)
    texts = get_features_category_tuples(test1_category_texts, 'bin')

    evaluate(classifier, test1_category_tuples, 'result.txt', texts)

    test2_category_tuples = binning(test2_category_texts, None)
    texts = get_features_category_tuples(test2_category_texts, 'bin')

    evaluate(classifier, test2_category_tuples, 'result.txt', texts)

#word_embedding---------------
    feature_category_tuples = word_embedding(train_category_texts, None)

    #nb
    #classifier = SklearnClassifier(BernoulliNB()).train(feature_category_tuples)
    #dt
    #classifier = SklearnClassifier(DecisionTreeClassifier()).train(feature_category_tuples)
    #svm
    classifier = SklearnClassifier(SVC()).train(feature_category_tuples)

    develop_category_tuples = word_embedding(develop_category_texts, None)
    texts = get_features_category_tuples(develop_category_texts, 'bin')

    evaluate(classifier, develop_category_tuples, 'result.txt', texts)

    test1_category_tuples = word_embedding(test1_category_texts, None)
    texts = get_features_category_tuples(test1_category_texts, 'bin')

    evaluate(classifier, test1_category_tuples, 'result.txt', texts)

    test2_category_tuples = word_embedding(test2_category_texts, None)
    texts = get_features_category_tuples(test2_category_texts, 'bin')

    evaluate(classifier, test2_category_tuples, 'result.txt', texts)

#word_ALL_LIWC----------------
    feature_category_tuples = all_liwc(train_category_texts, None)

    #nb
    #classifier = SklearnClassifier(BernoulliNB()).train(feature_category_tuples)
    #dt
    #classifier = SklearnClassifier(DecisionTreeClassifier()).train(feature_category_tuples)
    #svm
    classifier = SklearnClassifier(SVC()).train(feature_category_tuples)

    develop_category_tuples = all_liwc(develop_category_texts, None)
    texts = get_features_category_tuples(develop_category_texts, 'bin')

    evaluate(classifier, develop_category_tuples, 'result.txt', texts)

    test1_category_tuples = all_liwc(test1_category_texts, None)
    texts = get_features_category_tuples(test1_category_texts, 'bin')

    evaluate(classifier, test1_category_tuples, 'result.txt', texts)

    test2_category_tuples = all_liwc(test2_category_texts, None)
    texts = get_features_category_tuples(test2_category_texts, 'bin')

    evaluate(classifier, test2_category_tuples, 'result.txt', texts)
    """
#competition_pickle
    feature_category_tuples, a = get_features_category_tuples(train_category_texts, 'word_features')
    binning_category_tuples = binning(train_category_texts, None)
    all_liwc_category_tuples = all_liwc(train_category_texts, None)

    final_features = combine_features(feature_category_tuples, binning_category_tuples, [])

    classifier = SklearnClassifier(BernoulliNB()).train(final_features)
    create_pickle(classifier, 'blogs-competition-model-P2.pickle')



if __name__ == "__main__":
    #features_stub()
    features_stub_2()
