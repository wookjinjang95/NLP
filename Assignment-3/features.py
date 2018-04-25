import nltk
import re
import word_category_counter
import data_helper
import pickle
import sys



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


FEATURE_SETS = {"word_pos_features", "word_features", "word_pos_liwc_features", "word_liwc_features"}

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
        for text in category_text_dict[category]:

            words, tags = get_words_tags(text)
            feature_vectors = {}

            ###     YOUR CODE GOES HERE
            words = normalize(words, True)
            if feature_set == "word_features":
                feature_vectors.update(get_ngram_features(words))

            if feature_set == "word_pos_features":
                feature_vectors.update(get_ngram_features(words))
                feature_vectors.update(get_pos_features(tags))

            if feature_set == "word_pos_liwc_features":
                feature_vectors.update(get_liwc_features(words))
                feature_vectors.update(get_ngram_features(words))
                feature_vectors.update(get_pos_features(tags))

            if feature_set == "word_liwc_features":
                feature_vectors.update(get_liwc_features(words))
                feature_vectors.update(get_unigram_features(words))

            #raise NotImplemented

            features_category_tuples.append((feature_vectors, category))
            texts.append(text)

    return features_category_tuples, texts


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
    """
    Save the feature values to file.

    :param features_category_tuples:
    :param outfile_name:
    :return: nothing, just creates a file and output.
    """

    training_data = []

    for(features, category) in features_category_tuples:
        tup = (features, category)
        training_data.append(tup)

    classifier = nltk.classify.NaiveBayesClassifier.train(training_data)
    f = open(outfile_name, 'wb')
    pickle.dump(classifier, f)
    f.close()


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


def features_stub():
    #needs to change the datafile later.
    datafiles = {"restaurant-training.data", "restaurant-development.data", "restaurant-testing.data"}
    #datafile = "restaurant-training.data"
    for datafile in datafiles:
        raw_data = data_helper.read_file(datafile)
        positive_texts, negative_texts = data_helper.get_reviews(raw_data)

        category_texts = {"positive": positive_texts, "negative": negative_texts}
        feature_set = FEATURE_SETS

    #for filename in feature_set:
    #print(filename)
        for filename in feature_set:
            key = ""
            pickle_key = ""
            if datafile == "restaurant-training.data":
                key = filename+"-training-features.txt"
                pickle_key = "restaurant-"+filename+"-model-P1.pickle"

            if datafile == "restaurant-development.data":
                key = filename+"-development-features.txt"

            if datafile == "restaurant-testing.data":
                key = filename+"-testing-features.txt"


            features_category_tuples, texts = get_features_category_tuples(category_texts, filename)
            write_features_category(features_category_tuples, key)

            if datafile == "restaurant-training.data":
                training_the_data(features_category_tuples, pickle_key)

            if datafile == 'restaurant-training.data':
                new_key = filename+"-training-informative-features.txt"
                getting_most_informative(features_category_tuples, new_key)





if __name__ == "__main__":
    features_stub()
