import nltk
import re
import word_category_counter
import data_helper
import pickle
import sys
import features

#a) The first should be the trained classifer model (one of the pickled models)
#b) The second should be the file with the reviews in it to be classified
def opening_the_pickle(picklefile):
    #f = open(pickle, 'r')
    #print(pickle)
    f = open(picklefile, 'rb')
    classifier = pickle.load(f)
    f.close()

    return classifier

def opening_the_file(datafile, key):
    raw_data = data_helper.read_file(datafile)
    positive_texts, negative_texts = data_helper.get_reviews(raw_data)

    category_texts = {"positive": positive_texts, "negative": negative_texts}
    feature_vector, texts = features.get_features_category_tuples(category_texts, key)
    #print(texts)
    return feature_vector, texts

def opening_for_binning(datafile, key):
    raw_data = data_helper.read_file(datafile)
    positive_texts, negative_texts = data_helper.get_reviews(raw_data)
    category_texts = {"positive": positive_texts, "negative": negative_texts}

    texts = features.get_features_category_tuples(category_texts, key)
    feature_category_tuples = features.binning(category_texts)

    return feature_category_tuples, texts

def opening_for_word_embedding(datafile, key):
    raw_data = data_helper.read_file(datafile)
    positive_texts, negative_texts = data_helper.get_reviews(raw_data)
    category_texts = {"positive": positive_texts, "negative": negative_texts}

    texts = features.get_features_category_tuples(category_texts, key)
    feature_category_tuples = features.word_embedding(category_texts)

    return feature_category_tuples, texts

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

def evaluate_binning(classifier, data, reviews, output_file):
    fh = open(output_file, 'w', encoding='utf-8')

    # test on the data
    accuracy = nltk.classify.accuracy(classifier, data)
    print(accuracy)
    fh.write("{0:10s} {1:8.5f}\n\n".format("Accuracy", accuracy))

    features_only = [example[0] for example in data]

    reference_labels = [example[1] for example in data]
    predicted_labels = classifier.classify_many(features_only)
    reference_text = [review[0] for review in reviews]
    confusion_matrix = nltk.ConfusionMatrix(reference_labels, predicted_labels)
    print(confusion_matrix)

    fh.write(str(confusion_matrix))
    fh.write('\n\n')

    for reference, predicted, text in zip(
                                          reference_labels,
                                          predicted_labels,
                                          reference_text
                                          ):
        if reference != predicted:
            fh.write("{0} {1}\n{2}\n\n".format(reference, predicted, text))

    fh.close()


if __name__ == "__main__":
    args = list(sys.argv)
    #print(args)
    picklefile = args[1]
    datafile = args[2]
    #output_file = args[3]

    classifier = opening_the_pickle(picklefile)

    key = ""
    #if picklefile == 'restaurant-word_features-model-P1.pickle':
    #    key = 'word_features'
    #if picklefile == 'restaurant-word_pos_features-model-P1.pickle':
    #    key = 'word_pos_features'
    #if picklefile == 'restaurant-word_pos_liwc_features-model-P1.pickle':
    #    key = 'word_pos_liwc_features'
    #if picklefile == 'restaurant-competition-model-P1.pickle':
    #    key = 'word_liwc_features'

    if picklefile == 'dt-word_features-model-P2.pickle':
        key = 'word_features'
    if picklefile == 'dt-word_pos_features-model-P2.pickle':
        key = 'word_pos_features'
    if picklefile == 'dt-word_pos_liwc_features-model-P2.pickle':
        key = 'word_pos_liwc_features'
    if picklefile == 'dt-competition-model-P2.pickle':
        key = 'word_liwc_features'

    if len(key) > 0:
        data, texts = opening_the_file(datafile, key)
        evaluate(classifier, data, key, texts)
    else:
        data, texts = opening_for_binning(datafile, 'bin')
        if 'nb_bin_word' in picklefile:
            key = 'result_of_bin_nb'
        if 'dt_bin_word' in picklefile:
            key = 'result_of_bin_dt'
        evaluate(classifier, data, key, texts)
