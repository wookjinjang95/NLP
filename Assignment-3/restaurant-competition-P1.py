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
    return feature_vector

def evaluate(classifier, data):
    #fh = open(output_file, 'w', encoding='utf-8')

    print(nltk.classify.accuracy(classifier, data))
    #print(nltk.ConfusionMatrix(test_set, test_set))

    features_only = [item[0] for item in data]
    #print(features_only)
    reference_labels = [item[1] for item in data]
    #print(reference_labels)
    predicted_labels = classifier.classify_many(features_only)
    #print(predicted_labels)

    confusion_matrix = nltk.ConfusionMatrix(reference_labels, predicted_labels)
    print(confusion_matrix)
    #print(reference_labels)


#c) The third should be the output file.


if __name__ == "__main__":
    args = list(sys.argv)
    #print(args)
    picklefile = args[1]
    datafile = args[2]
    #output_file = args[3]

    classifier = opening_the_pickle(picklefile)

    key = ""
    if picklefile == 'restaurant-word_features-model-P1.pickle':
        key = 'word_features'
    if picklefile == 'restaurant-word_pos_features-model-P1.pickle':
        key = 'word_pos_features'
    if picklefile == 'restaurant-word_pos_liwc_features-model-P1.pickle':
        key = 'word_pos_liwc_features'
    if picklefile == 'restaurant-competition-model-P1.pickle':
        key = 'word_liwc_features'
    data = opening_the_file(datafile, key)

    evaluate(classifier, data)
