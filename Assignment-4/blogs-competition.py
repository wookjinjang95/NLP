import features, sys, pickle

def opening_the_pickle(picklefile):
    f = open(picklefile, 'rb')
    classifier = pickle.load(f)
    f.close()

    return classifier

if __name__ == "__main__":
    args = list(sys.argv)
    picklefile = args[1]
    datafile = args[2]
    output_file = args[3]

    classifier = opening_the_pickle(picklefile)
    contents = features.unzip_the_file(datafile)
    negative_texts, positive_texts = features.turn_into_category(contents)
    category_texts = {"positive": positive_texts, "negative": negative_texts}

    feature_category_tuples, a = features.get_features_category_tuples(category_texts, 'word_features')
    binning_category_tuples = features.binning(category_texts, None)
    all_liwc_category_tuples = features.all_liwc(category_texts, None)

    final_features = features.combine_features(feature_category_tuples, binning_category_tuples, [])

    features.evaluate(classifier, final_features, output_file, a)
