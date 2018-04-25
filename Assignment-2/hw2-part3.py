import re, nltk

def get_score(review):
    return int(re.search(r'Overall = ([1-5])', review).group(1))

def get_text(review):
    return re.search(r'Text = "(.*)"', review).group(1)

def process_reviews(file_name):
    file = open(file_name, "rb")
    raw_data = file.read().decode("latin1")
    file.close()

    positive_texts = []
    negative_texts = []
    first_sent = None
    for review in re.split(r'\.\n', raw_data):
        overall_score = get_score(review)
        review_text = get_text(review)
        if overall_score > 3:
            positive_texts.append(review_text)
        elif overall_score < 3:
            negative_texts.append(review_text)
        if first_sent == None:
            sent = nltk.sent_tokenize(review_text)
            if (len(sent) > 0):
                first_sent = sent[0]

    # There are 150 positive reviews and 150 negative reviews.
    #print((positive_texts))
    #print(len(negative_texts))
    # Your code goes here

    ## Testing: ##
    final = []
    n_final = []
    category = []
    n_category = []

    for text in positive_texts:
        words = nltk.word_tokenize(text)
        lowered = [word.lower() for word in words]
        category.append(lowered)
        for word in lowered:
            final.append(word)

    ####
    #print(len(final))
    #uni = list(nltk.ngrams(final, 1))
    #dd = nltk.FreqDist(uni)
    #for sample in sorted(dd, key=dd.get, reverse=True):
    #    print(sample, dd[sample])
    #bigrams = list(nltk.ngrams(final, 3))
    #dd = nltk.ConditionalFreqDist(bigrams)
    #for sample in sorted(dd, key=dd.get, reverse=True):
    #    print(sample, dd[sample])
    #    print(str((sample, dd[sample])))
    #trigrams = list(nltk.ngrams(final, 3))


    ####
    stopwords = nltk.corpus.stopwords.words('english')

    for text in negative_texts:
        words = nltk.word_tokenize(text)
        lowered = [word.lower() for word in words]
        n_category.append(lowered)
        for word in lowered:
            n_final.append(word)

    content = [w for w in final if w not in stopwords]

    n_content = [w for w in n_final if w not in stopwords]

    r = re.compile('\w')
    final = []
    n_final = []
    for item in content:
        if r.match(item):
            final.append(item)
    for item in n_content:
        if r.match(item):
            n_final.append(item)

    #total = 0
    #for w in final:
    #    if w == 'mashed':
    #        total += 1
    #print(total, len(final))
    #print(final)
    bigrams = list(nltk.ngrams(final,1))
    n_bigrams = list(nltk.ngrams(n_final,1))
    #print(bigrams)
    cfd = nltk.FreqDist(bigrams)
    #print("here")
    #print(str(cfd['(excellent, )']))
    fh = open("positive-unigram-freq.txt", "w")
    for sample in sorted(cfd, key=cfd.get, reverse=True):
        #print(sample, cfd[sample])
        #print(sample, cfd[sample])
        result = str((sample, cfd[sample]))+"\n"
        fh.write(result)
    fh.close()

    cfd = nltk.FreqDist(n_bigrams)
    fh = open("negative-unigram-freq.txt", "w")
    for sample in sorted(cfd, key=cfd.get, reverse=True):
        #print(sample, cfd[sample])
        result = str((sample, cfd[sample]))+"\n"
        fh.write(result)
    fh.close()

    bigrams = list(nltk.bigrams(final))
    n_bigrams = list(nltk.bigrams(n_final))
    bcfd = nltk.ConditionalFreqDist(bigrams)

    fh = open("positive-bigram-freq.txt", "w")
    for sample in sorted(bcfd, key=bcfd.get, reverse=True):
        #print(sample, bcfd[sample])
        result = str((sample, bcfd[sample]))+"\n"
        fh.write(result)
    fh.close()

    bcfd = nltk.ConditionalFreqDist(n_bigrams)

    fh = open("negative-bigram-freq.txt", "w")
    for sample in sorted(bcfd, key=bcfd.get, reverse=True):
        result = str((sample, bcfd[sample]))+"\n"
        fh.write(result)
    fh.close()

    #print(category)
    #print(category[0])

    ps = nltk.Text(final)
    ns = nltk.Text(n_final)

    print("content: positive")
    ps.collocations()
    print("content: negative")
    ns.collocations()



    #print(cfd.most_commom())


    #2) Create a frequency distribution of the unigrams for each category. Write these frequency
    #   in descending order to a file named CATEGORY-unigram-freq.txt where CATEGORY is either
    #   positive or negative. A positive review is one that has an Overall score > 3 and a negative
    #   review is one that has an Overall score < 3. Each line of the file should contain a word and
    #   its frequency separated by whitespace.

    #3) Create a conditional frequency distribution from the bigrams for each category. Write these
    #   frequencies in descending order to a file named CATEGORY-bigram-freq.txt. Each line of the file
    #   should contain the condition word, the word, and the frequency, all separated by whitespace.

    #4) Find the collocations for each category using the collocations() function. Write the ouput to stdout


    pass

# Write to File, this function is just for reference, because the encoding matters.
def write_file(file_name, data):
    file = open(file_name, 'w', encoding="utf-8")    # or you can say encoding="latin1"
    file.write(data)
    file.close()

if __name__ == '__main__':
    #filename = sys.argv[1]
    file_name = "restaurant-training.data"
    process_reviews(file_name)
