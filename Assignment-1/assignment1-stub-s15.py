#!/usr/bin/env python

import nltk, zipfile, argparse, sys

###############################################################################
## Utility Functions ##########################################################
###############################################################################
# This method takes the path to a zip archive.
# It first creates a ZipFile object.
# Using a list comprehension it creates a list where each element contains
# the raw text of the fable file.
# We iterate over each named file in the archive:
#     for fn in zip_archive.namelist()
# For each file that ends with '.txt' we open the file in read only
# mode:
#     zip_archive.open(fn, 'rU')
# Finally, we read the raw contents of the file:
#     zip_archive.open(fn, 'rU').read()
def unzip_corpus(input_file):
    zip_archive = zipfile.ZipFile(input_file)
    contents = [zip_archive.open(fn, 'r').read().decode('utf-8') for fn in zip_archive.namelist() if fn.endswith(".txt")]
    return contents

###############################################################################
## Stub Functions #############################################################
###############################################################################
def process_corpus(corpus_name):
    input_file = corpus_name + ".zip"
    corpus_contents = unzip_corpus(input_file)

    # Your code goes here

    """
        ##### Part 1 of the assignment: Tokenization #####
    """
    delimit = []
    #delimit_lowercased = []
    total = 0
    #Defining the name
    print("Corpus name:", corpus_name)
    #print(corpus_contents)

    #print(len(corpus_contents))
    #sents = nltk.word_tokenize(corpus_contents)
    #print(len(corpus_contents))
    sentences = []
    final = []

    for each_file in corpus_contents:
        #print(each_file+"\n")
        #Puts it into a list of sentences.
        """
            This removes the punctuation in the list.
        """
        #sents = nltk.sent_tokenize(each_file)
        sents = nltk.sent_tokenize(each_file)
        for elem in sents:
            res = nltk.word_tokenize(elem)
            res = nltk.pos_tag(res)
            for item in res:
                final.append(item)

        words = nltk.word_tokenize(each_file)
        total += len(sents)
        for word in words:
            delimit.append(word)
            #delimit_lowercased.append(word.lower())
        #print(sents)

    print("Total words in the corpus:", total)

    #print(delimit)

    """
        ##### Part 2 of the assignment: Part-of-Speech #####
    """

    postag = nltk.pos_tag(delimit)
    fomat = ""
    # The first thing I need to do is to combine it into the form in a single text
    for elem in final:
        text = elem[0]+"/"+elem[1]
        fomat = fomat+text+" "

    # Once I do that, I need to learn how to write onto the file CORPUSE-NAME-pos.txt
    filename = corpus_name+"-pos.txt"
    with open(filename, 'w') as out:
        out.write(fomat)

    """
        #### Part 3 of the assignment: Frequency ####
    """

    #a) Write the vocabulary size of corpus
    words = [w.lower() for w in delimit]
    vocab = sorted(set(words))

    #print(vocab)

    print("Vocabulary size of corpus:", len(vocab))

    #b) Write the most frequent part-of-speech tag and its frequency to the stdout.
    words = nltk.pos_tag(delimit)
    tag_fd = nltk.FreqDist(tag for (word, tag) in words)
    frequent = tag_fd.most_common()

    print("The most frequent part-of-speech tag is "+frequent[0][0]+" with frequency " + str(frequent[0][1]))

    #c) Find the frequency of each unique word (after lowercasing) using the FreqDist module and write
    #   the list in decreasing order to a file named CORPUS-NAME-word-freq.txt
    words = [w.lower() for w in delimit]
    fdist = nltk.FreqDist(words)
    list_for_frequency = []
    for elem in fdist:
        item = (elem, fdist[elem])
        list_for_frequency.append(item)

    list_for_frequency = sorted(list_for_frequency, key=lambda x: x[1], reverse=True)
    #converted into single text
    text = ""
    for elem in list_for_frequency:
        text += str(elem[0])+" "+str(elem[1])+"\n"

    #write it onto the file
    filename = corpus_name+"-word-freq.txt"
    with open(filename, 'w') as out:
        out.write(text)

    #d) Find the frequency of each word given its part-of-speech tag. Use a conditional frequency distribution
    # for this (CondDistFreq) where the first item in the pair is the part-of-speech and the second item is the
    # lowered word. Note, the part-of-speech tagger requires uppercase words and returns the word/tag pair in the
    # inverse order of what we are asking here. Use the tabulate() method for the CondFreqDist class to write the
    # results to a file named CORPUS-NAME-pos-word-freq.txt.
    #use postag
    #words = nltk.pos_tag(words)
    words = nltk.pos_tag(delimit)
    #print(words)
    #final = []
    #conditions = []
    #for elem in words:
    #    item = (elem[1], elem[0].lower())
    #    conditions.append(elem[0].lower())
    #    final.append(item)

    cfs = nltk.ConditionalFreqDist((elem[0].lower(), elem[1]) for elem in words)

    filename = corpus_name+"-pos-word-freq.txt"

    stdout = sys.stdout
    sys.stdout = open(filename, 'w')
    cfs.tabulate()

    sys.stdout.close()
    sys.stdout = stdout


    """
        ##### Part 4: Similar Wards ######
    """
    #a) For the most frequent word in the NN (nouns), VBD (past-tense verbs), JJ (adjectives), and RB(adverbs)
    # pat-of-speech tags, find the most similar words using Text.similar(). Write the output to stdout (this will
    # happen by default)
    result = [0, 0, 0, 0]
    most_NN = ""
    most_VBD = ""
    most_JJ = ""
    most_RB = ""
    pos = ['NN', 'VBD', 'JJ', 'RB']

    #cfs = nltk.ConditionalFreqDist((p,w.lower()) for p in pos for (w,_) in words)

    stdout = sys.stdout
    sys.stdout = open('temp.txt', 'w')
    cfs.tabulate(samples=pos)

    sys.stdout.close()
    sys.stdout = stdout

    firstTime = True
    for line in open('temp.txt'):
        if(firstTime):
            firstTime = False
        else:
            elem = line.split()
            #print(elem)
            if int(elem[1]) > result[0]:
                result[0] = int(elem[1])
                most_NN = str(elem[0])
            if int(elem[2]) > result[1]:
                result[1] = int(elem[2])
                most_VBD = str(elem[0])
            if int(elem[3]) > result[2]:
                result[2] = int(elem[3])
                most_JJ = str(elem[0])
            if int(elem[4]) > result[3]:
                result[3] = int(elem[4])
                most_RB = str(elem[0])

    #print(result)
    #print(type(most_NN))
    #print(most_NN, most_VBD, most_JJ, most_RB)

    text = nltk.Text(word.lower() for word in delimit)

    #print("\n")
    for item in pos:
        if item == 'NN':
            print("The most frequent word in the "+item+" is "+most_NN+" and its similar words are: ")
            text.similar(most_NN)
        if item == 'VBD':
            print("The most frequent word in the "+item+" is "+most_VBD+" and its similar words are: ")
            text.similar(most_VBD)
        if item == 'JJ':
            print("The most frequent word in the "+item+" is "+most_JJ+" and its similar words are: ")
            text.similar(most_JJ)
        if item == 'RB':
            print("The most frequent word in the "+item+" is "+most_RB+" and its similar words are: ")
            text.similar(most_RB)

    print("Collocations: ")
    text.collocations()

    pass

###############################################################################
## Program Entry Point ########################################################
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment 1')
    parser.add_argument('--corpus', required=True, dest="corpus", metavar='NAME',  help='Which corpus to process {fables, blogs}')

    args = parser.parse_args()

    corpus_name = args.corpus

    if corpus_name == "fables" or "blogs":
        process_corpus(corpus_name)
    else:
        print("Unknown corpus name: {0}".format(corpus_name))
