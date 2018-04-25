import argparse, re, nltk

# https://docs.python.org/3/howto/regex.html
# https://docs.python.org/3/library/re.html
# https://www.debuggex.com/

def get_words(pos_sent):
    #print(pos_sent)
    #tokenize = nltk.word_tokenize(pos_sent)
    #print(tokenize)
    #a) Write a regular expression that matches only the words of each word/pos-tag in the sentence.
    #print(tokenize)
    pattern = r'([A-Z]*[a-z,.]+)\/'
    #new_pattern = r'\w{2,'
    result = re.findall(pattern, pos_sent)
    #rresult = re.findall(new_pattern, pos_sent)


    #print(result)
    #print(rresult)

    #b) Use regular language built in part a to find list of words. This function should return without the tagger
    #   with white spaces.
    text = ""
    for word in result:
        text = text+word+" "
    print(text)


    # Your code goes here
    pass

def get_noun_phrase(pos_sent):
    # Penn Tagset
    # Adjetive can be JJ,JJR,JJS
    # Noun can be NN,NNS,NNP,NNPS
    prev_pattern = r'[A-Z]*[a-z]+'
    pattern = r'([A-Za-z]*[a-z]+\/DT\s)*([A-Za-z]+(?=\/[NNS]))'
    result = re.findall(pattern, pos_sent)
    output = []
    print(result)
    for (x,y) in result:
        a = re.findall(prev_pattern, x)
        b = re.findall(prev_pattern, y)
        #print(a)
        #print(b)
        str1 = ''.join(a)
        str2 = ''.join(b)
        if not a:
            text = str1+str2
        else:
            text = str1+" "+str2
        output.append(text)
    print(output)
    #return output

    # Your code goes here
    pass

def most_freq_noun_phrase(pos_sent_fname):
    # Your code goes here
    #print(pos_sent_fname)
    #sentences = nltk.sent_tokenize(pos_sent_fname)
    #print(sentences)
    filename = open(pos_sent_fname)
    text = filename.read()
    sentences = nltk.sent_tokenize(text)

    #print(sentences)
    prev_pattern = r'[A-Z]*[a-z]+'
    pattern = r'(([A-Za-z]+\/(DT|JJ|JJR))+([\s[A-Za-z]+\/NNP)+)'

    output = []
    for sent in sentences:
        result = re.findall(pattern, sent)
        if result:
            output.append(result[0][0])

    #print(output)
    final = []
    for item in output:
        r = re.findall(prev_pattern, item)
        str1 = " ".join(str(x).lower() for x in r)
        if str1:
            final.append(str1)

    print(final)
    fd = nltk.FreqDist(final)
    print("The most frqe NP in "+str(pos_sent_fname)+" : "+str(fd.most_common(3)))
    pass

if __name__ == '__main__':

    # python hw2-part1-stub.py -f fables-pos.txt
    # python hw2-part1-stub.py -f blogs-pos.txt
    parser = argparse.ArgumentParser(description='Assignment 2')
    parser.add_argument('-f', dest="pos_sent_fname", default="fables-pos.txt",  help='File name that contant the POS.')

    args = parser.parse_args()
    pos_sent_fname = args.pos_sent_fname

    pos_sent = 'All/DT animals/NNS are/VBP equal/JJ ,/, but/CC some/DT animals/NNS are/VBP more/RBR equal/JJ than/IN others/NNS ./.'
    print(pos_sent)
    print(str(get_words(pos_sent)))
    print(str(get_noun_phrase(pos_sent)))

    most_freq_noun_phrase(pos_sent_fname)
