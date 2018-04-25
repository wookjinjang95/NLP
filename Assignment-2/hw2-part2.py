from nltk.corpus import wordnet as wn
import argparse, re, nltk

#The word I am picking is "apple"

def print_syn_lemmas(word):
    with open('wordnet.txt', 'w') as the_file:
        #1) Pick a word and show all the sysnets of that word and their definitions.
        r = wn.synsets(word)
        for synset in r:
            result = str(synset)+": "+str(synset.definition())+"\n"
            the_file.write(result)


        #2) Pick one synset of that word and show all of its hyponyms and root hypernyms.
        r1 = r[0]
        result1 = "hypernyms: "+str(r1.hypernyms())+"\n"
        result2 = "hyponyms: "+str(r1.hyponyms())+"\n"

        the_file.write(result1)
        the_file.write(result2)

        #3) Show the hypernyn path from that word to the top of the hierarchy.
        paths = r1.hypernym_paths()
        #print(len(paths))
        #print(paths[0])

        for elem in paths:
            result = str([synset.name() for synset in elem])+"\n"
            the_file.write(result)
            #print([synset.name() for synset in elem])



if __name__ == '__main__':
    print_syn_lemmas('apple')
    #print_def_exp(wn.synset("dog.n.01"))
    #print_lexical_rel(wn.synset("dog.n.01"))
    #print_other_lexical_rel()
