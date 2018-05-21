
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from nltk.stem import PorterStemmer
from nltk.tree import Tree
import nltk

"""
    POS tagger
"""

def pos_tagger(text):
    words = nltk.word_tokenize(text)
    pos_tag = nltk.pos_tag(words)
    return pos_tag

def pos_to_word_list(pos_tag):
    words = []
    for item in pos_tag:
        words.append(item[0])
    return words

def convert_tree_to_tag(tree):
    for subtree in tree.subtrees():
        print(subtree)
    print('--------------')
    #print(tree)
"""
    Find the main_noun and main_verb in the pos_tag list.
    Return the main_noun and main_verb in the given pos_tag.
"""
def get_noun_verb(words):
    lowered = [w.lower() for w in words]
    pos_tag = nltk.pos_tag(lowered)
    #print(pos_tag)
    main_noun = []
    main_verb = []
    for item in pos_tag:
        if 'NN' in item[1]:
            main_noun.append(item[0])
        if 'VB' in item[1]:
            main_verb.append(item[0])

    return main_noun, main_verb

"""
    The normalize_the_sentence gets an input of a sentence
    and returns a list in pos_tag (x,y) where x is a word and
    y is the part of speech of that word.

    But before doing so, it also uses stem and returns it.
"""
def normalize_the_question(words):
    ps = PorterStemmer()
    #words = nltk.word_tokenize(sentence)
    main_noun, main_verb = get_noun_verb(words)

    stopwords = nltk.corpus.stopwords.words('english')

    #this is tuple
    noun_filtered = [w for w in main_noun if w not in stopwords]
    verb_filtered = [w for w in main_verb if w not in stopwords]

    final = []
    for item in verb_filtered:
        if 'ing' in item or 'ed' in item:
            word = ps.stem(item)
            final.append((word, 'v'))
        else:
            final.append((item, 'v'))

    for item in noun_filtered:
        final.append((item, 'n'))

    #print("items: ")
    #print(final)
    return final

def normalize_the_sentence(sentence):
    ps = PorterStemmer()
    words = nltk.word_tokenize(sentence)
    pos_tag = nltk.pos_tag(words)

    stopwords = nltk.corpus.stopwords.words('english')
    filtered = [(x,y) for (x,y) in pos_tag if x not in stopwords]

    final = []
    for (x,y) in filtered:
        if 'VB' in y:
            word = ps.stem(x)
        else:
            word = x
        final.append(word.lower())

    return final


"""
    Just get a sentence that is relative in the question and do for the
    matching by the sentence. The input for this is pos_tag of the question.
"""
def get_a_relative_question(question):
    words = nltk.word_tokenize(question)
    lowered = [w.lower() for w in words]
    #print(lowered)
    #pos_tag = pos_tagger(question)
    filtered = []
    five_w = ['who', 'when', 'where', 'why', 'how', 'what']

    for item in lowered:
        if item not in five_w:
            filtered.append(item)

    return filtered


"""
    This will compare the sentence with a question. We are going to
    find the main_verb in the question and also the main_noun in the question.
    Those two elements will be the key to find the sentence that answer the
    question.
"""
def compare_sentence(question, sentences):
    max_match = 0
    matched_sentence = "couldn't find the answer"

    rel_words = get_a_relative_question(question)
    rel_words = normalize_the_question(rel_words)

    print("question: "+question)
    #print("filtered: "+rel_string)
    final_sentence = []

    for sentence in sentences:
        count = 0
        rel_sentence = normalize_the_sentence(sentence)

        for (x,y) in rel_words:
            if x in rel_sentence:
                count += 1

        if count > max_match:
            final_sentence = rel_sentence
            matched_sentence = sentence
            max_match = count

    #print(final_sentence)
    #print("answer: "+matched_sentence)
    return matched_sentence


def justify(answer,question):

    ps = PorterStemmer()
    string = []
    word_answer = nltk.word_tokenize(answer)
    word_question = nltk.word_tokenize(question)
    clue = word_question[0].lower()
    word_answer = nltk.pos_tag(word_answer)

    words = normalize_the_question(word_question)

    #define the main_verb
    main_verb = [w for (w,a) in words if a == 'v']
    #print(main_verb)
    if clue == 'what':
        for index, item in enumerate(word_answer):
            if 'VB' in item[1] and ps.stem(item[0]) in main_verb:
                if 'VB' in word_answer[index-1][1]:
                    for item in word_answer[:index-1]:
                        #string = string+item[0]+" "
                        string.append(item[0])

                else:
                    for item in word_answer[index+1:]:
                        #string = string+item[0]+" "
                        string.append(item[0])

    if clue == 'why':
        found = False
        for item in nltk.word_tokenize(answer):
            if item == 'because':
                found = True
            if 'because' not in item and found == True:
                #string = string+item+" "
                string.append(item)
    #print("string: "+string)

    if clue == 'who':
        for index, item in enumerate(word_answer):
            if 'VB' in item[1] and ps.stem(item[0]) in main_verb:
                for item in word_answer[:index]:
                    if 'VB' not in item[1]:
                        #string = string+item[0]+" "
                        string.append(item[0])

    if len(string) == 0:
        return answer

    return " ".join(string)


def get_answer(question, story):
    answer = "don't know the answer"
    if 'Story' in question['type']:
        #convert_tree_to_tag(question['par'])
        answer = compare_sentence(question['text'], nltk.sent_tokenize(story['text']))
        print("before: "+answer)
        answer = justify(answer,question['text'])
        print("after: "+answer)
    else:
        answer = compare_sentence(question['text'], nltk.sent_tokenize(story['sch']))
        print("before: "+answer)
        answer = justify(answer,question['text'])
        print("after: "+answer)



    """
    :param question: dict
    :param story: dict
    :return: str


    question is a dictionary with keys:
        dep -- A list of dependency graphs for the question sentence.
        par -- A list of constituency parses for the question sentence.
        text -- The raw text of story.
        sid --  The story id.
        difficulty -- easy, medium, or hard
        type -- whether you need to use the 'sch' or 'story' versions
                of the .


    story is a dictionary with keys:
        story_dep -- list of dependency graphs for each sentence of
                    the story version.
        sch_dep -- list of dependency graphs for each sentence of
                    the sch version.
        sch_par -- list of constituency parses for each sentence of
                    the sch version.
        story_par -- list of constituency parses for each sentence of
                    the story version.
        sch --  the raw text for the sch version.
        text -- the raw text for the story version.
        sid --  the story id


    """
    #if(question['type'] == "sch"):
    #    text = story['sch']
    ###     Your Code Goes Here         ###

    #answer = "whatever you think the answer is"

    ###     End of Your Code         ###

    return answer



#############################################################
###     Dont change the code below here
#############################################################

class QAEngine(QABase):
    @staticmethod
    def answer_question(question, story):
        answer = get_answer(question, story)
        return answer


def run_qa():
    QA = QAEngine()
    QA.run()
    QA.save_answers()

def main():
    run_qa()
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()

if __name__ == "__main__":
    main()
