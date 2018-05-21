from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from nltk.stem import PorterStemmer
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
            final.append((word,4))
        else:
            final.append((item,4))

    for item in noun_filtered:
        final.append((item,1))

    #print("items: ")
    #print(final)
    return final

"""
    Just get a sentence that is relative in the question and do for the
    matching by the sentence. The input for this is pos_tag of the question.
"""
def get_a_relative_sentence(question):
    words = nltk.word_tokenize(question)
    lowered = [w.lower() for w in words]
    #print(lowered)
    #pos_tag = pos_tagger(question)
    filtered = []
    five_w = ['who', 'what', 'when', 'where', 'why', 'how']

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

    rel_words = get_a_relative_sentence(question)
    rel_words = normalize_the_question(rel_words)

    #this is for printing:
    #rel_string = ""
    #for item in rel_words:
    #    rel_string = rel_string+"("+item+") "

    print("question: "+question)
    #print("filtered: "+rel_string)
    final_sentence = []

    for sentence in sentences:
        count = 0
        rel_sentence = nltk.word_tokenize(sentence)
        #rel_sentence = normalize_the_sentence(tok_sentence)

        for (x,y) in rel_words:
            if x in rel_sentence:
                count += y

        if count > max_match:
            final_sentence = rel_sentence
            matched_sentence = sentence
            max_match = count

    #rel_string = ""
    #for item in final_sentence:
    #    rel_string = rel_string+"("+item+") "
    #print("sent_fil: "+rel_string)
    print("answer: "+matched_sentence)
    return matched_sentence
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################

# takes in multiple sentences in string format
def tokenize_sentences(paragraph):
    return [sentence for sentence in nltk.sent_tokenize(paragraph)]

# takes in a single sentence in string format
def tokenize_words(sentence):
    return [word for word in nltk.word_tokenize(sentence)]

# param words is a list of words i.e ['foo', 'bar']
def remove_question_words(words):
    lowered = [w.lower() for w in words]
    filtered = []
    question_words = ['?', "\'"]

    for word in lowered:
        if word not in question_words:
            filtered.append(word)
    return filtered

# param words is a list of words i.e ['foo', 'bar']
def remove_stop_words(words):
    stopwords = nltk.corpus.stopwords.words('english')

    filtered = [w.lower() for w in words if w not in stopwords]
    # print('before:', words)
    # print('after:', filtered)
    return filtered

def normalize_question(question, should_normalize=False):
    # if should_normalize:
    # return remove_stop_words(question)
    # else:
        # return question
    # return remove_question_words(question)
    return question

def find_key_words(question):
    key_words = []
    for word, pos in question:
        # print(word, pos)
        if ("NN" in pos) or ("VB" in pos):
            key_words.append(word)

    return key_words

# def normalize_story(story):

def get_answer(question, story):
    print('==========================================')
    tokenized_question = tokenize_words(question['text'])
    normalized_question = normalize_question(tokenized_question)
    tagged_question = nltk.pos_tag(normalized_question)
    key_words = find_key_words(tagged_question)

    story_to_use = ''
    if 'Story' in question['type']:
        story_to_use = story['text']
    else:
        story_to_use = story['sch']
    tokenzed_story = tokenize_sentences(story_to_use)

    result = []
    for sent in tokenzed_story:
        # print(nltk.pos_tag(remove_stop_words(tokenize_words(sent))))
        for word in tokenize_words(sent):
            if word in key_words and word not in result:
                print('found in key_word! ', word)
                result.append(sent)


    if(result):
        answer = result[0]
    else:
        answer = 'fd'


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
