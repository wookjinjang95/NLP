import re


def read_file(fname):
    with open(fname, "rb") as fin:
        raw_data = fin.read().decode("latin1")
    return raw_data


def get_score(review):
    """
    This function extracts the integer score from the review.

    Write a regular expression that searches for the Overall score
    and then extract the score number.

    :param review: All text associated with the review.
    :return: int: score --- the score of the review
    """
    ###     YOUR CODE GOES HERE
    #print(review)
    score = int(re.search(r'Overall = ([1-5])', review).group(1))
    #print(score)
    return score
    #aise NotImplemented

    #return score

def get_text(review):
    """
    This function extracts the description part of the
    restaurant review.

    Use regex to extract the Text field of the review,
    similar to the get_score() function.

    :param review:
    :return: str: text -- the textual description part of the restaurant review.
    """

    ###     YOUR CODE GOES HERE
    #print(review)
    text = re.search(r'Text = "(.*)"', review).group(1)
    #print(text)
    return text
    #raise NotImplemented

    #return text


def get_reviews(raw_data):
    """
    Process the restaurant review data. Split the data into two
    lists, one list for positive reviews and one list for negative
    reviews. The list items should be the descriptive text of
    each restaurant review.

    A positive review has a overall score of at least 3 and
    negative reviews have scores less than 3.

    :param raw_data:
    :return:
    """
    positive_texts = []
    negative_texts = []

    for review in re.split(r'\.\n', raw_data):
        overall_score = get_score(review)
        review_text = get_text(review)
        if overall_score > 3:
            positive_texts.append(review_text)
        elif overall_score < 3:
            negative_texts.append(review_text)

        ###     YOUR CODE GOES HERE
        #raise NotImplemented


    return positive_texts, negative_texts




def test_main():
    datafile = "restaurant-training.data"
    raw_data = read_file(datafile)
    p, n = get_reviews(raw_data)

    assert p[0].startswith("An excellent restaurant."), p[0]
    assert n[0].startswith("Place was nice did not care for the BBQ or the service."), n[0]



if __name__ == "__main__":
    test_main()
