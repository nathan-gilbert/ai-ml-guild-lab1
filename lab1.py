import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize


def word_feats(words):
    return dict([(word, True) for word in words])


if __name__ == "__main__":
    negative_ids = movie_reviews.fileids('neg')
    positive_ids = movie_reviews.fileids('pos')

    negative_features = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negative_ids]
    positive_features = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in positive_ids]

    neg_cutoff = round(len(negative_features) * 0.80)
    pos_cutoff = round(len(positive_features) * 0.80)
    training_features = negative_features[:neg_cutoff] + positive_features[:pos_cutoff]
    testing_features = negative_features[neg_cutoff:] + positive_features[pos_cutoff:]
    print('train on %d instances, test on %d instances' % (len(training_features), len(testing_features)))

    classifier = NaiveBayesClassifier.train(training_features)
    print('accuracy:', nltk.classify.util.accuracy(classifier, testing_features))

    classifier.show_most_informative_features()

    test_reviews = [
        """Wow! That's about all one can say about this movie. The first time that I saw
        it I was mesmerized. The movie looked so cool and hey, it actually had a good
        plot. If you haven't seen this movie yet, get out from your cave and see it
        right away. I have seen this movie umpteen times and it still shocks and
        surprises me. """,
        """Anyway, back to the movie. It is as bad as you've no doubt heard. The scene
        changes from night to day to night, the spaceship is a hubcap (you can see the
        string it hangs from catch on fire at one point), I could do a better job
        acting, etc."""]

    for review in test_reviews:
        review_features = word_feats(word_tokenize(review.lower()))
        label = classifier.classify(review_features)
        prob_results = classifier.prob_classify(review_features)
        prob_str = " ({0:.2}/{1:.2})".format(prob_results.prob("pos"), prob_results.prob("neg"))
        print(review[:25], ": ", label, prob_str)