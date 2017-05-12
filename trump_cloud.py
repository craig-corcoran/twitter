import sys
import twitter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib


def trump_cloud(count=200, img_filename='wordcloud.png'):
    '''
    Make a wordcloud from trump's most recent tweets; creates a png
    :param count: number of tweets to use (max 200?)
    :param img_filename: name of image file for word cloud
    :return: None
    '''

    # HACK: this makes the png look like shit, but runs in the virtualenv (without conda)
    matplotlib.use('TkAgg')  # must be done before importing pyplot
    import matplotlib.pyplot as plt

    key_frame = pd.read_csv('keys.csv')

    api = twitter.Api(
        consumer_key=key_frame['consumer_key'][0],
        consumer_secret=key_frame['consumer_secret'][0],
        access_token_key=key_frame['access_token'][0],
        access_token_secret=key_frame['access_secret'][0],
    )

    tweets = api.GetUserTimeline(screen_name="realDonaldTrump", count=count)
    tweets = [s.text for s in tweets]
    counter = CountVectorizer(input='content', stop_words="english")
    counts = counter.fit_transform(tweets).toarray().sum(axis=0)
    words = counter.get_feature_names()
    freqs = dict(zip(words, counts))

    # remove some junk words
    for word in ['https', 'rt', 'amp']:
        freqs.pop(word)

    wordcloud = WordCloud().generate_from_frequencies(freqs)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig(img_filename, dpi=500)


if __name__ == '__main__':
    sys.exit(trump_cloud())
