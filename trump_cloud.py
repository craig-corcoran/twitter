import sys
import twitter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def trump_cloud():

    df = pd.read_csv('keys.csv')

    api = twitter.Api(
        consumer_key=df['consumer_key'][0],
        consumer_secret=df['consumer_secret'][0],
        access_token_key=df['access_token'][0],
        access_token_secret=df['access_secret'][0],
    )

    statuses = api.GetUserTimeline(screen_name="realDonaldTrump", count=200)  # 200 seems to be max
    tweets = [s.text for s in statuses]
    cv = CountVectorizer(input='content', stop_words="english")
    X = cv.fit_transform(tweets).toarray()
    counts = X.sum(axis=0)
    words = cv.get_feature_names()
    freqs = dict(zip(words, counts))

    blacklist = ['https', 'rt', 'amp']
    for word in blacklist:
        freqs.pop(word)

    wordcloud = WordCloud().generate_from_frequencies(freqs)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('wordcloud.png', dpi=300)

if __name__ == '__main__':
    sys.exit(trump_cloud())