import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("stopwords")
nltk.download('vader_lexicon')


class SentimentClassifier:
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
        self.stop = stopwords.words("english")

    def get_sentiment_score_and_name(self, text):
        sentiment_score = self.sid.polarity_scores(str(text))['compound']
        sentiment_name = self.sentiment_names(sentiment_score)
        return sentiment_score, sentiment_name

    @staticmethod
    def sentiment_names(sentiment_score):
        if sentiment_score < -0.8:
            return 'very negative'
        elif sentiment_score > 0.8:
            return 'very positive'
        elif sentiment_score < -0.4:
            return 'negative'
        if sentiment_score > 0.4:
            return 'positive'
        else:
            return 'neutral'
