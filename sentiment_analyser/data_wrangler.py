from sentiment_analyser.sentiment_classifier import SentimentClassifier

class DataWrangler:
    def __init__(self, data):
        self.sid = SentimentClassifier().sid
        self.data = data
        self.set_score_and_sentiment()

    def set_score_and_sentiment(self):
        self.data['sentiment_score'] = self.data['description']\
            .apply(lambda text: self.sid.polarity_scores(str(text))['compound'])
        self.data['duration_updated'] = self.data.apply(lambda row: self.duration(row), axis=1)
        self.data['sentiment'] = self.data.apply(lambda row: SentimentClassifier.sentiment_names(row['sentiment_score']), axis=1)

    @staticmethod
    def duration(row):
        if row['duration'] < 1.5:
            return 1
        elif 1.5 <= row['duration'] < 2.5:
            return 2
        elif 2.5 <= row['duration'] < 3.5:
            return 3
        elif 3.5 <= row['duration'] < 4.5:
            return 4
        elif 4.5 <= row['duration'] < 5.5:
            return 5
        elif 5.5 <= row['duration'] < 6.5:
            return 6
        else:
            return 7