import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

nltk.download("stopwords")
nltk.download('vader_lexicon')


class SentimentClassifier:
    def __init__(self, data=None):
        self.data = data
        self.sid = SentimentIntensityAnalyzer()
        self.stop = stopwords.words("english")
        self.set_score_and_sentiment()

    def set_score_and_sentiment(self):
        self.data['sentiment_score'] = self.data['description']\
            .apply(lambda text: self.sid.polarity_scores(str(text))['compound'])
        self.data['duration_updated'] = self.data.apply(lambda row: self.duration(row), axis=1)
        self.data['sentiment'] = self.data.apply(lambda row: self.sentiment_names(row), axis=1)

    @staticmethod
    def sentiment_names(data_row):
        if data_row['sentiment_score'] < -0.8:
            return 'very negative'
        elif data_row['sentiment_score'] > 0.8:
            return 'very positive'
        elif data_row['sentiment_score'] < -0.4:
            return 'negative'
        if data_row['sentiment_score'] > 0.4:
            return 'positive'
        else:
            return 'neutral'

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

    def select_participant_data(self, participant_id):
        participant_data = self.data[self.data['id'] == participant_id]\
            .sort_values(by=['date'], axis=0, ascending=True)[['date', 'description', 'sentiment', 'sentiment_score']]
        participant_data.columns = ['Date', 'Description', 'Sentiment', 'Sentiment Score']
        return participant_data

    def participant_sentiment_over_time(self, participant_id):
        participant_data = self.select_participant_data(participant_id)
        participant_data[['Date', 'Sentiment Score']].plot.bar()
        plt.ylabel('Sentiment Score', fontsize=10)
        plt.xlabel('Date', fontsize=10)
        plt.xticks(range(0, participant_data.shape[0]), participant_data.Date.values, rotation='vertical')
        plt.title('Sentiment Score over Time for ID: ' + str(participant_id))
        plt.show()

    def participant_word_cloud(self, participant_id):
        participant_data = self.select_participant_data(participant_id)
        single_string = ' '.join(participant_data['Description'])
        wordcloud = WordCloud(stopwords=STOPWORDS,
                              background_color='white', width=250, height=180).generate(single_string)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title('Word Cloud for ID#' + str(id))
        plt.show()

    def overall_word_cloud(self):
        all_descriptions = ' '.join(self.data['description'])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=250, height=180).generate(
            all_descriptions)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title('Word Cloud created from all descriptions')
        plt.show()

    def get_sentiment_breakdown(self):
        return self.data.groupby(['sentiment']).count()[['date']].rename(columns={'date': 'count'}).reset_index(level=0)

    def duration_hist(self):
        self.data.duration.hist()
        plt.show()

    def sentiment_breakdown_by_duration(self):
        agg_data = self.data[['sentiment', 'sentiment_score', 'duration_updated']].copy(deep=True)
        agg_data['sessions_in_bin'] = agg_data['duration_updated'].astype('int')
        agg_data['percentage_of_sessions'] = agg_data['duration_updated'].astype('int')
        agg_data = agg_data.groupby(['duration_updated', 'sentiment']).agg({'sessions_in_bin': 'sum'}).reset_index()
        agg_data['percentage_of_sessions'] = agg_data[
            ['duration_updated', 'sessions_in_bin']].groupby('duration_updated').apply(
            lambda x: round(100 * x / float(x.sum()), 2))
        pd.pivot_table(agg_data[['duration_updated', 'percentage_of_sessions', 'sentiment']], index='duration_updated',
                       columns='sentiment', values='percentage_of_sessions').plot(kind='bar')
        plt.show()

    def sentiment_by_session_count(self):
        agg_session_count_data = self.data[['sentiment', 'sentiment_score', 'duration_updated', 'id', 'date']].copy(
            deep=True)
        agg_session_count_data = agg_session_count_data.sort_values(by=['id', 'date'], axis=0, ascending=True)
        agg_session_count_data['session_count'] = agg_session_count_data.groupby(['id']).cumcount() + 1
        agg_session_count_data[['sentiment_score', 'session_count']].groupby('session_count').mean()[:20].plot.bar()
        plt.show()
