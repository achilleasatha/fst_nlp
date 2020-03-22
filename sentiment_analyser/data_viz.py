import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


class DataPlotter:
    def __init__(self, data):
        self.data = data

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

