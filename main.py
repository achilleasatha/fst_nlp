import pandas as pd
from sentiment_analyser.sentiment_classifier import SentimentClassifier
from sentiment_analyser.data_viz import DataPlotter
from sentiment_analyser.data_wrangler import DataWrangler

if __name__ == "__main__":
    data_path = r'C:\Users\Achilles\PycharmProjects\data\FirstSkillsTraining'
    data = pd.read_csv(data_path + r'\FST_NEW_IDs.csv')

    sc = SentimentClassifier()
    dw = DataWrangler(data)
    DataPlotter(dw.data)
