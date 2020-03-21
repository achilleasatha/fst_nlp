import pandas as pd
from sentiment_analyser.sentiment_classifier import SentimentClassifier

if __name__ == "__main__":
    data_path = r'C:\Users\Achilles\PycharmProjects\data\FirstSkillsTraining'
    data = pd.read_csv(data_path + r'\FST_NEW_IDs.csv')

    SentimentClassifier(data=data)
