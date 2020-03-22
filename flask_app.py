from flask import Flask, request, render_template
from sentiment_analyser.sentiment_classifier import SentimentClassifier

app = Flask(__name__)
sc = SentimentClassifier()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    req_dict = request.form.to_dict(flat=False)
    text = req_dict['description'][0]
    output_score, output_sentiment = SentimentClassifier.get_sentiment_score_and_name(sc, text)
    return render_template('index.html', prediction_text='The sentiment for this session was '
                                                         '{} '.format(output_sentiment) +
                                                         'with a score of {}'.format(round(output_score, 2)))


if __name__ == "__main__":
    app.run(debug=False)
