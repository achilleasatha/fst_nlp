FROM python:3.8-alpine

RUN mkdir -p /fst_app
WORKDIR /fst_app
COPY . ./

RUN echo "http://dl-8.alpinelinux.org/alpine/edge/testing" >> /etc/apk/repositories \
  && apk update && apk add py3-numpy py3-pandas
RUN pip install flask nltk
ENV PYTHONPATH "/usr/lib/python3.8/"
ENV FLASK_APP "flask_app.py"
EXPOSE 5000

CMD ["/bin/sh", "-c", "flask run --host=3.15.233.105"]