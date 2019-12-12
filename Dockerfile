FROM python:3.7-slim

EXPOSE 8501

WORKDIR /app

COPY src src
COPY app.py app.py
COPY requirements.txt requirements.txt

RUN pip install --trusted-host pypi.python.org -r requirements.txt
