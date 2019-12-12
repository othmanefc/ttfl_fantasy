FROM python:3.7-slim

WORKDIR /app

# The data dir is not created, it has to be mounted dynamically onto the
# container
COPY src src
COPY app.py app.py
COPY requirements.txt requirements.txt

RUN pip install --trusted-host pypi.python.org -r requirements.txt

# No entry point, just running the server
CMD ["streamlit", "run", "app.py"]
