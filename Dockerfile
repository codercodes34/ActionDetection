FROM python:3.8.7

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
# Define the Gunicorn command as the entrypoint
CMD [ "streamlit", "run", "DataCollection.py" ]
