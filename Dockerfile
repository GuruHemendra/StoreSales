FROM python:3.12.4-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD python /app/src/app.py
