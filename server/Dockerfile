FROM python:3.10

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV ENABLE_WEB_INTERFACE=true
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]