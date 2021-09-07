FROM python:latest

WORKDIR /app

COPY . ./

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python3", "./api.py"]
