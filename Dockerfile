FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

COPY ./requirements.txt /back/requirements.txt

RUN apt-get update && apt-get install -y gcc libgl1 libglib2.0-0 libpq-dev python3-dev
RUN pip install --no-cache-dir -r /back/requirements.txt

COPY ./back /back
COPY ./worker /worker

WORKDIR /back

CMD ["sh", "-c", "python -m uvicorn src.main:app --host 0.0.0.0 --port 8080"]