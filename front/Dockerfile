FROM ubuntu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y curl git

RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

WORKDIR /front

COPY . /front

WORKDIR /front/kidneySegNet

RUN npm install

EXPOSE 3000

CMD ["sh", "-c", "npm start --port 3000 --host 0.0.0.0"]