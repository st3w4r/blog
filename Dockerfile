FROM golang:latest

ARG version=0.115.2

RUN apt-get update & apt-get upgrade -y
RUN wget https://github.com/gohugoio/hugo/releases/download/v${version}/hugo_${version}_linux-arm64.deb
RUN dpkg -i hugo_${version}_linux-arm64.deb

EXPOSE 1313

CMD hugo server --bind="0.0.0.0"
