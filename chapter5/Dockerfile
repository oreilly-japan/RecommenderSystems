FROM python:3.7.8

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends cmake # for xlearn install

RUN pip install --upgrade pip && \
    pip install poetry

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install
