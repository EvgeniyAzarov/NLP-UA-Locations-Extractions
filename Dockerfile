# First stage -- export dependencies from poetry to install them later with pip
FROM python:3.11 as requirements-stage

WORKDIR /tmp

RUN pip install poetry

COPY ./pyproject.toml ./poetry.lock* /tmp/

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Bild container based on the image with ready-to-go gunicorn and uvicorn 
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

# Copying only dependencies file allows docker to cache installation step, without triggering it each time code changes
COPY --from=requirements-stage /tmp/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app

COPY ./models /app/models

