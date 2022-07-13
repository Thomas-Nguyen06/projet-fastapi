FROM ubuntu:latest

WORKDIR /deploy

RUN apt-get update -y &&\
    apt-get install -y pip

COPY .docker/requirements.txt /deploy/requirements.txt

RUN pip install --upgrade pip && pip install --upgrade -r /deploy/requirements.txt

COPY /api/ /deploy/api/

ENTRYPOINT [ "uvicorn", "--host=0.0.0.0", "--app-dir=/deploy/api", "app:app" ]
EXPOSE 8000
