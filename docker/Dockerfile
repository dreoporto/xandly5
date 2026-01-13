FROM tensorflow/tensorflow:2.15.0-gpu

RUN apt-get update
RUN pip install --upgrade pip

WORKDIR /app

COPY ./requirements.txt /app
RUN apt remove python3-blinker -y
RUN pip install -r /app/requirements.txt
RUN pip install --no-index -f https://github.com/dreoporto/ptmlib/releases ptmlib

ENV PYTHONPATH="${PYTHONPATH}:/app"
