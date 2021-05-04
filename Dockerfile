FROM python:3.7.10-buster

WORKDIR /usr/src/app/

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY asip_v2 asip_v2
WORKDIR /usr/src/app/asip_v2


ENTRYPOINT [ "python" ]
