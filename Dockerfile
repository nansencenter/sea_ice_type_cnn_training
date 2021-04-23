FROM python:3.7.10-buster

WORKDIR /usr/src/app/

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY asid_v2 asid_v2
WORKDIR /usr/src/app/asid_v2


ENTRYPOINT [ "python" ]
