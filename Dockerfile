FROM python:3.8

ADD . /base/

WORKDIR /base

RUN python setup.py develop

WORKDIR /base/demos_urbansim

RUN apt-get update && \
	apt-get install -y gcc libhdf5-serial-dev

RUN pip install -r requirements.txt

WORKDIR /base/demos_urbansim
ENTRYPOINT ["python", "-u", "simulate.py", "-c", "-cf", "custom", "-l"]
