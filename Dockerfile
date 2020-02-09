FROM python:3.7.4-slim-stretch

COPY reqs .
RUN pip install -r base.txt

COPY geomadi .
RUN python setup.py install && pip install -e .

CMD sh

