FROM python:3.11-slim

ARG WHEEL=matrixai_core-1.0.0-py3-none-any.whl
COPY ${WHEEL} /tmp/wheel.whl
RUN pip install --no-cache-dir /tmp/wheel.whl && rm /tmp/wheel.whl

EXPOSE 8000

ENTRYPOINT ["matrixai"]
