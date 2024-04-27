FROM --platform=linux/amd64 python:3.10-slim
WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

ENV VIRTUAL_ENV /usr/local/

COPY ./requirements.txt ./requirements.txt
COPY ./setup.py ./setup.py

RUN uv pip install --no-cache .

# RUN python -m unidic download

# ENV HF_HUB_ENABLE_HF_TRANSFER=1

COPY ./melo /app/melo


RUN ls
ENV PYTHONUNBUFFERED=1

# CMD ["sleep", "infinity"]
CMD ["python3", "-m","melo.app", "--port", "8888"]