FROM python:3.10-slim
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*
ENV POETRY_VERSION=1.8.5
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
WORKDIR /app
COPY pyproject.toml poetry.lock* /app/
RUN poetry install --no-root
COPY . /app
CMD ["/bin/bash"]