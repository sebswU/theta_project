# syntax=docker/dockerfile:1
# TODO: Harden image with non-root runtime and pinned OS package versions.
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY . /app

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e .[dev]

CMD ["python", "-m", "pytest", "-q"]
