# TODO: Keep this root Dockerfile aligned with docker/Dockerfile if structure changes.
FROM python:3.11-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e .[dev]

CMD ["python", "-m", "pytest", "-q"]
