FROM python:3.12-slim
# WeasyPrint needs Pango/HarfBuzz; DejaVu fonts cover Latin + Cyrillic names.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libharfbuzz-subset0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY cv_screener ./cv_screener
RUN pip install --no-cache-dir -e ".[dev]"
COPY evals ./evals
COPY tests ./tests
ENTRYPOINT ["cvs"]
