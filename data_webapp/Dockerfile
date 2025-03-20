FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml .
RUN pip install poetry==2.1.1 && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-root --no-interaction && \
    poetry run python -m spacy download en_core_web_sm && \
    poetry cache clear pytorch-cpu --all --no-interaction && \
    poetry cache clear PyPI --all --no-interaction

FROM python:3.12-slim AS runtime

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY main.py .

EXPOSE 8080
CMD ["python", "-u", "main.py", "--dev-env"]
