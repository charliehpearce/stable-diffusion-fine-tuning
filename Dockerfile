FROM python:3.9-slim

# Copy dependencies to tmp
COPY poetry.lock pyproject.toml /tmp/
WORKDIR /tmp
RUN pip install poetry 
RUN poetry config virtualenvs.create false 
RUN poetry install  --no-interaction --no-ansi --no-dev

WORKDIR /app
COPY ./app/app /app/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]