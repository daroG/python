FROM python:3.10

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN python -m pip install poetry==1.1.15 && poetry config virtualenvs.in-project true
COPY poetry.lock pyproject.toml /
RUN poetry install --no-root
COPY rest_image_tracker /rest_image_tracker
COPY setup.cfg /setup.cfg

ENV PATH="/.venv/bin:${PATH}"
ENTRYPOINT [ "uvicorn", "rest_image_tracker.api:app", "--port", "8000" ]
