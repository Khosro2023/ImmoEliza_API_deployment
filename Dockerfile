FROM python:3.1.1.4
WORKDIR /app
COPY ./src/app/src
COPY ./requirements.txt/app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "src.app:app","--host=0.0.0.0"]
