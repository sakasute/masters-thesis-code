FROM python:3.7.9

WORKDIR /code

ENV FLASK_APP=server
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=development

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000

CMD [ "flask", "run" ]