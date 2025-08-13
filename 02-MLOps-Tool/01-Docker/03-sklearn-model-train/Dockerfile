FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY model_trainer.py /app/

CMD ["python", "./model_trainer.py"]