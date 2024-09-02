# Dockerfile
FROM python:3.12

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

COPY . .

# Скачиваем модели во время сборки
RUN python download_models.py

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "main.py"]



