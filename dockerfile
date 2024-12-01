# Базовый образ
FROM python:3.10-slim

# Установим рабочую директорию
WORKDIR /app

# Скопируем файлы в контейнер
COPY . /app

# Установка системных зависимостей для OpenGL и других графических библиотек
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Установим зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Запуск Python скрипта
CMD ["python", "inference.py"]
