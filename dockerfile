
FROM python:3.11

WORKDIR /usr/src/app

COPY . /usr/src/app

# Настраиваем volume (опционально для справки)
VOLUME ["/usr/src/app/app"]

# Команда по умолчанию
CMD ["python", "-m", "http.server"]
