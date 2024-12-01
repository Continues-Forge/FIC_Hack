
## Предварительные требования

Перед началом работы убедитесь, что у вас установлены:

    Docker (https://docs.docker.com/get-docker/)
    Docker Compose (https://docs.docker.com/compose/install/)


Установка и запуск
### 1. Клонируйте репозиторий


### 2. Настройте структуру папок



### 3. Сборка Docker-образа

Для сборки образа выполните:

docker-compose build

### 4. Запуск контейнера

Для инициализации и запуска сервиса выполните:

```bash
docker-compose up --build
```


Сервис начнёт обработку видео и сохранит результаты в папку data/predictions/.
Параметры запуска

Вы можете настроить пути к папкам с входными данными, моделями и результатами, изменив `docker-compose.yml`:

```docker-compose
command: ["python", "inference.py", "--videos_dir", "/app/data/videos", "--mount", "/app/data/models", "--save_dir", "/app/data/predictions"]
```

Основные параметры:

*    --videos_dir — путь до папки с видео (по умолчанию: /app/data/videos).
*    --mount — путь до папки с моделями (по умолчанию: /app/data/models).
*    --save_dir — путь до папки для сохранения результатов (по умолчанию: /app/data/predictions).
