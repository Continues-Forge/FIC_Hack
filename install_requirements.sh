#!/bin/bash

if [ -f "requirements.txt" ]; then
    echo "Устанавливаем зависимости из requirements.txt..."
    pip install -r requirements.txt
else
    echo "Файл requirements.txt не найден!"
    exit 1
fi
