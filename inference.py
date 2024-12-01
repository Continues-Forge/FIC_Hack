import os
import argparse
import numpy as np
import json  # Добавляем модуль для работы с JSON
from utils import (  # Импортируем обработчики из handlers.py
    get_images_from_folder,
    predict_in_folders,
    crop_objects,
    segment_cropped_images_in_memory,
    process_lines_and_generate_stripes,
    perform_dynamic_clustering,
    visualize_steps
)

def main(args):
    # Пути к моделям
    YOLO_MODEL_PATH = os.path.join(args.mount, "yolo11n.pt")
    SEGMENTATION_MODEL_PATH = os.path.join(args.mount, "yolo11n-seg.pt")

    # Настройка параметров
    frames_folder = args.videos_dir  # Папка с извлеченными кадрами
    save_dir = args.save_dir  # Папка для сохранения предсказаний
    os.makedirs(save_dir, exist_ok=True)  # Создаем папку, если её нет

    num_videos = len(os.listdir(frames_folder))  # Количество кадров
    print(f"Количество файлов в папке: {num_videos}")
    
    max_stripe_len_so_far = 0  # Изначально максимальная длина = 0

    # Получаем словарь кадров из папки
    frame_dict = get_images_from_folder(frames_folder, num_videos=num_videos)
    print(f"Словарь кадров: {frame_dict}")

    # Обработка каждого фрейма по очереди для всех видео
    for frame_id in range(num_videos):  # Можно изменить, чтобы обрабатывать все фреймы
        print(f"Обработка фрейма {frame_id + 1}")
        detection_data_for_json = []

        # Получаем пути изображений для текущего кадра
        image_paths = frame_dict[frame_id]
        if not image_paths:
            print(f"Нет доступных кадров для фрейма {frame_id + 1}")
            continue

        # Определяем номер видео
        video_id = frame_id + 1

        # Этап 1: Детекция объектов
        detection_data = predict_in_folders(YOLO_MODEL_PATH, image_paths, output_classes=[0])
        
        # Этап 2: Обрезка обнаруженных объектов
        cropped_data = crop_objects(detection_data)

        # Этап 3: Сегментация обрезанных изображений
        segmented_data = segment_cropped_images_in_memory(SEGMENTATION_MODEL_PATH, cropped_data)

        # Этап 4: Генерация полос и сбор данных о цветах
        stripe_images, color_data = process_lines_and_generate_stripes(segmented_data)

        # Этап 5: Динамическая кластеризация
        clusters, max_stripe_len_so_far = perform_dynamic_clustering(
            stripe_images, color_data, max_stripe_len_so_far
        )

        # Этап 6: Визуализация результатов
        visualize_steps(cropped_data, segmented_data, stripe_images, frame_id, video_id, clusters)
        
        # Дополнительно: сохраняем результаты в JSON-файл
        for idx, detection in enumerate(detection_data):
            image_path = detection["image_path"]
            bbox = detection["bbox"]  # Параметры bounding box
            object_id = detection["id"]

            # Структура для сохранения
            frame_data = {
                "frame_id": frame_id + 1,
                "object_id": object_id,
                "bb_left": bbox[0],
                "bb_top": bbox[1],
                "bb_width": bbox[2],
                "bb_height": bbox[3],
                "ids": clusters[idx]  # Идентификаторы кластера для объекта
            }

            detection_data_for_json.append(frame_data)
            
            
        for frame_data in detection_data_for_json:
            # Преобразуем все элементы 'ids' в обычные int
            frame_data['ids'] = int(frame_data['ids']) if isinstance(frame_data['ids'], np.int32) else frame_data['ids']

        # Теперь сохраняем данные в JSON
        output_path = os.path.normpath(os.path.join(save_dir, f"clusters_frame_{frame_id + 1}.json"))
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(detection_data_for_json, json_file, ensure_ascii=False, indent=4)  # Сохраняем с отступами для читаемости

        
    print("Обработка завершена.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт для обработки видео и извлечения предсказаний")
    parser.add_argument("--videos_dir", type=str, required=True, help="Путь до папки с видео (например, /private_test/)")
    parser.add_argument("--mount", type=str, required=True, help="Путь до папки с необходимыми файлами (например, веса моделей)")
    parser.add_argument("--save_dir", type=str, required=True, help="Путь до папки для сохранения предсказаний")

    args = parser.parse_args()
    main(args)
