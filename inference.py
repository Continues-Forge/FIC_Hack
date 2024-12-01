import os
import argparse
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
    for frame_id in range(4):  # Можно изменить, чтобы обрабатывать все фреймы
        print(f"Обработка фрейма {frame_id + 1}")

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

        # Дополнительно: сохраняем результаты
        output_path = os.path.normpath(os.path.join(save_dir, f"clusters_frame_{frame_id + 1}.json"))

        print(f"Результаты для фрейма {frame_id + 1} сохранены в {output_path}")
        # Здесь можно добавить код для сохранения `clusters` в файл
        
    while True:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт для обработки видео и извлечения предсказаний")
    parser.add_argument("--videos_dir", type=str, required=True, help="Путь до папки с видео (например, /private_test/)")
    parser.add_argument("--mount", type=str, required=True, help="Путь до папки с необходимыми файлами (например, веса моделей)")
    parser.add_argument("--save_dir", type=str, required=True, help="Путь до папки для сохранения предсказаний")

    args = parser.parse_args()
    main(args)
