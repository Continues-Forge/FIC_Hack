# handlers.py
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def get_images_from_folder(folder, num_videos=5):
    """
    Возвращает список путей к изображениям в указанной папке.

    :param folder: Путь к папке, содержащей изображения.
    :return: Список путей к изображениям.
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []

    for filename in os.listdir(folder):
        if any(filename.lower().endswith(ext) for ext in supported_formats):
            image_paths.append(os.path.join(folder, filename))
    
    return image_paths[:num_videos+1]
 

def predict_in_folders(yolo_model_path, image_paths, output_classes=[0]):
    
    yolo_model = YOLO(yolo_model_path)
    detection_data = []
    
    results = yolo_model.predict(
        source=image_paths,
        save=False,
        classes=output_classes,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False  # Отключаем сообщения от YOLO
    )
    
    for result in results:
        image_path = result.path
        for idx, box in enumerate(result.boxes.xyxy):
            bbox = box.cpu().numpy().tolist()
            detection_data.append({
                "image_path": image_path, 
                "bbox": bbox, 
                "id": f"{image_path}_{idx}"
            })
    
    return detection_data
 

def crop_objects(detection_data, target_size=(224, 224)):
    cropped_data = []
    seen_ids = set()

    for detection in detection_data:
        image_path = detection["image_path"]
        bbox = detection["bbox"]
        person_id = detection["id"]

        if person_id not in seen_ids:
            seen_ids.add(person_id)
    
            try:
                with Image.open(image_path) as img:
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    cropped = img.crop((x_min, y_min, x_max, y_max))
                    cropped = make_square(cropped)
                    cropped_resized = cropped.resize(target_size)
                    
                    cropped_data.append({
                        "cropped_image": cropped_resized,
                        "image_path": image_path,
                        "bbox": bbox,
                        "id": person_id
                    })
            except Exception as e:
                print(f"Ошибка обработки файла {image_path}: {e}")

    return cropped_data


def make_square(image):
    width, height = image.size
    size = max(width, height)
    new_image = Image.new("RGB", (size, size), (0, 0, 0))  # Черный фон
    new_image.paste(image, ((size - width) // 2, (size - height) // 2))
    return new_image


def segment_cropped_images_in_memory(segmentation_model_path, cropped_data, target_size=(224, 224)):
    segmentation_model = YOLO(segmentation_model_path)
    segmented_data = []

    for item in cropped_data:
        cropped_image = np.array(item["cropped_image"].convert("RGB"))
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)

        results = segmentation_model.predict(
            source=cropped_image,
            save=False,
            classes=[0],  # Ограничиваем только классом 0 (человек)
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False  # Отключаем сообщения от YOLO
        )

        if not results or not hasattr(results[0], "masks") or results[0].masks is None:
            print(f"Сегментация не обнаружила объектов на изображении: {item['image_path']}")
            continue
        
        person_id = item["id"]
        mask = results[0].masks.data[0]
        mask_array = mask.cpu().numpy()
        mask_array = (mask_array * 255).astype(np.uint8)

        if mask_array.shape[:2] != cropped_image.shape[:2]:
            mask_array = cv2.resize(mask_array, (cropped_image.shape[1], cropped_image.shape[0]))

        color_mask = cv2.cvtColor(mask_array, cv2.COLOR_GRAY2RGB)
        masked_image = cv2.bitwise_and(cropped_image, color_mask)
        masked_resized = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        masked_resized = cv2.resize(masked_resized, target_size)

        segmented_data.append({
            "segmented_image": masked_resized,
            "image_path": item["image_path"],
            "bbox": item["bbox"],
            "id": item["id"]
        })

    return segmented_data


def process_lines_and_generate_stripes(segmented_data, target_size=(224, 224)):
    stripe_images = []
    color_data = []

    for item in segmented_data:
        segmented_image = item["segmented_image"]
        height, width, _ = segmented_image.shape

        stripe_image = np.zeros_like(segmented_image)

        for y in range(height):
            line = segmented_image[y, :, :]
            non_black_pixels = line[line[:, 0] != 0]  # Игнорируем черные пиксели

            if len(non_black_pixels) > 0:
                avg_color = np.mean(non_black_pixels, axis=0)
                stripe_image[y, :, :] = avg_color

        stripe_image_resized = cv2.resize(stripe_image, target_size)
        stripe_images.append({
            "stripe_image": stripe_image_resized,
            "image_path": item["image_path"],
            "bbox": item["bbox"],
            "id": item["id"]
        })

        # Добавляем средний цвет для кластеризации
        color_data.append(np.mean(stripe_image, axis=(0, 1)))  # Средний цвет по всему изображению

    return stripe_images, np.array(color_data)


def perform_clustering(color_data, num_clusters=6):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    clusters = kmeans.fit_predict(color_data)
    return clusters


def perform_dynamic_clustering(stripe_images, color_data, max_stripe_len_so_far):
    num_images = len(stripe_images)  # Количество изображений в кадре
    num_clusters = max(num_images, max_stripe_len_so_far)  # Минимальное количество кластеров = max(текущая длина, максимальная длина за все кадры)
    
    # Кластеризация по цветам полосок
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    clusters = kmeans.fit_predict(color_data)
    
    # Обновляем максимальную длину stripe_images, если текущий кадр имеет больше изображений
    new_max_stripe_len = max(max_stripe_len_so_far, num_images)
    
    return clusters, new_max_stripe_len


def visualize_steps(cropped_data, segmented_data, stripe_images, frame_id, video_id, clusters):
    """
    Визуализирует результаты: вырезанное изображение, сегментированное изображение и полоски.
    Для каждого человека отдельный столбик, все изображения выстроены по одной строке.
    """
    num_people = len(cropped_data)
    plt.figure(figsize=(6, 3 * num_people))  # Сжимаем по оси X и увеличиваем по оси Y

    # Заголовок для текущего кадра
    plt.suptitle(f"Frame {frame_id + 1}", fontsize=14, fontweight='bold')

    # Для каждого человека показываем картинку и ее сегментацию и полоски
    for i, item in enumerate(cropped_data):
        segmented_image = next((seg for seg in segmented_data if seg["id"] == item["id"]), None)
        stripe_image = next((stripe for stripe in stripe_images if stripe["id"] == item["id"]), None)

        if segmented_image and stripe_image:
            # Подпись с номером видео и ID человека
            person_label = f"Video {video_id}"

            # Вырезанное изображение
            plt.subplot(num_people, 3, 3 * i + 1)
            plt.imshow(item["cropped_image"])
            plt.title(f"Cropped\n{person_label}", fontsize=8)
            plt.axis('off')

            # Сегментированное изображение
            plt.subplot(num_people, 3, 3 * i + 2)
            plt.imshow(segmented_image["segmented_image"])
            plt.title(f"Segmented\n{person_label}", fontsize=8)
            plt.axis('off')

            # Полоски (цветовая карта)
            plt.subplot(num_people, 3, 3 * i + 3)
            plt.imshow(stripe_image["stripe_image"])
            plt.title(f"Stripes\nCluster: {clusters[i]}", fontsize=8)
            plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Убираем лишние пробелы между изображениями
    plt.show()