import cv2
import numpy as np
import dlib
import csv

# Открываем видеофайл
video_path = 'C:/Users/Таня/PycharmProjects/paresis_detection/практика/Measurementsmp4/Andreeva_1_27.04.19.mp4'
cap = cv2.VideoCapture(video_path)

# Загрузка предобученной модели детекции лица (модель HOG)
detector = dlib.get_frontal_face_detector()
# Загрузка предобученной модели для поиска 68 точек контуров лица
predictor = dlib.shape_predictor('models/shape_predictor_81_face_landmarks.dat')

# Получаем информацию о видео (ширина, высота, количество кадров в секунду и т. д.)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Создаем видеовыход для сохранения результата
#output_path = 'path/to/save/output.mp4'
output = cv2.VideoWriter("outputt.mp4", -1, fps, (frame_width, frame_height), isColor=True)

VPG_result = []
kol = 0
k = []

# Обработка каждого кадра видео
while True:
    # Чтение кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование изображения в пространство цветов HSV
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # Определение диапазона цвета кожи в пространстве HSV
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)

    # Создание маски цветовой сегментации
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    # Применение морфологических операций для удаления шума
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    # Нахождение контуров на маске
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Поиск контура с максимальной площадью (предполагаемое лицо)
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    face_mask = np.zeros_like(frame)
    if max_contour is not None:
        cv2.drawContours(face_mask, [max_contour], 0, (255, 255, 255), -1)

    # Применение маски лица к изображению
    masked_image = cv2.bitwise_and(frame, face_mask)

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # Детекция лиц на кадре
    faces = detector(gray)

    for face in faces:
        # Поиск контуров лица
        landmarks = predictor(gray, face)
        landmarks_points = []
        for i in range(81):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            landmarks_points.append((x, y))

        # Создание маски для лица
        mask = np.zeros_like(gray)
        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, convexhull, 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        skin_mask_cleaned = cv2.morphologyEx(skin_mask_closed, cv2.MORPH_OPEN, kernel)

        # Обрезание видео по контуру лица
        masked_image = cv2.bitwise_and(masked_image, masked_image, mask=skin_mask_cleaned)

        # Закрашивание глаз и губ
        left_eye_points = landmarks_points[36:42]
        right_eye_points = landmarks_points[42:48]
        lips_points = landmarks_points[48:60]
        left_eyebrow_points = landmarks_points[17:22]
        right_eyebrow_points = landmarks_points[22:27]

        cv2.fillConvexPoly(masked_image, np.array(left_eye_points), (0, 0, 0))
        cv2.fillConvexPoly(masked_image, np.array(right_eye_points), (0, 0, 0))
        cv2.fillConvexPoly(masked_image, np.array(lips_points), (0, 0, 0))
        cv2.fillConvexPoly(masked_image, np.array(left_eyebrow_points), (0, 0, 0))
        cv2.fillConvexPoly(masked_image, np.array(right_eyebrow_points), (0, 0, 0))

        # Расчет размера сетки
        grid_size = 8  # Размер сетки (количество рядов и столбцов)

        # Создание сетки на лице
        grid = np.zeros((grid_size + 1, grid_size + 1, 2), dtype=np.int32)
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                x = int((landmarks_points[16][0] - landmarks_points[0][0]) * (i / grid_size) + landmarks_points[0][0])
                y = int((landmarks_points[8][1] - landmarks_points[70][1]) * (j / grid_size) + landmarks_points[70][1])
                grid[j, i] = [x, y]

        # Расчет видеоплетизмограммы для каждого сектора сетки
        VPG_values = []
        for i in range(grid_size):
            for j in range(grid_size):
                x1, y1 = grid[i, j]
                x2, y2 = grid[i, j + 1]
                x3, y3 = grid[i + 1, j]
                x4, y4 = grid[i + 1, j + 1]
                cv2.line(masked_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.line(masked_image, (x1, y1), (x3, y3), (0, 255, 0), 1)
                cv2.line(masked_image, (x2, y2), (x4, y4), (0, 255, 0), 1)
                cv2.line(masked_image, (x3, y3), (x4, y4), (0, 255, 0), 1)

                # Извлечение области изображения в секторе сетки
                sector_image = masked_image[y1:y4, x1:x4]

                # Вычисление цветовых компонентов в секторе
                R = sector_image[:, :, 2]
                G = sector_image[:, :, 1]
                B = sector_image[:, :, 0]

                # Вычисление значения видеоплетизмограммы
                if np.sum(R) + np.sum(B) != 0:
                    VPG = np.sum(G) / (np.sum(R) + np.sum(B))
                    VPG_values.append(VPG)

        #print(VPG_values)

        mean = np.mean(VPG_values) # Среднее значение
        std = np.std(VPG_values) # СКО

        #print(mean)
        #print(std)

        filtered_VPG = []
        for v in VPG_values:
            if (v <= mean + std) and (v >= mean - std):
                filtered_VPG.append(v)

        VPG_result.append(np.mean(filtered_VPG))
        # kol += 1
        # k.append(kol)

        # Вывод значений видеоплетизмограммы для каждого сектора сетки
        # for i, VPG in enumerate(VPG_values):
        #     print(f'Сектор {i + 1}: {VPG}')

        # Запись обработанного кадра в выходное видео
        output.write(masked_image)

    # Отображение кадра
    cv2.imshow('Face Detection', masked_image)

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Путь к файлу CSV
csv_file = 'Results.csv'

# Запись массива в файл CSV
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Значение VPG'])  # Запись заголовка
    writer.writerows(zip(VPG_result))  # Запись значений
print(f"Массив успешно записан в файл CSV: {csv_file}")

# Освобождение ресурсов
cap.release()
output.release()
cv2.destroyAllWindows()