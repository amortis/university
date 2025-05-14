from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import uuid

app = Flask(__name__)

# Глобальная переменная для хранения исходного изображения
original_image = None

# Словарь для преобразования цветовых пространств
COLOR_SPACES = {
    'RGB': cv2.COLOR_BGR2RGB,
    'HSV': cv2.COLOR_BGR2HSV,
    'LAB': cv2.COLOR_BGR2LAB,
    'GRAY': cv2.COLOR_BGR2GRAY,
    'YCrCb': cv2.COLOR_BGR2YCrCb,
    'XYZ': cv2.COLOR_BGR2XYZ,
    'HLS': cv2.COLOR_BGR2HLS
}

# Вспомогательные функции
def base64_to_image(base64_string):
    try:
        img_data = base64.b64decode(base64_string.split(',')[1])
        img = Image.open(BytesIO(img_data))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), None
    except Exception as e:
        return None, str(e)


def image_to_base64(img):
    try:
        _, buffer = cv2.imencode('.png', img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return f'data:image/png;base64,{img_str}'
    except Exception as e:
        return None, str(e)


@app.route('/')
def index():
    return send_file('templates/index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        global original_image
        data = request.json
        img_base64 = data.get('image')
        img, error = base64_to_image(img_base64)
        if img is None:
            return jsonify({'error': f'Ошибка загрузки: {error}'}), 400
        
        # Сохраняем исходное изображение
        original_image = img_base64
        preview = image_to_base64(img)
        return jsonify({'preview': preview})
    except Exception as e:
        return jsonify({'error': f'Ошибка загрузки: {str(e)}'}), 400


@app.route('/reset', methods=['POST'])
def reset_image():
    try:
        global original_image
        if original_image is None:
            return jsonify({'error': 'Нет исходного изображения'}), 400
        
        img, error = base64_to_image(original_image)
        if img is None:
            return jsonify({'error': f'Ошибка обработки: {error}'}), 400
            
        preview = image_to_base64(img)
        return jsonify({'result': preview})
    except Exception as e:
        return jsonify({'error': f'Ошибка сброса: {str(e)}'}), 400


@app.route('/resize', methods=['POST'])
def resize_image():
    try:
        data = request.json
        img_base64 = data.get('image')
        scale = float(data.get('scale', 1.0)) if data.get('scale') else 1.0
        width = int(data.get('width')) if data.get('width') else None
        height = int(data.get('height')) if data.get('height') else None

        # Ограничения на масштаб
        if scale <= 0 or scale > 10:
            return jsonify({'error': 'Масштаб должен быть больше 0 и не более 10'}), 400

        # Ограничения на размеры
        MAX_DIMENSION = 10000  # Максимальный размер в пикселях
        if width and (width <= 0 or width > MAX_DIMENSION):
            return jsonify({'error': f'Ширина должна быть больше 0 и не более {MAX_DIMENSION} пикселей'}), 400
        if height and (height <= 0 or height > MAX_DIMENSION):
            return jsonify({'error': f'Высота должна быть больше 0 и не более {MAX_DIMENSION} пикселей'}), 400

        img, error = base64_to_image(img_base64)
        if img is None:
            return jsonify({'error': f'Ошибка обработки: {error}'}), 400

        h, w = img.shape[:2]
        
        # Проверка на максимальный размер после масштабирования
        if width and height:
            if width * height > MAX_DIMENSION * MAX_DIMENSION:
                return jsonify({'error': 'Результирующее изображение слишком большое'}), 400
        else:
            new_width = int(w * scale)
            new_height = int(h * scale)
            if new_width > MAX_DIMENSION or new_height > MAX_DIMENSION:
                return jsonify({'error': 'Результирующее изображение слишком большое'}), 400

        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR

        if width and height:
            img = cv2.resize(img, (width, height), interpolation=interpolation)
        else:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interpolation)

        result = image_to_base64(img)
        return jsonify({'result': result})
    except ValueError as e:
        return jsonify({'error': f'Некорректные значения параметров: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Ошибка изменения размера: {str(e)}'}), 400


@app.route('/crop', methods=['POST'])
def crop_image():
    try:
        data = request.json
        img_base64 = data.get('image')
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        width = int(data.get('width', 0))
        height = int(data.get('height', 0))

        img, error = base64_to_image(img_base64)
        if img is None:
            return jsonify({'error': f'Ошибка обработки: {error}'}), 400

        h, w = img.shape[:2]
        if x < 0 or y < 0 or x + width > w or y + height > h:
            return jsonify({'error': 'Координаты вырезки вне изображения'}), 400

        img = img[y:y + height, x:x + width]
        result = image_to_base64(img)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': f'Ошибка вырезки: {str(e)}'}), 400


@app.route('/flip', methods=['POST'])
def flip_image():
    try:
        data = request.json
        img_base64 = data.get('image')
        mode = data.get('mode')  # 0: vertical, 1: horizontal, -1: both

        img, error = base64_to_image(img_base64)
        if img is None:
            return jsonify({'error': f'Ошибка обработки: {error}'}), 400

        img = cv2.flip(img, int(mode))
        result = image_to_base64(img)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': f'Ошибка отражения: {str(e)}'}), 400


@app.route('/rotate', methods=['POST'])
def rotate_image():
    try:
        data = request.json
        img_base64 = data.get('image')
        angle = float(data.get('angle', 0))

        img, error = base64_to_image(img_base64)
        if img is None:
            return jsonify({'error': f'Ошибка обработки: {error}'}), 400

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

        result = image_to_base64(img)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': f'Ошибка поворота: {str(e)}'}), 400


@app.route('/brightness_contrast', methods=['POST'])
def adjust_brightness_contrast():
    try:
        data = request.json
        img_base64 = data.get('image')
        brightness = float(data.get('brightness', 0))
        contrast = float(data.get('contrast', 1))

        img, error = base64_to_image(img_base64)
        if img is None:
            return jsonify({'error': f'Ошибка обработки: {error}'}), 400

        img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        result = image_to_base64(img)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': f'Ошибка яркости/контрастности: {str(e)}'}), 400


@app.route('/color_balance', methods=['POST'])
def adjust_color_balance():
    try:
        data = request.json
        img_base64 = data.get('image')
        red = float(data.get('red', 1))
        green = float(data.get('green', 1))
        blue = float(data.get('blue', 1))

        # Ограничения на коэффициенты
        if not (0 <= red <= 2 and 0 <= green <= 2 and 0 <= blue <= 2):
            return jsonify({'error': 'Коэффициенты должны быть от 0 до 2'}), 400

        img, error = base64_to_image(img_base64)
        if img is None:
            return jsonify({'error': f'Ошибка обработки: {error}'}), 400

        # Преобразуем кортеж в список для возможности изменения
        channels = list(cv2.split(img))
        
        # Применяем коэффициенты к каждому каналу
        channels[2] = cv2.multiply(channels[2], red)  # Красный канал
        channels[1] = cv2.multiply(channels[1], green)  # Зеленый канал
        channels[0] = cv2.multiply(channels[0], blue)  # Синий канал
        
        # Объединяем каналы обратно
        img = cv2.merge(channels)

        result = image_to_base64(img)
        return jsonify({'result': result})
    except ValueError as e:
        return jsonify({'error': f'Некорректные значения параметров: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Ошибка цветового баланса: {str(e)}'}), 400


@app.route('/noise', methods=['POST'])
def add_noise():
    try:
        data = request.json
        img_base64 = data.get('image')
        noise_type = data.get('type')

        img, error = base64_to_image(img_base64)
        if img is None:
            return jsonify({'error': f'Ошибка обработки: {error}'}), 400

        if noise_type == 'gaussian':
            noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        elif noise_type == 'salt_pepper':
            noise = np.zeros(img.shape, np.uint8)
            cv2.randu(noise, 0, 255)
            img[noise > 240] = 255
            img[noise < 15] = 0

        result = image_to_base64(img)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': f'Ошибка добавления шума: {str(e)}'}), 400


@app.route('/blur', methods=['POST'])
def blur_image():
    try:
        data = request.json
        img_base64 = data.get('image')
        blur_type = data.get('type')
        ksize = int(data.get('ksize', 5))

        img, error = base64_to_image(img_base64)
        if img is None:
            return jsonify({'error': f'Ошибка обработки: {error}'}), 400

        if blur_type == 'average':
            img = cv2.blur(img, (ksize, ksize))
        elif blur_type == 'gaussian':
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        elif blur_type == 'median':
            img = cv2.medianBlur(img, ksize)

        result = image_to_base64(img)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': f'Ошибка размытия: {str(e)}'}), 400


@app.route('/save', methods=['POST'])
def save_image():
    try:
        data = request.json
        img_base64 = data.get('image')
        format = data.get('format', 'png').lower()
        quality = int(data.get('quality', 90))

        img, error = base64_to_image(img_base64)
        if img is None:
            return jsonify({'error': f'Ошибка обработки: {error}'}), 400

        filename = f'output_{uuid.uuid4()}.{format}'
        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if format == 'jpeg' else []
        cv2.imwrite(filename, img, params)

        return send_file(filename, as_attachment=True, download_name=f'edited_image.{format}')
    except Exception as e:
        return jsonify({'error': f'Ошибка сохранения: {str(e)}'}), 400
    finally:
        if os.path.exists(filename):
            os.remove(filename)


@app.route('/convert_color_space', methods=['POST'])
def convert_color_space():
    try:
        data = request.json
        img_base64 = data.get('image')
        target_space = data.get('space', 'BGR')

        if target_space == 'BGR':
            # Если выбрано BGR, просто возвращаем исходное изображение
            img, error = base64_to_image(img_base64)
            if img is None:
                return jsonify({'error': f'Ошибка обработки: {error}'}), 400
            result = image_to_base64(img)
            return jsonify({'result': result})

        if target_space not in COLOR_SPACES:
            return jsonify({'error': f'Неподдерживаемое цветовое пространство. Доступные: BGR, {", ".join(COLOR_SPACES.keys())}'}), 400

        img, error = base64_to_image(img_base64)
        if img is None:
            return jsonify({'error': f'Ошибка обработки: {error}'}), 400

        # Преобразование в целевое цветовое пространство
        converted = cv2.cvtColor(img, COLOR_SPACES[target_space])
        
        # Если это одноканальное изображение (GRAY), преобразуем его в трехканальное для отображения
        if len(converted.shape) == 2:
            converted = cv2.cvtColor(converted, cv2.COLOR_GRAY2BGR)

        result = image_to_base64(converted)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': f'Ошибка преобразования цветового пространства: {str(e)}'}), 400


@app.route('/find_by_color', methods=['POST'])
def find_by_color():
    try:
        data = request.json
        img_base64 = data.get('image')
        color = data.get('color', [0, 0, 0])  # [R,G,B]
        mode = data.get('mode', 'box')  # 'box' или 'crop'
        max_objects = int(data.get('maxObjects', 5))

        img, error = base64_to_image(img_base64)
        if img is None:
            return jsonify({'error': f'Ошибка обработки: {error}'}), 400

        # Конвертируем RGB в HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Преобразуем RGB в HSV для центрального значения
        rgb_color = np.uint8([[color]])
        hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)[0][0]
        
        # Определяем границы HSV
        h, s, v = hsv_color
        
        # Особый случай для красного цвета
        if h < 10 or h > 170:  # Красный цвет находится около 0 и 180 градусов
            # Создаем две маски для красного цвета
            lower_red1 = np.array([0, 50, 50], dtype=np.uint8)
            upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
            lower_red2 = np.array([170, 50, 50], dtype=np.uint8)
            upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
            
            # Объединяем маски
            mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # Для остальных цветов используем стандартный диапазон
            color_low = np.array([max(0, h - 30), 50, 50], dtype=np.uint8)
            color_high = np.array([min(180, h + 30), 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv_img, color_low, color_high)

        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return jsonify({'error': 'Объекты указанного цвета не найдены'}), 400

        # Сортируем контуры по площади (от большего к меньшему)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_objects]

        # Собираем информацию о найденных объектах
        objects_info = []
        result_img = img.copy()

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            if mode == 'box':
                # Рисуем рамку вокруг объекта
                color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Первый объект зеленый, остальные красные
                cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
                
                # Добавляем текст с номером объекта и координатами центра
                label = f'#{i+1} ({center_x}, {center_y})'
                cv2.putText(result_img, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:  # crop
                # Для режима обрезки берем только первый (самый большой) объект
                if i == 0:
                    result_img = img[y:y+h, x:x+w]
                    break

            objects_info.append({
                'number': i + 1,
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'area': float(area),
                'center_x': int(center_x),
                'center_y': int(center_y)
            })

        result = image_to_base64(result_img)
        return jsonify({
            'result': result,
            'objects': objects_info,
            'total_found': len(contours)
        })
    except Exception as e:
        return jsonify({'error': f'Ошибка поиска по цвету: {str(e)}'}), 400


if __name__ == '__main__':
    app.run(debug=True)