import cv2
import numpy as np
from pyzbar.pyzbar import decode

class ImageProcessor:
    @staticmethod
    def load_image(uploaded_file):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def process_pipeline(image, settings):
        """
        Applique les filtres en lisant directement le session_state (settings).
        """
        img = image.copy()
        debug_info = []

        # 1. GÉOMÉTRIE
        rot = settings.get('rotation', 0)
        if rot > 0:
            if rot == 90: img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180: img = cv2.rotate(img, cv2.ROTATE_180)
            elif rot == 270: img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 2. LUMIÈRE & COULEUR (PRE-PROCESS)
        if settings.get('clahe', False):
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

        # 3. FILTRES DE FLOU & LISSAGE
        if settings.get('blur', False):
            k = settings.get('blur_intensity', 5) | 1
            img = cv2.GaussianBlur(img, (k, k), 0)

        if settings.get('bilateral', False):
            d = settings.get('bilateral_d', 9)
            sigma = settings.get('bilateral_sigma', 75)
            img = cv2.bilateralFilter(img, d, sigma, sigma)

        # 4. EFFETS ARTISTIQUES & CONTOURS
        if settings.get('reduce_colors', False):
            data = np.float32(img).reshape((-1, 3))
            k = settings.get('k_colors', 8)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            img = np.uint8(center)[label.flatten()].reshape(img.shape)

        if settings.get('cartoon_mode', False):
            gray_c = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_c = cv2.medianBlur(gray_c, 7)
            edges_c = cv2.adaptiveThreshold(gray_c, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
            img = cv2.bitwise_and(img, cv2.cvtColor(edges_c, cv2.COLOR_GRAY2RGB))
            
        if settings.get('edges', False):
            t1 = settings.get('edge_t1', 50)
            t2 = settings.get('edge_t2', 150)
            edges = cv2.Canny(img, t1, t2)
            img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) # Affichage contours blancs

        # 5. COULEURS SPÉCIFIQUES & MASQUES
        if settings.get('color_extract', False):
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            # Utilisation de valeurs par défaut si non définies
            lower = np.array(settings.get('hsv_lower', [0, 0, 0]))
            upper = np.array(settings.get('hsv_upper', [179, 255, 255]))
            mask = cv2.inRange(hsv, lower, upper)
            
            if settings.get('show_mask', False):
                img = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.bitwise_and(img, img, mask=mask)

        if settings.get('tint_active', False):
            hex_color = settings.get('tint_color', '#0000FF')
            h = hex_color.lstrip('#')
            rgb_color = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            color_layer = np.full(img.shape, rgb_color, dtype=np.uint8)
            img = cv2.addWeighted(gray_rgb, 0.5, color_layer, 0.5, 0)

        # 6. CONVERSION FINALE EN GRIS (SI DEMANDÉ À LA FIN)
        # Note : Si on veut du N&B simple sans effet cartoon couleur avant
        if settings.get('grayscale', False):
             img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
             img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # Retour en 3 canaux pour affichage uniforme

        if settings.get('binary', False):
            if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            is_otsu = settings.get('binary_otsu', True)
            thresh_val = settings.get('binary_thresh', 127)
            method = cv2.THRESH_BINARY + cv2.THRESH_OTSU if is_otsu else cv2.THRESH_BINARY
            val = 0 if is_otsu else thresh_val
            _, img = cv2.threshold(img, val, 255, method)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 7. DÉTECTION (OVERLAY)
        # Visages
        if settings.get('detect_faces', False):
            gray_d = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray_d, 1.1, 5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) # Rouge
                
                if settings.get('detect_eyes', False):
                    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                    roi_gray = gray_d[y:y+h, x:x+w]
                    roi_color = img[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2) # Vert

        # QR Codes
        if settings.get('scan_codes', False):
            decoded = decode(img)
            for obj in decoded:
                data = obj.data.decode("utf-8")
                debug_info.append(f"[{obj.type}] {data}")
                (x, y, w, h) = obj.rect
                cv2.rectangle(img, (x, y), (x + w, y + h), (50, 205, 50), 3)
                cv2.putText(img, obj.type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 205, 50), 2)

        return img, debug_info