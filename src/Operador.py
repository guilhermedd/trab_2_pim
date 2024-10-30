import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

class Operador:
    def __init__(self):
        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        self.prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        self.prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    def load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return image


    def gaussian_filter(self, image, kernel_size=5, sigma=1.0):
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return cv2.filter2D(image, -1, kernel)




    # 4. Aplicar o filtro derivativo (Gradientes Gx e Gy)
    def apply_gradient_filters(self, image, filter_x, filter_y):
        Gx = cv2.filter2D(image, -1, filter_x)
        Gy = cv2.filter2D(image, -1, filter_y)
        return Gx, Gy


    # 5. Calcular a magnitude e direção do gradiente
    def gradient_magnitude_direction(self, Gx, Gy):
        magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
        direction = np.arctan2(Gy, Gx) * 180 / np.pi  # Convertendo para graus
        return magnitude, direction


    # 6. Supressão de não-máximos (máximos locais)
    def non_max_suppression(self, magnitude, direction):
        suppressed = np.zeros_like(magnitude)
        angle = direction % 180  # Restringe a direção entre 0 e 180 graus

        for i in range(1, magnitude.shape[0] - 1):
            for j in range(1, magnitude.shape[1] - 1):
                q, r = 255, 255  # Valores altos para comparação

                # Verificar direção e vizinhos colineares
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                elif (22.5 <= angle[i, j] < 67.5):
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                # Verifica se a magnitude atual é máxima local
                if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                    suppressed[i, j] = magnitude[i, j]

        return suppressed


    # 7. Função principal para detecção de bordas
    def edge_detection(self, image_path, method='sobel'):
        # Carregar e pré-processar a imagem
        image = self.load_image(image_path)
        smoothed = self.gaussian_filter(image)

        # Escolha de operador
        if method == 'sobel':
            Gx, Gy = self.apply_gradient_filters(smoothed, self.sobel_x, self.sobel_y)
        elif method == 'prewitt':
            Gx, Gy = self.apply_gradient_filters(smoothed, self.prewitt_x, self.prewitt_y)
        else:
            raise ValueError("Operador desconhecido. Escolha 'sobel' ou 'prewitt'.")

        # Calcular magnitude e direção
        magnitude, direction = self.gradient_magnitude_direction(Gx, Gy)

        # Supressão de não-máximos para detectar bordas
        edges = self.non_max_suppression(magnitude, direction)

        return edges


    # Sobel com OpenCV
    def sobel_opencv(self, image):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        return sobel_magnitude


    # Prewitt com OpenCV (usando convolução com filtros personalizados)
    def prewitt_opencv(self, image):
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        prewittx = cv2.filter2D(image, -1, kernelx)
        prewitty = cv2.filter2D(image, -1, kernely)
        prewitt_magnitude = np.sqrt(prewittx ** 2 + prewitty ** 2)
        return prewitt_magnitude


    # Função para calcular o SSIM
    # Função para calcular o SSIM
    def calculate_ssim(self, imageA, imageB):
        score, _ = ssim(imageA, imageB, data_range=255, full=True)  # Especificar o intervalo de dados
        return score