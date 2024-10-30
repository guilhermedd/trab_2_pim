import matplotlib.pyplot as plt
import os
from src.Operador import Operador


if __name__ == '__main__':
    operador = Operador()
    for dirpath, _, filenames in os.walk("src/img"):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            # Carregar imagem
            image = operador.load_image(path)
            
            # Processar com o seu método
            edges_custom = operador.edge_detection(os.path.join(dirpath, filename), method="sobel")
            
            # Processar com OpenCV
            edges_opencv = operador.sobel_opencv(image)
            
            # Calcular SSIM
            score = operador.calculate_ssim(edges_custom, edges_opencv)
            print(f"SSIM entre custom e OpenCV (Sobel) para {filename}: {score}")
            
            # Exibir e salvar a imagem
            plt.imshow(edges_custom, cmap='gray')
            plt.title(f"Custom Edges - {filename}")
            plt.axis("off")
            plt.savefig(f"results/custom_edge/detection_result_{filename}.png", bbox_inches='tight', pad_inches=0)
            plt.imshow(edges_opencv, cmap='gray')
            plt.title(f"OpenCV Sobel Edges - {filename}")
            plt.axis("off")
            plt.savefig(f"results/opencv_edge/detection_result_{filename}.png", bbox_inches='tight', pad_inches=0)
