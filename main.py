import matplotlib.pyplot as plt
import os
from src.Operador import Operador

if __name__ == '__main__':
    operador = Operador()
    # Diretório para armazenar resultados, se não existir
    os.makedirs("results/custom_edge", exist_ok=True)
    os.makedirs("results/opencv_edge", exist_ok=True)
    
    for dirpath, _, filenames in os.walk("src/img"):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            # Carregar imagem
            image = operador.load_image(path)

            # Lista de métodos para aplicar
            methods = ["sobel", "prewitt", "scharr"]
            
            # Processar cada método e exibir resultados
            for method in methods:
                # Processar com o método customizado
                edges_custom = operador.edge_detection(path, method=method)
                
                # Processar com OpenCV apenas para Sobel, se necessário
                if method == "sobel":
                    edges_opencv = operador.sobel_opencv(image)
                elif method == "prewitt":
                    edges_opencv = operador.prewitt_opencv(image)
                else:
                    edges_opencv = operador.apply_gradient_filters(image, operador.scharr_x, operador.scharr_y)[0]

                # Calcular SSIM
                score = operador.calculate_ssim(edges_custom, edges_opencv)
                print(f"SSIM entre custom e OpenCV ({method}) para {filename}: {score}")
                
                # Exibir e salvar a imagem dos métodos custom e OpenCV
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 3, 1)
                plt.imshow(image, cmap='gray')
                plt.title(f"Imagem Original - {filename}")
                plt.axis("off")
                
                plt.subplot(1, 3, 2)
                plt.imshow(edges_custom, cmap='gray')
                plt.title(f"Custom {method.capitalize()} - {filename}")
                plt.axis("off")
                # plt.savefig(f"results/custom_edge/{method}_detection_result_{filename}.png", bbox_inches='tight', pad_inches=0)
                
                plt.subplot(1, 3, 3)
                plt.imshow(edges_opencv, cmap='gray')
                plt.title(f"OpenCV {method.capitalize()} - {filename}")
                plt.axis("off")
                plt.savefig(f"results/opencv_edge/{method}_detection_result_{filename}.png", bbox_inches='tight', pad_inches=0)
                
