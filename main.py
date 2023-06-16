# Importa os módulos necessários
import cv2
import mediapipe as mp
import webbrowser

# Inicializa a captura de vídeo (0 indica a webcam padrão)
video = cv2.VideoCapture(0)

# Inicializa a solução de mãos do mediapipe
hand = mp.solutions.hands

# Define um objeto Hand com o número máximo de mãos a serem detectadas como 1
Hand = hand.Hands(max_num_hands=1)

# Define um objeto de desenho do mediapipe para desenhar as marcações da mão na imagem
mpDraw = mp.solutions.drawing_utils

website_opened = False

# Iniciamos um loop infinito para processar cada frame da webcam
while True:
    # Capturamos um único frame da webcam
    check,img = video.read()
    
    # Convertendo a imagem capturada para o formato RGB, já que o mediapipe processa imagens nesse formato
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Usamos o objeto Hand para processar a imagem e detectar as mãos
    results = Hand.process(imgRGB)
    
    # Extrair as coordenadas dos pontos de referência da mão detectada
    handPoints = results.multi_hand_landmarks
    
    # Extraindo as dimensões da imagem. Isso será usado para escalar as coordenadas dos pontos de referência para a imagem
    h,w,_ = img.shape
    
    # Criamos uma lista para armazenar as coordenadas dos pontos de referência
    pontos = []

    # Se a mão foi detectada na imagem
    if handPoints:
        # Para cada conjunto de pontos de referência detectados na imagem
        for points in handPoints:
            # Desenhamos os pontos de referência e suas conexões na imagem
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
            
            # Para cada ponto individual de referência
            for id,cord in enumerate(points.landmark):
                # Calculamos as coordenadas x e y do ponto de referência na imagem
                cx, cy = int(cord.x*w), int(cord.y*h)
                
                # Adicionamos as coordenadas à nossa lista de pontos
                pontos.append((cx,cy))

        # Definimos os índices dos pontos de referência para as pontas dos dedos na mão
        dedos = [8, 12, 16, 20]

        # Este é o gesto que queremos detectar. Neste caso, é quando o dedo indicador e o mindinho estão para cima,
        # e os outros dedos estão para baixo. Quando este gesto é detectado, abrimos um site no navegador.
        if pontos[8][1] < pontos[6][1] and pontos[20][1] < pontos[18][1] and pontos[12][1] > pontos[10][1] and pontos[16][1] > pontos[14][1]:
            if not website_opened:
                webbrowser.open("https://youtu.be/WfmXJuurKH8")
                website_opened = True
        else:
            website_opened = False

    # Mostramos a imagem com os pontos de referência desenhados
    cv2.imshow("Imagem",img)
    
    # Verificamos se uma tecla foi pressionada
    key = cv2.waitKey(1) & 0xFF
    
    # Se a tecla ESC foi pressionada, saímos do loop
    if key == 27:  # ASCII value for ESC key
        break

# Liberamos a captura de vídeo e fechamos todas as janelas abertas
video.release()
cv2.destroyAllWindows()