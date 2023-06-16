import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)

mp_holistic = mp.solutions.holistic # holistic model
mp_drawing = mp.solutions.drawing_utils # drawing utilities
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while webcam.isOpened():
    validacao, frame = webcam.read()
    if not validacao:
        break

    # Convert the BGR image to RGB before processing
    imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    imagem.flags.writeable = False
    results = holistic.process(imagem)

    imagem.flags.writeable = True
    imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR)

    if results.face_landmarks: # if face landmarks detected
        upper_lip = [results.face_landmarks.landmark[i] for i in range(61, 69)]
        lower_lip = [results.face_landmarks.landmark[i] for i in range(146, 162)]
        lip_distance = sum([p.y for p in upper_lip]) - sum([p.y for p in lower_lip])

        if lip_distance > 0.01: # threshold for smile
            cv2.putText(imagem, 'Smiling', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw face landmarks
        mp_drawing.draw_landmarks(imagem, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

    cv2.imshow('Smile Detection', imagem)

    if cv2.waitKey(5) == 27: 
        break

webcam.release()
cv2.destroyAllWindows()
