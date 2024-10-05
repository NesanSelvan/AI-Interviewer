import cv2
import mediapipe as mp
import os
import math 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
def faceDetection():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    my_drawing_specs = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
    cap = cv2.VideoCapture(0)  
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image = cv2.resize(image, (640, 480))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=my_drawing_specs
                    )
                    nose_tip = face_landmarks.landmark[1]  
                    chin = face_landmarks.landmark[152] 
                    left_cheek = face_landmarks.landmark[234] 
                    right_cheek = face_landmarks.landmark[454] 
                    forehead = face_landmarks.landmark[10]  
                    glabella_pos = face_landmarks.landmark[9] 
                    left_eye = face_landmarks.landmark[468]  
                    right_eye = face_landmarks.landmark[473] 
                    left_eye_end1 = face_landmarks.landmark[33] 
                    left_eye_end2 = face_landmarks.landmark[155] 
                    right_eye_end1 = face_landmarks.landmark[362] 
                    right_eye_end2 = face_landmarks.landmark[263] 
                    # Calculate the pixel coordinates
                    h, w, _ = image.shape
                    forehead_coords = (int(forehead.x * w), int(forehead.y * h))
                    nose_tip_coords = (int(nose_tip.x * w), int(nose_tip.y * h))
                    chin_coords = (int(chin.x * w), int(chin.y * h))
                    left_cheek_coords = (int(left_cheek.x * w), int(left_cheek.y * h))
                    right_cheek_coords = (int(right_cheek.x * w), int(right_cheek.y * h))
                    glabella_coords = (int(glabella_pos.x * w), int(glabella_pos.y * h))
                    left_eye_outer_coords = (int(left_eye.x * w), int(left_eye.y * h))
                    right_eye_outer_coords = (int(right_eye.x * w), int(right_eye.y * h))
                    left_eye_end1_cords =  (int(left_eye_end1.x * w), int(left_eye_end1.y * h))
                    left_eye_end2_cords =  (int(left_eye_end2.x * w), int(left_eye_end2.y * h))
                    right_eye_end1_cords =  (int(right_eye_end1.x * w), int(right_eye_end1.y * h))
                    right_eye_end2_cords =  (int(right_eye_end2.x * w), int(right_eye_end2.y * h))
                    # Calculate face height (chin to glabella)
                    face_height = math.dist(chin_coords, glabella_coords)
                    cv2.circle(image, left_eye_outer_coords, 5, (255, 0, 0), -1)
                    cv2.circle(image, right_eye_outer_coords, 5, (255, 0, 0), -1)
                    cv2.circle(image, left_eye_end1_cords, 5, (255, 0, 0), -1)
                    cv2.circle(image, left_eye_end2_cords, 5, (255, 0, 0), -1)
                    cv2.circle(image, right_eye_end1_cords, 5, (255, 0, 0), -1)
                    cv2.circle(image, right_eye_end2_cords, 5, (255, 0, 0), -1)
                    # Define thresholds based on face height
                    super_close_threshold = 250 
                    close_threshold = 170
                    far_threshold = 130  # Adjust this to 100 if necessary
                    super_far_threshold = 95
                    # Determine proximity level
                    is_super_close = face_height >= super_close_threshold
                    is_close = close_threshold <= face_height < super_close_threshold
                    is_far = far_threshold <= face_height < close_threshold
                    is_very_far = face_height < far_threshold and face_height >= super_far_threshold
                    is_super_far = face_height < super_far_threshold
                    # Display proximity on the image
                    if is_super_close:
                        proximity = "Super Close"
                    elif is_close:
                        proximity = "Close"
                    elif is_far:
                        proximity = "Far"
                    elif is_very_far:
                        proximity = "Come close to camera"
                    elif is_super_far:
                        proximity = "Come close to camera"

                    cv2.putText(image, f"Proximity: {proximity}", (10, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    glabella_nose_threshold = 0
                    chin_nose_threshold = 0
                    if is_super_close:
                        glabella_nose_threshold = 35
                        chin_nose_threshold = 40
                    elif is_close:
                        glabella_nose_threshold = 45
                        chin_nose_threshold = 60
                    elif is_far:  # Far or Very Far
                        glabella_nose_threshold = 50
                        chin_nose_threshold = 65
                    elif is_very_far:
                        glabella_nose_threshold = 38
                        chin_nose_threshold = 50
                    glabella_nose_dist = abs(glabella_coords[1] - nose_tip_coords[1])
                    chin_nose_dist = abs(chin_coords[1] - nose_tip_coords[1])
                    if glabella_nose_dist < glabella_nose_threshold: 
                        head_tilt = "Up"
                    elif chin_nose_dist < chin_nose_threshold:
                        head_tilt = "Down"
                    else:
                        head_tilt = "Neutral"
                    cv2.putText(image, f"Head Tilt: {head_tilt}", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cheek_midpoint_x = (left_cheek.x + right_cheek.x) / 2
                    dynamic_threshold_orientation = 0.07 if is_close or is_super_close else 0.05
                    if nose_tip.x < cheek_midpoint_x - dynamic_threshold_orientation:
                        orientation = "Left"
                    elif nose_tip.x > cheek_midpoint_x + dynamic_threshold_orientation:
                        orientation = "Right"
                    else:
                        orientation = "Straight"
                    cv2.putText(image, f"Orientation: {orientation}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Face Height: {int(face_height)} px", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    circle_center = (320, 240)
                    circle_radius = 150
                    cv2.circle(image, circle_center, circle_radius, (255, 255, 0), 2) 
                    def is_outside_circle(point, center, radius):
                        distance = math.dist(point, center)
                        return distance > radius
                    landmarks_outside_circle = []
                    if is_outside_circle(chin_coords, circle_center, circle_radius):
                        landmarks_outside_circle.append("Chin")
                    if is_outside_circle(left_cheek_coords, circle_center, circle_radius):
                        landmarks_outside_circle.append("Left Cheek")
                    if is_outside_circle(right_cheek_coords, circle_center, circle_radius):
                        landmarks_outside_circle.append("Right Cheek")
                    if is_outside_circle(forehead_coords, circle_center, circle_radius):
                        landmarks_outside_circle.append("Forehead")
                    if landmarks_outside_circle:
                        warning_message = "WARNING: "
                        warning_message += ", ".join(landmarks_outside_circle) + " are out of the safe zone!"
                        cv2.putText(image, warning_message, (10, 200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        safe_message = "All landmarks are within the safe zone."
                        cv2.putText(image, safe_message, (10, 200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("My video capture", image)
            if cv2.waitKey(5) & 0xFF == 27:  
                break
            
    cv2.destroyAllWindows()
    cap.release()

# faceDetection()