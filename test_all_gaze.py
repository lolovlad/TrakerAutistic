from AllGaze.pygaze import PyGaze, PyGazeRenderer
import cv2

cap = cv2.VideoCapture(0)
pg = PyGaze()
pgren = PyGazeRenderer()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gaze_result = pg.predict(frame)
    for face in gaze_result:
        print(f"Face bounding box: {face.bbox}")
        pitch, yaw, roll = face.get_head_angles()
        g_pitch, g_yaw = face.get_gaze_angles()
        print(f"Face angles: pitch={pitch}, yaw={yaw}, roll={roll}.")
        print(f"Distance to camera: {face.distance}")
        print(f"Gaze angles: pitch={g_pitch}, yaw={g_yaw}")
        print(f"Gaze vector: {face.gaze_vector}")
        print(f"Looking at camera: {pg.look_at_camera(face)}")

        pgren.render(frame, face, draw_face_bbox=True, draw_face_landmarks=False, draw_3dface_model=False,
                     draw_head_pose=False, draw_gaze_vector=True)

    cv2.imshow("Iris Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break