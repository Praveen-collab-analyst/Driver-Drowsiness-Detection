from tkinter import *
import tkinter
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import os
from playsound import playsound  # Cross-platform sound
import platform

# Initialize GUI window
main = tkinter.Tk()
main.title("Driver Drowsiness Monitoring")
main.geometry("700x500")

# EAR: Eye Aspect Ratio
def EAR(drivereye):
    A = dist.euclidean(drivereye[1], drivereye[5])
    B = dist.euclidean(drivereye[2], drivereye[4])
    C = dist.euclidean(drivereye[0], drivereye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# MAR: Mouth Aspect Ratio
def MOR(drivermouth):
    A = dist.euclidean(drivermouth[0], drivermouth[6])
    B = dist.euclidean(drivermouth[2], drivermouth[10])
    C = dist.euclidean(drivermouth[4], drivermouth[8])
    mar = (B + C) / (2.0 * A)
    return mar

# Detect head turn
def detect_head_turn(shape):
    nose = shape[33]
    left_eye_center = np.mean(shape[36:42], axis=0)
    right_eye_center = np.mean(shape[42:48], axis=0)
    eyes_center_x = (left_eye_center[0] + right_eye_center[0]) / 2.0
    offset_x = nose[0] - eyes_center_x
    return offset_x, nose[1]  # horizontal and vertical offset

def play_alert_sound():
    try:
        if platform.system() == "Darwin":  # macOS
            playsound('/System/Library/Sounds/Ping.aiff')
        elif platform.system() == "Windows":
            playsound('C:\\Windows\\Media\\Alarm10.wav')
        else:
            playsound('alert.wav')  # Provide a generic .wav for Linux
    except:
        print("Sound alert failed.")

# Start real-time monitoring
def startMonitoring():
    pathlabel.config(text="Webcam Connected Successfully")

    shape_predictor = 'shape_predictor_68_face_landmarks.dat'

    if not os.path.exists(shape_predictor):
        pathlabel.config(text="Predictor file not found!")
        return

    webcamera = cv2.VideoCapture(0)
    if not webcamera.isOpened():
        pathlabel.config(text="Unable to access webcam")
        return

    # Thresholds
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 10
    MOU_AR_THRESH = 0.75
    HEAD_TURN_THRESHOLD = 25
    HEAD_DOWN_THRESHOLD = 20

    COUNTER = 0
    yawnStatus = False
    yawns = 0
    normal_nose_y = None

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    while True:
        ret, frame = webcamera.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_yawn_status = yawnStatus
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            leftEAR = EAR(leftEye)
            rightEAR = EAR(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = MOR(mouth)

            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

            head_offset_x, nose_y = detect_head_turn(shape)

            if normal_nose_y is None:
                normal_nose_y = nose_y

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            head_moved = False

            if head_offset_x > HEAD_TURN_THRESHOLD:
                cv2.putText(frame, "Head Turned RIGHT", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                head_moved = True
            elif head_offset_x < -HEAD_TURN_THRESHOLD:
                cv2.putText(frame, "Head Turned LEFT", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                head_moved = True
            elif nose_y - normal_nose_y > HEAD_DOWN_THRESHOLD:
                cv2.putText(frame, "Head Looking DOWN", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                head_moved = True

            if head_moved:
                play_alert_sound()

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0) if head_moved else (0, 255, 0), 2)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                cv2.putText(frame, "Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    play_alert_sound()
            else:
                COUNTER = 0
                cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if mar > MOU_AR_THRESH:
                cv2.putText(frame, "Yawning, DROWSINESS ALERT!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                play_alert_sound()
                yawnStatus = True
            else:
                yawnStatus = False

            if prev_yawn_status and not yawnStatus:
                yawns += 1

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "MAR: {:.2f}".format(mar), (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Yawns: {yawns}", (480, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcamera.release()
    cv2.destroyAllWindows()

# GUI Layout
font = ('times', 16, 'bold')
title = Label(main, text='Driver Drowsiness Monitoring System using Live Visual\nBehaviour and Machine Learning', anchor=W, justify=CENTER)
title.config(bg='black', fg='white', font=font)
title.place(x=0, y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Start Monitoring", command=startMonitoring, font=font1)
upload.place(x=50, y=200)

pathlabel = Label(main, bg='DarkOrange1', fg='white', font=font1)
pathlabel.place(x=50, y=250)

main.config(bg='chocolate1')
main.mainloop()
