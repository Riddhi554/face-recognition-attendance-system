import face_recognition as fr
import numpy as np
import time
import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

exclude_names = ['Unknown', 'HOD', 'Principal']
lecture_count = 1
MAX_LECTURES = 5

def get_current_lecture():
    global lecture_count
    today = time.strftime('%d_%m_%Y')
    session_file = f'Records/{today}/lecture_session.txt'
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            lecture_count = int(f.read().strip())
    else:
        os.makedirs(f'Records/{today}', exist_ok=True)
        lecture_count = 1
        with open(session_file, 'w') as f:
            f.write(str(lecture_count))
    return lecture_count

def increment_lecture():
    global lecture_count
    lecture_count += 1
    today = time.strftime('%d_%m_%Y')
    session_file = f'Records/{today}/lecture_session.txt'
    with open(session_file, 'w') as f:
        f.write(str(lecture_count))

def encode_faces():
    encoded_data = {}
    for dirpath, dnames, fnames in os.walk("./Images"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file(f"Images/{f}")
                encoding = fr.face_encodings(face)[0]
                encoded_data[f.split(".")[0]] = encoding
    return encoded_data

def Attendance(name):
    if name in exclude_names:
        return
    today = time.strftime('%d_%m_%Y')
    current_lecture = get_current_lecture()
    date_directory = f'Records/{today}'
    os.makedirs(date_directory, exist_ok=True)
    filename = f'{date_directory}/lecture_{current_lecture}.csv'
    with open(filename, 'a+'):
        pass
    with open(filename, 'r') as f:
        data = f.readlines()
        names = [line.split(',')[0] for line in data]
    if name not in names:
        current_time = time.strftime('%H:%M:%S')
        with open(filename, 'a') as fs:
            fs.write(f"{name}, {current_time}\n")

def process_frame(frame, encoded_faces, faces_name):
    face_locations = fr.face_locations(frame)
    unknown_face_encodings = fr.face_encodings(frame, face_locations)
    face_names = []
    for face_encoding in unknown_face_encodings:
        matches = fr.compare_faces(encoded_faces, face_encoding)
        name = "Unknown"
        face_distances = fr.face_distance(encoded_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = faces_name[best_match_index]
        face_names.append(name)
    return face_locations, face_names

def handle_real_time():
    faces = encode_faces()
    encoded_faces = list(faces.values())
    faces_name = list(faces.keys())
    video_frame = True
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break
        if video_frame:
            face_locations, face_names = process_frame(frame, encoded_faces, faces_name)
        video_frame = not video_frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left - 20, top - 20), (right + 20, bottom + 20), (0, 255, 0), 2)
            cv2.rectangle(frame, (left - 20, bottom - 15), (right + 20, bottom + 20), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left - 20, bottom + 15), font, 0.85, (255, 255, 255), 2)
            Attendance(name)
        cv2.imshow('Real-Time Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

def handle_group_photo(photo_path):
    faces = encode_faces()
    encoded_faces = list(faces.values())
    faces_name = list(faces.keys())
    if os.path.exists(photo_path):
        group_photo = cv2.imread(photo_path)
        face_locations, face_names = process_frame(group_photo, encoded_faces, faces_name)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(group_photo, (left - 20, top - 20), (right + 20, bottom + 20), (0, 255, 0), 2)
            cv2.rectangle(group_photo, (left - 20, bottom - 15), (right + 20, bottom + 20), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(group_photo, name, (left - 20, bottom + 15), font, 0.85, (255, 255, 255), 2)
            Attendance(name)
        cv2.imshow('Group Photo Face Recognition', group_photo)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        messagebox.showerror("Error", "Group photo not found!")

def start_real_time():
    handle_real_time()

def load_group_photo():
    file_path = filedialog.askopenfilename(title="Select Group Photo", filetypes=[("Image Files", "*.jpg *.png")])
    if file_path:
        handle_group_photo(file_path)

def new_lecture_session():
    global lecture_count
    today = time.strftime('%d_%m_%Y')
    if lecture_count < MAX_LECTURES:
        increment_lecture()
        messagebox.showinfo("New Lecture", f"Started a new lecture session: Lecture {lecture_count}")
    else:
        messagebox.showwarning("Lecture Limit Reached", "Maximum of 5 lectures reached for the day.")

root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("400x250")
title_label = tk.Label(root, text="Face Recognition System", font=("Helvetica", 16))
title_label.pack(pady=10)
real_time_btn = tk.Button(root, text="Start Real-Time Video", command=start_real_time, width=25)
real_time_btn.pack(pady=10)
group_photo_btn = tk.Button(root, text="Load Group Photo", command=load_group_photo, width=25)
group_photo_btn.pack(pady=10)
new_lecture_btn = tk.Button(root, text="Start New Lecture", command=new_lecture_session, width=25)
new_lecture_btn.pack(pady=10)
root.mainloop()
