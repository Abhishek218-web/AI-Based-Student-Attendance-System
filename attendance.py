import cv2
import face_recognition
import numpy as np
import sqlite3
import os
import csv
from datetime import datetime

# ==========================================
# 1. CREATE PROJECT FOLDER AND DATABASE
# ==========================================

os.makedirs("students", exist_ok=True)

conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Students table
cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    student_name TEXT NOT NULL,
    face_encoding BLOB NOT NULL
)
""")

# Attendance table
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    student_name TEXT NOT NULL,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    status TEXT NOT NULL,
    UNIQUE(student_id, date)
)
""")

conn.commit()


# ==========================================
# 2. REGISTER STUDENT
# ==========================================

def register_student():

    student_id = input("Enter Student ID: ").strip()
    student_name = input("Enter Student Name: ").strip()

    if not student_id or not student_name:
        print("Student ID and name cannot be empty.")
        return

    cursor.execute(
        "SELECT student_id FROM students WHERE student_id = ?",
        (student_id,)
    )

    if cursor.fetchone():
        print("Student ID already exists.")
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Cannot access webcam.")
        return

    print("\nLook at the camera.")
    print("Press SPACE to capture your face.")
    print("Press Q to cancel.")

    encoding = None
    face_image = None

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read from webcam.")
            break

        cv2.imshow("Student Registration", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 32:

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            face_locations = face_recognition.face_locations(
                rgb_frame
            )

            if len(face_locations) != 1:
                print("Please show exactly one face.")
                continue

            encodings = face_recognition.face_encodings(
                rgb_frame, face_locations
            )

            if len(encodings) == 1:

                encoding = encodings[0]
                face_image = frame.copy()
                break

    camera.release()
    cv2.destroyAllWindows()

    if encoding is None:
        print("Registration cancelled.")
        return

    encoding_blob = encoding.astype(
        np.float64
    ).tobytes()

    cursor.execute(
        """
        INSERT INTO students
        (student_id, student_name, face_encoding)
        VALUES (?, ?, ?)
        """,
        (
            student_id,
            student_name,
            encoding_blob
        )
    )

    conn.commit()

    image_path = os.path.join(
        "students", f"{student_id}.jpg"
    )

    cv2.imwrite(image_path, face_image)

    print(
        f"\nStudent {student_name} registered successfully!"
    )


# ==========================================
# 3. LOAD REGISTERED STUDENTS
# ==========================================

def load_students():

    cursor.execute("""
        SELECT student_id, student_name, face_encoding
        FROM students
    """)

    records = cursor.fetchall()

    student_ids = []
    student_names = []
    known_encodings = []

    for student_id, name, encoding_blob in records:

        encoding = np.frombuffer(
            encoding_blob,
            dtype=np.float64
        ).copy()

        if encoding.shape == (128,):

            student_ids.append(student_id)
            student_names.append(name)
            known_encodings.append(encoding)

    return student_ids, student_names, known_encodings


# ==========================================
# 4. MARK ATTENDANCE WITHOUT REPEATED MESSAGES
# ==========================================

def mark_attendance(student_id, student_name):

    now = datetime.now()

    today = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    cursor.execute(
        """
        SELECT id
        FROM attendance
        WHERE student_id = ? AND date = ?
        """,
        (student_id, today)
    )

    existing_record = cursor.fetchone()

    if existing_record:
        # Attendance already exists.
        # Return silently to avoid repeated messages.
        return False

    cursor.execute(
        """
        INSERT INTO attendance
        (student_id, student_name, date, time, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            student_id,
            student_name,
            today,
            current_time,
            "Present"
        )
    )

    conn.commit()

    print(
        f"Attendance marked: {student_name} "
        f"at {current_time}"
    )

    return True


# ==========================================
# 5. START FACE RECOGNITION
# ==========================================

def start_attendance():

    student_ids, student_names, known_encodings = (
        load_students()
    )

    if not known_encodings:
        print("No students registered.")
        print("Please register students first.")
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Cannot access webcam.")
        return

    print("\nFace recognition started.")
    print("Press Q to stop attendance.")

    # Track recognition status during this camera session.
    announced_students = set()

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read from webcam.")
            break

        # Resize image to improve processing speed.
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=0.25,
            fy=0.25
        )

        rgb_frame = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        # Detect faces.
        face_locations = face_recognition.face_locations(
            rgb_frame
        )

        # Generate face encodings.
        face_encodings = face_recognition.face_encodings(
            rgb_frame,
            face_locations
        )

        # Students visible in the current frame.
        visible_students = set()

        for face_encoding, location in zip(
            face_encodings,
            face_locations
        ):

            name = "Unknown"
            student_id = None
            color = (0, 0, 255)

            if known_encodings:

                distances = face_recognition.face_distance(
                    known_encodings,
                    face_encoding
                )

                best_match = int(np.argmin(distances))

                if distances[best_match] < 0.50:

                    name = student_names[best_match]
                    student_id = student_ids[best_match]
                    color = (0, 255, 0)

                    visible_students.add(student_id)

                    # Print attendance information only once
                    # during this continuous recognition session.
                    if student_id not in announced_students:

                        marked = mark_attendance(
                            student_id,
                            name
                        )

                        if marked:
                            print(
                                f"{name}: Attendance recorded."
                            )
                        else:
                            print(
                                f"{name}: Already marked "
                                "present today."
                            )

                        announced_students.add(student_id)

            # Face coordinates in the original frame.
            top, right, bottom, left = location

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw face rectangle.
            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                color,
                2
            )

            # Draw name background.
            cv2.rectangle(
                frame,
                (left, bottom - 35),
                (right, bottom),
                color,
                cv2.FILLED
            )

            # Display name.
            cv2.putText(
                frame,
                name,
                (left + 6, bottom - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # If a student leaves the camera view, allow a fresh
        # recognition event if they return.
        announced_students.intersection_update(
            visible_students
        )

        cv2.imshow(
            "AI Attendance System",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Face recognition stopped.")


# ==========================================
# 6. EXPORT ATTENDANCE TO CSV
# ==========================================

def export_attendance():

    cursor.execute("""
        SELECT student_id, student_name, date, time, status
        FROM attendance
        ORDER BY date DESC, time DESC
    """)

    records = cursor.fetchall()

    if not records:
        print("No attendance records available.")
        return

    with open(
        "attendance.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Student ID",
            "Student Name",
            "Date",
            "Time",
            "Status"
        ])

        writer.writerows(records)

    print("\nAttendance exported successfully!")
    print("File location: attendance.csv")


# ==========================================
# 7. MAIN MENU
# ==========================================

def main():

    try:

        while True:

            print("\n===== AI ATTENDANCE SYSTEM =====")
            print("1. Register Student")
            print("2. Start Face Recognition Attendance")
            print("3. Export Attendance to CSV")
            print("4. Exit")

            choice = input(
                "Enter your choice (1-4): "
            ).strip()

            if choice == "1":

                register_student()

            elif choice == "2":

                start_attendance()

            elif choice == "3":

                export_attendance()

            elif choice == "4":

                print("Exiting AI Attendance System.")
                break

            else:

                print("Invalid choice. Please try again.")

    finally:

        conn.close()


# ==========================================
# 8. RUN PROGRAM
# ==========================================

if __name__ == "__main__":
    main()

       
