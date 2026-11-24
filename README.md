# EduVision-AI

AI-based college attendance and performance management prototype. Features include:

Python prototype for a college attendance platform with:

- Role-based login for Admin, HOD, Teacher, and Student
- SQLite master data and attendance storage
- Student ERP validation and face enrollment
- Face embeddings using `face_recognition`
- Image, video, and webcam-based attendance capture
- Subject-wise attendance analytics and risk alerts
- Prototype pass/fail prediction using attendance patterns

## Configuration

Create a `.env` file in the project root with unique values for:

```text
JWT_SECRET=<a random secret used to sign authentication tokens>
SECRET_KEY=<a different random secret used by Flask sessions>
DEFAULT_ADMIN_PASSWORD=<a unique password meeting the password policy>
```

Do not commit `.env`. The application requires these values and will not start
without them. The admin account is created with `DEFAULT_ADMIN_PASSWORD` when
the database is first initialized.

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Project Notes

- Student signup uses ERP validation and stores the student login as the ERP ID.
- The local package for recognition utilities is named `face_services/` to avoid shadowing the external `face_recognition` library.
- Local databases, runtime models, and face media are kept out of Git.
- Live camera attendance is available with:

```bash
python main.py --camera-attendance --assignment-id 1 --teacher-id 1
```
