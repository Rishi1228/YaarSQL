"""Builds data/college.db, the sample database every question is asked against.

Run: python data/build_db.py
The data is random but seeded, so every run produces the same database.
"""
import random
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "college.db"

DEPARTMENTS = [
    # id, name, building, hod
    (1, "Computer", "A Block", "Dr. Mehta"),
    (2, "IT", "A Block", "Dr. Kulkarni"),
    (3, "Mechanical", "C Block", "Dr. Iyer"),
    (4, "Electrical", "B Block", "Dr. Joshi"),
    (5, "Civil", "D Block", "Dr. Reddy"),
    (6, "Electronics", "B Block", "Dr. Banerjee"),
]

CITIES = ["Mumbai", "Pune", "Delhi", "Ahmedabad", "Surat", "Nagpur", "Jaipur", "Indore"]

FIRST_M = ["Aarav", "Rohan", "Vivaan", "Arjun", "Kabir", "Ishaan", "Rahul", "Aditya", "Siddharth",
           "Yash", "Karan", "Dev", "Nikhil", "Harsh", "Manav", "Pranav", "Rishi", "Shubh", "Om", "Tanmay"]
FIRST_F = ["Ananya", "Diya", "Isha", "Priya", "Sneha", "Kavya", "Riya", "Pooja", "Neha", "Aditi",
           "Meera", "Tanvi", "Shreya", "Darshi", "Khushi", "Nisha", "Sara", "Zoya", "Aisha", "Anjali"]
LAST = ["Shah", "Patel", "Sharma", "Verma", "Gupta", "Mehta", "Desai", "Joshi", "Kulkarni", "Nair",
        "Iyer", "Reddy", "Singh", "Jain", "Vora", "Rao", "Kapoor", "Malhotra", "Chopra", "Bose"]

COURSES = {
    1: ["Data Structures", "Operating Systems", "Machine Learning", "Compilers"],
    2: ["Web Technology", "Computer Networks", "Cloud Computing", "Database Systems"],
    3: ["Thermodynamics", "Fluid Mechanics", "Machine Design"],
    4: ["Power Systems", "Control Systems", "Electrical Machines"],
    5: ["Structural Analysis", "Surveying", "Geotechnics"],
    6: ["Digital Electronics", "Signal Processing", "VLSI Design"],
}

DESIGNATIONS = ["Professor", "Associate Professor", "Assistant Professor"]


def grade_for(marks):
    for cutoff, g in [(85, "A"), (70, "B"), (55, "C"), (40, "D")]:
        if marks >= cutoff:
            return g
    return "F"


def build(path=DB_PATH, seed=42):
    rng = random.Random(seed)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.executescript(
        """
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, building TEXT, hod TEXT);
        CREATE TABLE students (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, gender TEXT, age INTEGER, city TEXT,
            dept_id INTEGER REFERENCES departments(id), year INTEGER, cgpa REAL, fees INTEGER);
        CREATE TABLE faculty (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, dept_id INTEGER REFERENCES departments(id),
            designation TEXT, salary INTEGER, experience INTEGER);
        CREATE TABLE courses (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, dept_id INTEGER REFERENCES departments(id),
            credits INTEGER, faculty_id INTEGER REFERENCES faculty(id));
        CREATE TABLE enrollments (
            student_id INTEGER REFERENCES students(id), course_id INTEGER REFERENCES courses(id),
            marks INTEGER, grade TEXT, PRIMARY KEY (student_id, course_id));
        """
    )
    cur.executemany("INSERT INTO departments VALUES (?,?,?,?)", DEPARTMENTS)

    # Students
    students, used = [], set()
    for sid in range(1, 121):
        while True:
            gender = rng.choice("MF")
            first = rng.choice(FIRST_M if gender == "M" else FIRST_F)
            name = f"{first} {rng.choice(LAST)}"
            if name not in used:
                used.add(name)
                break
        year = rng.randint(1, 4)
        students.append((sid, name, gender, 17 + year + rng.randint(0, 2), rng.choice(CITIES),
                         rng.randint(1, 6), year, round(rng.uniform(5.5, 9.9), 2),
                         rng.choice([85000, 95000, 110000, 125000, 140000])))
    cur.executemany("INSERT INTO students VALUES (?,?,?,?,?,?,?,?,?)", students)

    # Faculty (the HODs plus 3 more per department)
    faculty, fid = [], 1
    for dept_id, _, _, hod in DEPARTMENTS:
        names = [hod] + [f"Prof. {rng.choice(FIRST_M + FIRST_F)} {rng.choice(LAST)}" for _ in range(3)]
        for i, n in enumerate(names):
            exp = rng.randint(15, 30) if i == 0 else rng.randint(2, 20)
            desig = "Professor" if i == 0 else rng.choice(DESIGNATIONS)
            salary = 60000 + exp * 3500 + rng.randint(0, 10) * 1000
            faculty.append((fid, n, dept_id, desig, salary, exp))
            fid += 1
    cur.executemany("INSERT INTO faculty VALUES (?,?,?,?,?,?)", faculty)

    # Courses, taught by faculty of the same department
    courses, cid = [], 1
    for dept_id, names in COURSES.items():
        dept_fac = [f[0] for f in faculty if f[2] == dept_id]
        for n in names:
            courses.append((cid, n, dept_id, rng.choice([2, 3, 4]), rng.choice(dept_fac)))
            cid += 1
    cur.executemany("INSERT INTO courses VALUES (?,?,?,?,?)", courses)

    # Enrollments: each student takes 2-3 courses of their department
    enroll = []
    for s in students:
        options = [c[0] for c in courses if c[2] == s[5]]
        for c in rng.sample(options, k=min(len(options), rng.randint(2, 3))):
            marks = rng.randint(30, 99)
            enroll.append((s[0], c, marks, grade_for(marks)))
    cur.executemany("INSERT INTO enrollments VALUES (?,?,?,?)", enroll)

    con.commit()
    con.close()
    return path


if __name__ == "__main__":
    p = build()
    print(f"Built {p}")
