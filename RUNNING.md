# How to run YaarSQL on another PC

These steps were tested on a fresh clone of this repository (Windows 11, Python 3.14.4).
After step 4 the site runs; nothing else needs to be downloaded, because the database,
dataset, NLTK corpora and evaluation results are already in the repository.

## What you need

| Tool | Version | Check with |
|---|---|---|
| Python | 3.10 or newer (tested on 3.14) | `python --version` |
| Git | any recent version | `git --version` |
| Internet | only for cloning and `pip install` | |

Download Python from <https://www.python.org/downloads/>. On Windows, tick
**"Add python.exe to PATH"** in the installer.

## Steps on Windows (PowerShell or Command Prompt)

**1. Download the project**

```powershell
git clone https://github.com/Rishi1228/YaarSQL.git
cd YaarSQL
```

If you have an SSH key set up on GitHub you can use `git clone git@github.com:Rishi1228/YaarSQL.git` instead.
Without Git, use **Code → Download ZIP** on the GitHub page, unzip it, and open a terminal in that folder.

**2. Create a virtual environment**

```powershell
python -m venv .venv
```

**3. Install the libraries** (Flask, NLTK, NumPy, pytest)

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```

**4. Start the website**

```powershell
.venv\Scripts\python app.py
```

Open <http://127.0.0.1:5000> in a browser. Press `Ctrl+C` in the terminal to stop the server.

**5. (Optional) Run the tests**

```powershell
.venv\Scripts\python -m pytest tests
```

Expected result: `60 passed`.

**6. (Optional) Rebuild everything from scratch**

```powershell
.venv\Scripts\python setup_project.py
```

This re-downloads the NLTK corpora, rebuilds the database and dataset, retrains the HMM taggers
and re-runs the evaluation. It takes about 3–4 minutes. You only need it if you changed the code
or data, or if some files under `data/` or `models/` are missing.

## Steps on macOS / Linux

The same steps, with `.venv/bin/python` instead of `.venv\Scripts\python`
(use `python3` if `python` is not found):

```bash
git clone https://github.com/Rishi1228/YaarSQL.git
cd YaarSQL
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py                  # open http://127.0.0.1:5000
.venv/bin/python -m pytest tests         # optional: 60 passed
.venv/bin/python setup_project.py        # optional: full rebuild
```

## Quick check that it works

On the **Ask the Database** page, type:

```
computer dept me kitney studnts h?
```

You should get:

```sql
SELECT COUNT(*) FROM students JOIN departments ON students.dept_id = departments.id
WHERE departments.name = 'Computer'
```

with the answer **24**.

## Common problems

| Problem | Fix |
|---|---|
| `python` is not recognized | Reinstall Python with "Add python.exe to PATH" ticked, or use `py` instead of `python` on Windows. |
| `No module named flask` (or nltk/numpy) | You ran the plain `python`, not the one in `.venv`. Use `.venv\Scripts\python ...` as shown, or redo step 3. |
| Port 5000 is already in use (common on macOS because of AirPlay) | Start on another port: `.venv\Scripts\python -m flask --app app run --port 5001`, then open <http://127.0.0.1:5001>. |
| `LookupError: Resource ... not found` from NLTK | The `data/nltk_data/` folder is missing or incomplete. Run step 6. |
| Evaluation page is empty, or `models/` files are missing | Run step 6. |
| `pip install` fails behind a proxy or without internet | Connect to the internet; `pip` must download Flask, NLTK, NumPy and pytest once. |
