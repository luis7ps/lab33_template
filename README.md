# Lab 3.3 - MapReduce and Spark on a mini-cluster of your own

Data Engineering - Block 3 (Distributed Systems). Instructions, rubric and deadline: see the **3.3-lab** handout.

You will run a **real Spark standalone cluster** (1 master + 2 workers = separate JVM processes) inside this Codespace and use a Jupyter notebook as the driver.

## Start

1. Create your own copy: **Use this template -> Create a new repository** (name it `lab33`, keep it **Private**).
2. In your copy: **Code -> Codespaces -> Create codespace on main** (machine type: **2-core**). The first build takes a few minutes: Java 21, Python 3.13 and PySpark 4.1.3 are installed for you.
3. Open `lab_3_3.ipynb` and select the kernel **Python 3 (lab33)** (top right).
4. Check the machine in a terminal: `java -version`, `python --version`, `nproc`, `free -h`.

## Files

| File | What it is |
|---|---|
| `lab_3_3.ipynb` | the lab notebook: **this is what you submit** (plus its HTML export) |
| `lab33_utils.py` | helpers (cluster start/stop, measurements, failure injection). Do not edit it, but do read it |
| `evidence/` | put your 3 screenshots here (see `evidence/README.md`) |
| `requirements.txt`, `.devcontainer/` | the environment definition |

## Good habits

- **Stop your Codespace when you finish** (Command Palette -> *Codespaces: Stop Current Codespace*). Free accounts get 120 core-hours per month: this lab needs about 3 to 4 hours of a 2-core machine, i.e. 6 to 8 core-hours.
- Commit and push your notebook from time to time (your private repo is your backup).
- If something looks stuck: `stop_cluster()` and run the notebook cells of Part 0 again.
- Policy (same as labs 1.3 and 1.4): no AI tools, no plagiarism.
