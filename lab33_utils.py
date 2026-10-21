"""
lab33_utils.py - helpers for Lab 3.3 (Distributed Computing: MapReduce and Spark).

What this file does for you
---------------------------
* start_cluster() launches a REAL Spark *standalone* cluster on this machine:
      1 master process + N worker processes (each one a separate JVM) ...
  ... and your notebook plays the role of the *driver*.  Nothing is simulated:
  executors are separate OS processes that talk to the driver over the network
  stack, exactly as they would on N different machines (just with 127.0.0.1).
* get_spark() opens a SparkSession connected to that cluster.
* stages_df() / stage_skew() read Spark's own monitoring REST API, so you can
  *measure* shuffle bytes, task times, etc. without clicking around the web UI.
* kill_worker() / restart_worker() let you break (and heal) the cluster.
* make_dataset() generates the synthetic "ride-hailing trips" data of the lab.

You are NOT expected to modify this file - but you ARE expected to read it:
it is ~300 lines and shows what a cluster manager actually does.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
HOST = "127.0.0.1"
MASTER_PORT = 7077            # RPC port workers/drivers use to talk to the master
MASTER_UI_PORT = 8080         # master web UI (forward this port in Codespaces)
MASTER_URL = f"spark://{HOST}:{MASTER_PORT}"
MASTER_UI = f"http://{HOST}:{MASTER_UI_PORT}"

LOG_DIR = Path("cluster_logs")
_STATE_FILE = LOG_DIR / "cluster.json"
_MASTER_CLASS = "org.apache.spark.deploy.master.Master"
_WORKER_CLASS = "org.apache.spark.deploy.worker.Worker"
_EXECUTOR_CLASS = "org.apache.spark.executor.CoarseGrainedExecutorBackend"


def _spark_home() -> Path:
    import pyspark
    return Path(pyspark.__file__).resolve().parent


def _daemon_env() -> dict:
    env = os.environ.copy()
    env["SPARK_HOME"] = str(_spark_home())
    env["SPARK_LOCAL_IP"] = HOST            # never advertise a public address
    env["SPARK_DAEMON_MEMORY"] = "256m"     # heap of master/worker JVMs (not executors!)
    env["PYSPARK_PYTHON"] = sys.executable  # python used by executors
    env["SPARK_NO_DAEMONIZE"] = "1"
    return env


# --------------------------------------------------------------------------
# Environment check
# --------------------------------------------------------------------------
def _java_version() -> str | None:
    """First 'version' line printed by `java -version`, or None when Java is not installed."""
    import shutil
    java = shutil.which("java")
    if java is None:
        return None
    out = subprocess.run([java, "-version"], capture_output=True, text=True).stderr
    return next((ln.strip() for ln in out.splitlines() if "version" in ln and "JAVA_TOOL_OPTIONS" not in ln), "unknown")


def check_environment() -> None:
    """Print what this machine offers. Run it once at the start, and again if something looks wrong."""
    import platform
    import pyspark
    ram_gib = psutil.virtual_memory().total / 2**30
    print(f"Python {platform.python_version()} | pyspark {pyspark.__version__} | {_java_version() or 'JAVA NOT FOUND'}")
    print(f"CPU cores: {os.cpu_count()} | RAM: {ram_gib:.1f} GiB | free disk: {shutil.disk_usage('.').free / 2**30:.0f} GiB")
    if _java_version() is None:
        print("!! Java was not found. In a Codespace: Command Palette -> 'Codespaces: Rebuild Container'.")
    if ram_gib < 6.5:
        print("!! Less than ~7 GiB of RAM: the cluster still starts, but the 20 M-row test of Part 5c may fail.")
    if (os.cpu_count() or 0) < 2:
        print("!! Only one CPU core: timings will not show any parallelism.")


# --------------------------------------------------------------------------
# Cluster life-cycle
# --------------------------------------------------------------------------
def _http_json(url: str, timeout: float = 3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def _spark_class(main_class: str, *args: str) -> list[str]:
    return [str(_spark_home() / "bin" / "spark-class"), main_class, *args]


def _launch(cmd: list[str], logname: str, extra_env: dict | None = None) -> subprocess.Popen:
    LOG_DIR.mkdir(exist_ok=True)
    log = open(LOG_DIR / logname, "wb")
    env = _daemon_env()
    env.update(extra_env or {})
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)  # survives notebook interrupts


def _save_state(master_pid: int, worker_pids: list[int], cfg: dict) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    _STATE_FILE.write_text(json.dumps({"master": master_pid, "workers": worker_pids, "cfg": cfg}))


def _load_state() -> dict:
    try:
        return json.loads(_STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _spark_daemon_pids(*classes: str) -> list[int]:
    """PIDs of running JVMs whose command line mentions any of the given main classes."""
    found = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any(c in cmd for c in classes):
            found.append(p.info["pid"])
    return found


def _kill_tree(pid: int, sig=signal.SIGKILL) -> None:
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    for child in proc.children(recursive=True):
        try:
            child.send_signal(sig)
        except psutil.NoSuchProcess:
            pass
    try:
        proc.send_signal(sig)
    except psutil.NoSuchProcess:
        pass


def stop_cluster() -> None:
    """Stop master, workers and any executor left behind (safe to call twice)."""
    for pid in _spark_daemon_pids(_MASTER_CLASS, _WORKER_CLASS, _EXECUTOR_CLASS):
        _kill_tree(pid)
    time.sleep(0.5)
    _STATE_FILE.unlink(missing_ok=True)


def start_cluster(workers: int = 2, cores: int = 1, memory: str = "1g",
                  timeout: float = 90.0) -> None:
    """
    Start 1 master + `workers` workers, each offering `cores` cores and `memory`
    to executors.  Blocks until all workers are registered (state ALIVE).
    """
    if _java_version() is None:
        raise RuntimeError("Java (17 or newer) is required to run Spark and was not found: run check_environment().")
    stop_cluster()  # idempotent: clean slate even after a crashed kernel
    t0 = time.time()
    master = _launch(_spark_class(_MASTER_CLASS, "--host", HOST, "--port", str(MASTER_PORT),
                                  "--webui-port", str(MASTER_UI_PORT)), "master.log")
    _save_state(master.pid, [], {"cores": cores, "memory": memory})
    _wait_for(lambda: _master_json() is not None, timeout, "master to come up")
    pids = []
    for i in range(workers):
        pids.append(_start_worker(i, cores, memory).pid)
    _save_state(master.pid, pids, {"cores": cores, "memory": memory})
    _wait_for(lambda: (_master_json() or {}).get("aliveworkers", 0) >= workers,
              timeout, f"{workers} workers to register")
    atexit.register(stop_cluster)
    print(f"Cluster up in {time.time() - t0:.1f}s -> {cluster_status_line()}")
    print(f"  Master RPC : {MASTER_URL}")
    print(f"  Master UI  : {MASTER_UI}   (port {MASTER_UI_PORT})")


def _can_bind(ip: str) -> bool:
    import socket
    try:
        with socket.socket() as sk:
            sk.bind((ip, 0))
        return True
    except OSError:
        return False


def worker_ip(i: int) -> str:
    """Every worker gets its own loopback address (127.0.0.2, 127.0.0.3, ...), so that Spark sees
    different 'hosts' - as it would on a real cluster.  Falls back to 127.0.0.1 if not possible."""
    ip = f"127.0.0.{2 + i}"
    return ip if _can_bind(ip) else HOST


def _worker_dir(i: int) -> Path:
    return (Path("cluster_work") / f"worker{i}").resolve()


def _start_worker(i: int, cores: int, memory: str) -> subprocess.Popen:
    """Each worker gets its own 'disk' (a directory): logs, and the shuffle files of its executor."""
    work = _worker_dir(i)
    (work / "local").mkdir(parents=True, exist_ok=True)
    ip = worker_ip(i)
    cmd = _spark_class(_WORKER_CLASS, "--host", ip, "--webui-port", str(8081 + i),
                       "--cores", str(cores), "--memory", memory,
                       "--work-dir", str(work), MASTER_URL)
    return _launch(cmd, f"worker{i}.log", {"SPARK_LOCAL_DIRS": str(work / "local"),
                                           "SPARK_LOCAL_IP": ip})


def _wait_for(cond, timeout: float, what: str) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if cond():
                return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {what}. See the files in {LOG_DIR}/")


def _master_json():
    try:
        return _http_json(f"{MASTER_UI}/json/")
    except (urllib.error.URLError, ConnectionError, OSError):
        return None


def cluster_status() -> pd.DataFrame:
    """One row per worker, as the master sees it (the same data as the master web UI)."""
    js = _master_json()
    if js is None:
        raise RuntimeError("No master is answering on " + MASTER_UI + " - did you run start_cluster()?")
    rows = [{"worker": w["id"], "state": w["state"], "cores": w["cores"],
             "cores_used": w["coresused"], "memory_MB": w["memory"],
             "memory_used_MB": w["memoryused"]} for w in js["workers"]]
    return pd.DataFrame(rows)


def cluster_status_line() -> str:
    js = _master_json() or {}
    return (f"master {js.get('status', 'DOWN')}, {js.get('aliveworkers', 0)} alive workers, "
            f"{js.get('cores', 0)} cores, {js.get('memory', 0)} MB")


def worker_pids() -> list[int]:
    """PIDs of the worker JVMs (index in the list = worker number)."""
    return list(_load_state().get("workers", []))


def kill_worker(i: int, mode: str = "all") -> None:
    """
    Simulate a failure (SIGKILL: no shutdown hooks, no goodbye messages).
      mode="all"      the whole machine died: kill the worker JVM *and* its executor, and
                      wipe the worker's disk (so its shuffle files are gone for good)
      mode="executor" only the executor process died; the worker (the "machine") survives
    """
    pid = worker_pids()[i]
    if mode == "all":
        _kill_tree(pid)
        time.sleep(0.3)
        shutil.rmtree(_worker_dir(i), ignore_errors=True)
    elif mode == "executor":
        for child in psutil.Process(pid).children(recursive=True):
            if _EXECUTOR_CLASS in " ".join(child.cmdline()):
                child.kill()
    else:
        raise ValueError("mode must be 'all' or 'executor'")
    print(f"[{time.strftime('%H:%M:%S')}] killed worker {i} ({mode})")


def kill_master() -> None:
    """Simulate the crash of the cluster manager (the single point of failure of this setup)."""
    _kill_tree(_load_state()["master"])
    print(f"[{time.strftime('%H:%M:%S')}] killed the master")


def restart_worker(i: int) -> None:
    """Bring a (killed) worker back: a fresh JVM that registers with the master again."""
    st = _load_state()
    old = st["workers"][i]
    _kill_tree(old)
    new = _start_worker(i, st["cfg"]["cores"], st["cfg"]["memory"])
    st["workers"][i] = new.pid
    _STATE_FILE.write_text(json.dumps(st))
    print(f"[{time.strftime('%H:%M:%S')}] restarted worker {i}")


# --------------------------------------------------------------------------
# Spark session + monitoring through Spark's own REST API
# --------------------------------------------------------------------------
def get_spark(app_name: str = "lab33", executor_memory: str = "768m",
              driver_memory: str = "1g", **extra_conf):
    """Open a SparkSession whose executors run on the mini-cluster (the driver is this kernel)."""
    from pyspark.sql import SparkSession
    os.environ["SPARK_LOCAL_IP"] = HOST          # the driver JVM inherits this
    b = (SparkSession.builder.appName(app_name).master(MASTER_URL)
         .config("spark.driver.host", HOST)
         .config("spark.driver.bindAddress", HOST)
         .config("spark.driver.memory", driver_memory)
         .config("spark.executor.memory", executor_memory)
         # All our "machines" are really one host; on a real cluster an executor can never
         # read another machine's disk directly, so switch that shortcut off to be realistic.
         .config("spark.shuffle.readHostLocalDisk", "false")
         .config("spark.pyspark.python", sys.executable)
         .config("spark.pyspark.driver.python", sys.executable)
         .config("spark.ui.showConsoleProgress", "false"))
    for k, v in extra_conf.items():
        b = b.config(k.replace("__", "."), v)
    t0 = time.perf_counter()
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print(f"SparkSession ready in {time.perf_counter() - t0:.1f}s | application UI: "
          f"{spark.sparkContext.uiWebUrl} | app id: {spark.sparkContext.applicationId}")
    return spark


def _app_api(spark, path: str):
    sc = spark.sparkContext
    return _http_json(f"{sc.uiWebUrl}/api/v1/applications/{sc.applicationId}{path}", timeout=10)


def _ts(s: str | None):
    """Parse Spark REST timestamps ('2026-10-21T10:00:00.123GMT')."""
    if not s:
        return None
    return pd.Timestamp(s.replace("GMT", "")).tz_localize("UTC")


def stages_df(spark, include_skipped: bool = False) -> pd.DataFrame:
    """
    One row per stage attempt executed so far by this application.
    Columns: stage, attempt, status, tasks, failed, seconds,
             shuffle_write_MB, shuffle_read_MB.
    (With adaptive execution Spark may list 'SKIPPED' stages - hidden by default.)
    """
    rows = []
    for s in _app_api(spark, "/stages"):
        if s["status"] == "SKIPPED" and not include_skipped:
            continue
        start, end = _ts(s.get("firstTaskLaunchedTime")), _ts(s.get("completionTime"))
        rows.append({
            "stage": s["stageId"], "attempt": s["attemptId"], "status": s["status"],
            "tasks": s["numTasks"], "failed": s["numFailedTasks"] + s.get("numKilledTasks", 0),
            "seconds": round((end - start).total_seconds(), 2) if start is not None and end is not None else None,
            "shuffle_write_MB": round(s["shuffleWriteBytes"] / 1e6, 2),
            "shuffle_read_MB": round(s["shuffleReadBytes"] / 1e6, 2),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["stage", "attempt"]).reset_index(drop=True) if len(df) else df


def new_stages(spark, since: int) -> pd.DataFrame:
    """Stages with id >= `since` (use `since = next_stage_id(spark)` before running a query)."""
    df = stages_df(spark)
    return df[df.stage >= since].reset_index(drop=True) if len(df) else df


def next_stage_id(spark) -> int:
    df = stages_df(spark)
    return int(df.stage.max()) + 1 if len(df) else 0


def stage_skew(spark, stage_id: int, attempt: int = 0) -> pd.DataFrame:
    """
    How evenly was a stage's work spread over its tasks?  Quantiles (min / 25% / median /
    75% / max) of the data each task read from the shuffle and of its running time.
    A max far above the median means data skew: one straggler task decides the stage time.
    """
    js = _app_api(spark, f"/stages/{stage_id}/{attempt}/taskSummary?quantiles=0,0.25,0.5,0.75,1.0")
    q = ["min", "p25", "median", "p75", "max"]
    out = pd.DataFrame({
        "shuffle_read_MB": [round(x / 1e6, 3) for x in js["shuffleReadMetrics"]["readBytes"]],
        "task_seconds": [round(x / 1000, 3) for x in js["executorRunTime"]],
    }, index=q)
    return out.T


def executors_df(spark) -> pd.DataFrame:
    """All executors the application has had (alive and dead), with their task counters."""
    rows = []
    for e in _app_api(spark, "/allexecutors"):
        rows.append({"id": e["id"], "host": e["hostPort"], "alive": e["isActive"],
                     "cores": e["totalCores"], "completed_tasks": e["completedTasks"],
                     "failed_tasks": e["failedTasks"],
                     "remove_reason": (e.get("removeReason") or "")[:60]})
    return pd.DataFrame(rows)


@contextmanager
def timed(label: str = ""):
    """with timed('my query'): ...   -> prints the wall-clock time of the block."""
    t0 = time.perf_counter()
    yield
    print(f"{label}: {time.perf_counter() - t0:.2f} s")


def run(df) -> None:
    """Execute a DataFrame query completely without bringing results to the driver
    (Spark's built-in 'noop' sink - the standard idiom for benchmarking)."""
    df.write.format("noop").mode("overwrite").save()


# --------------------------------------------------------------------------
# Looking at (and breaking) the cluster
# --------------------------------------------------------------------------
def ps_spark() -> pd.DataFrame:
    """The operating-system processes that make up your cluster: one row per process."""
    roles = [("org.apache.spark.deploy.master.Master", "master"),
             ("org.apache.spark.deploy.worker.Worker", "worker"),
             (_EXECUTOR_CLASS, "executor"),
             ("org.apache.spark.deploy.SparkSubmit", "driver (JVM)"),
             ("pyspark.daemon", "python worker (forked by an executor)")]
    rows = []
    for p in psutil.process_iter(["pid", "ppid", "cmdline", "memory_info"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        for key, role in roles:
            if key in cmd:
                rows.append({"role": role, "pid": p.info["pid"], "parent_pid": p.info["ppid"],
                             "memory_MB": round(p.info["memory_info"].rss / 1e6)})
                break
    order = {r: i for i, (_, r) in enumerate(roles)}
    df = pd.DataFrame(rows)
    return df.sort_values(["role", "pid"], key=lambda c: c.map(order) if c.name == "role" else c).reset_index(drop=True) if len(df) else df


def crash_during_reduce(spark, worker: int = 1, delay: float = 1.5, timeout: float = 300.0):
    """
    Arm a background 'saboteur'.  Call it BEFORE you start the job: as soon as one stage has
    finished (the map side) and the next one is running (the reduce side), it waits `delay`
    seconds and crashes `worker` (whole machine, disk included).  Fires once.
    """
    baseline = next_stage_id(spark)

    def _saboteur():
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                st = stages_df(spark)
                st = st[st.stage >= baseline] if len(st) else st
                if len(st) and (st.status == "COMPLETE").any() and (st.status == "ACTIVE").any():
                    time.sleep(delay)
                    kill_worker(worker, "all")
                    return
            except Exception:
                pass
            time.sleep(0.2)

    th = threading.Thread(target=_saboteur, daemon=True)
    th.start()
    return th


def kill_master_after(seconds: float):
    """Crash the master `seconds` from now (in the background, so you can start a job meanwhile)."""
    t = threading.Timer(seconds, kill_master)
    t.start()
    return t


def try_new_application(timeout: float = 150.0) -> None:
    """Try to start a SECOND Spark application (in a separate process) and report what happens."""
    code = ("from lab33_utils import get_spark\n"
            "get_spark('second-application')\n"
            "print('STARTED')\n")
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=timeout, env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent)})
    except subprocess.TimeoutExpired:
        print(f"Still trying after {timeout:.0f}s ... (gave up waiting)")
        return
    out = r.stdout + r.stderr
    dt = time.time() - t0
    if "STARTED" in r.stdout:
        print(f"The second application STARTED in {dt:.0f}s")
    elif "All masters are unresponsive" in out:
        print(f"FAILED after {dt:.0f}s: 'All masters are unresponsive! Giving up.'")
    else:
        print(f"FAILED after {dt:.0f}s:\n" + out[-500:])


def show_evidence(path: str) -> None:
    """Display a screenshot inside the notebook (so it ends up in your .html) and print its timestamp."""
    from IPython.display import Image, display
    p = Path(path)
    if not p.exists():
        print(f"!! {path} not found - save your screenshot there first")
        return
    print(f"{path}  (file time: {time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))})")
    display(Image(filename=str(p)))


# --------------------------------------------------------------------------
# A small real-text corpus (works offline: the Python language reference)
# --------------------------------------------------------------------------
def corpus_lines() -> list[str]:
    """~10,000 non-empty lines of English text taken from Python's own documentation."""
    from pydoc_data.topics import topics
    return [ln for t in topics.values() for ln in t.splitlines() if ln.strip()]


def write_corpus(path: str = "data/corpus.txt", copies: int = 40) -> str:
    """Write the corpus `copies` times into one text file (so Spark has something big to read)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(corpus_lines()) + "\n"
    with open(p, "w") as f:
        for _ in range(copies):
            f.write(text)
    return str(p)


# --------------------------------------------------------------------------
# Data generation (synthetic, reproducible)
# --------------------------------------------------------------------------
N_ZONES = 260
HOT_ZONE = 1  # "Airport": receives a big share of all trips -> a hot key


def make_dataset(root: str = "data", n_rows: int = 5_000_000, files: int = 8,
                 hot_share: float = 0.35, seed: int = 2026, force: bool = False) -> dict:
    """
    Create (once) a synthetic ride-hailing dataset:
      <root>/trips/part-XXX.parquet   n_rows trips split into `files` files
      <root>/zones.parquet            260 rows: zone_id, zone_name, district
    Zone popularity is Zipf-like and zone 1 ("Airport") receives `hot_share` of
    all trips, so a groupBy / join on zone_id is skewed by construction.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root_p = Path(root)
    trips_dir = root_p / "trips"
    marker = root_p / f"_ok_{n_rows}_{files}_{hot_share}_{seed}"
    if marker.exists() and not force:
        return {"trips": str(trips_dir), "zones": str(root_p / "zones.parquet"), "rows": n_rows}
    trips_dir.mkdir(parents=True, exist_ok=True)
    for old in trips_dir.glob("*.parquet"):
        old.unlink()

    # zone popularity: Zipf(1) over zones 2..260, plus the hot zone with a fixed share
    ranks = np.arange(1, N_ZONES)                    # 259 "ordinary" zones
    p_rest = (1 / ranks) / (1 / ranks).sum() * (1 - hot_share)
    probs = np.concatenate([[hot_share], p_rest])    # index 0 -> zone 1
    zone_ids = np.arange(1, N_ZONES + 1)

    per_file = n_rows // files
    t0 = time.time()
    for f in range(files):
        rng = np.random.default_rng(seed + f)
        n = per_file + (n_rows - per_file * files if f == files - 1 else 0)
        dist = np.round(rng.gamma(shape=2.0, scale=2.2, size=n), 2)           # km
        fare = np.round(2.5 + 1.1 * dist + rng.normal(0, 0.8, size=n).clip(-2, None), 2)
        start = np.datetime64("2026-01-01T00:00:00")
        ts = start + rng.integers(0, 30 * 24 * 3600, size=n).astype("timedelta64[s]")
        table = pa.table({
            "trip_id": np.arange(f * per_file, f * per_file + n, dtype=np.int64),
            "pickup_ts": pa.array(ts.astype("datetime64[us]"), type=pa.timestamp("us")),
            "zone_id": rng.choice(zone_ids, size=n, p=probs).astype(np.int32),
            "driver_id": rng.integers(1, 20_001, size=n).astype(np.int32),
            "passengers": rng.integers(1, 5, size=n).astype(np.int8),
            "distance_km": dist,
            "fare_eur": fare,
            "payment": rng.choice(["card", "cash", "app"], size=n, p=[0.7, 0.2, 0.1]),
        })
        pq.write_table(table, trips_dir / f"part-{f:03d}.parquet", compression="snappy",
                       row_group_size=250_000)

    districts = np.array(["Centre", "North", "South", "East", "West"])
    zones = pd.DataFrame({
        "zone_id": zone_ids.astype(np.int32),
        "zone_name": ["Airport"] + [f"Zone {z:03d}" for z in zone_ids[1:]],
        "district": districts[np.arange(N_ZONES) % len(districts)],
    })
    zones.to_parquet(root_p / "zones.parquet", index=False)
    marker.write_text("ok")
    print(f"Dataset ready in {time.time() - t0:.1f}s: {n_rows:,} trips in {files} files "
          f"({sum(p.stat().st_size for p in trips_dir.glob('*.parquet')) / 1e6:.0f} MB on disk)")
    return {"trips": str(trips_dir), "zones": str(root_p / "zones.parquet"), "rows": n_rows}
