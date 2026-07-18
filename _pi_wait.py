"""Poll the Pi until the RITnet timing run finishes (or OOMs). Background use."""
import time, paramiko

HOST, USER, PW = "192.168.137.133", "roch", "1111"
CMD = ("cd ~/Desktop/Capture; cat _time_out.txt 2>/dev/null; "
       "echo '@@RUN='; pgrep -f _pi_time.py | grep -v pgrep | head -1; "
       "echo '@@OOM='; (echo 1111 | sudo -S dmesg 2>/dev/null) | "
       "grep -i '    process' | tail -1")


def run_once():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PW, timeout=30,
              banner_timeout=30, auth_timeout=30)
    _, out, _ = c.exec_command(CMD, timeout=60)
    txt = out.read().decode("utf-8", "replace")
    c.close()
    return txt


deadline = time.time() + 1500          # up to 25 min
last = ""
while time.time() < deadline:
    try:
        txt = run_once()
        last = txt
        body, _, tail = txt.partition("@@RUN=")
        running = tail.partition("@@OOM=")[0].strip()
        if "INFER_SECONDS=" in body or "DONE" in body:
            print("=== FINISHED ===")
            print(txt)
            break
        if not running and ("Killed process" in txt):
            print("=== OOM-KILLED (no result) ===")
            print(txt)
            break
        if not running and body.strip():
            # process gone but some output present (could be partial)
            print("=== process gone ===")
            print(txt)
            break
        print(f"[{time.strftime('%H:%M:%S')}] still running; output so far:\n{body.strip()[-200:]}")
    except Exception as e:                       # noqa: BLE001 — Pi off-net during infer
        print(f"[{time.strftime('%H:%M:%S')}] unreachable ({type(e).__name__}) — likely mid-inference")
    time.sleep(20)
else:
    print("=== TIMED OUT waiting ===")
    print(last)
