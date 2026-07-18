"""Run a command (or upload a file) on the EyeVu Pi over SSH (paramiko).

Usage:
    python _pi.py "shell command"          # run, print stdout/stderr/exit
    python _pi.py put LOCAL REMOTE         # sftp upload
Sudo: prefix commands with  echo 1111 | sudo -S ...
"""
import sys, time, paramiko

HOST, USER, PW = "192.168.137.133", "roch", "1111"


def client():
    # The Pi can be heavily loaded (RITnet pegs the single ARMv6 core), so sshd
    # is slow to answer — retry the banner/handshake patiently.
    last = None
    for attempt in range(6):
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            c.connect(HOST, username=USER, password=PW, timeout=60,
                      banner_timeout=60, auth_timeout=60)
            return c
        except Exception as e:                       # noqa: BLE001
            last = e
            time.sleep(5)
    raise last


def main():
    if sys.argv[1] == "put":
        c = client()
        sftp = c.open_sftp()
        sftp.put(sys.argv[2], sys.argv[3])
        print(f"uploaded {sys.argv[2]} -> {sys.argv[3]}")
        sftp.close(); c.close(); return
    cmd = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    c = client()
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout, get_pty=False)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    sys.stdout.write(out)
    if err.strip():
        sys.stdout.write("\n--- stderr ---\n" + err)
    print(f"\n[exit {rc}]")
    c.close()


if __name__ == "__main__":
    main()
