"""rsh.py -- tiny paramiko helper so we do not need sshpass.

Usage
    python rsh.py run "ls /mnt/mydisk"
    python rsh.py put local.py /mnt/mydisk/xxx/remote.py
    python rsh.py get /mnt/mydisk/xxx/f.md local_f.md

Why a helper instead of plain ssh: the school server only accepts a password for
this account, `sshpass` is not installed on the Windows box, and every
non-interactive `ssh` call would otherwise block on a prompt. paramiko is
already in the managed venv.
"""
import base64
import os
import sys

import paramiko

HOST = "10.10.52.210"
USER = "xinguan"
PWD = "xinguan666"
BASE = "/mnt/mydisk/xinguan-dendritic/bench"


def client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PWD, timeout=20)
    return c


def run(cmd, timeout=600, conn=None):
    own = conn is None
    c = conn or client()
    try:
        chan = c.get_transport().open_session()
        chan.settimeout(timeout)
        chan.exec_command(cmd)
        out = b""
        while True:
            if chan.recv_ready():
                out += chan.recv(65536)
            elif chan.exit_status_ready():
                while chan.recv_ready():
                    out += chan.recv(65536)
                break
        err = b""
        while chan.recv_stderr_ready():
            err += chan.recv_stderr(65536)
        return chan.recv_exit_status(), out.decode("utf8", "replace"), err.decode("utf8", "replace")
    finally:
        if own:
            c.close()


def put(local, remote, conn=None):
    """Upload via base64 over the exec channel.

    The server's SFTP subsystem is not usable from this box (stat on an
    existing, world-readable file returns ENOENT), so transfers go through
    `base64 -d` on stdin instead. Same result, one fewer moving part.
    """
    own = conn is None
    c = conn or client()
    try:
        data = base64.b64encode(open(local, "rb").read()).decode()
        chan = c.get_transport().open_session()
        chan.exec_command(f"base64 -d > {remote}")
        chan.sendall(data.encode())
        chan.shutdown_write()
        err = chan.makefile_stderr().read()
        rc = chan.recv_exit_status()
        if rc != 0:
            raise RuntimeError(f"put failed rc={rc}: {err.decode('utf8', 'replace')}")
    finally:
        if own:
            c.close()


def get(remote, local, conn=None):
    own = conn is None
    c = conn or client()
    try:
        rc, out, err = run(f"base64 {remote}", conn=c)
        if rc != 0:
            raise RuntimeError(f"get failed rc={rc}: {err}")
        open(local, "wb").write(base64.b64decode(out))
    finally:
        if own:
            c.close()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run":
        rc, o, e = run(sys.argv[2])
        sys.stdout.write(o)
        sys.stderr.write(e)
        sys.exit(rc)
    elif mode == "put":
        put(sys.argv[2], sys.argv[3])
        print("PUT", sys.argv[3])
    elif mode == "get":
        get(sys.argv[2], sys.argv[3])
        print("GOT", sys.argv[3])
    else:
        raise SystemExit(f"unknown mode {mode}")
