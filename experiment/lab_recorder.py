"""
Lab Recorder RCS controller for the multimodal experiment.
HCM Lab – Ain Shams University

Purpose
-------
Communicates with Lab Recorder over its local TCP Remote Control Service
(RCS, port 22345). The experiment creates all local LSL outlets first, then
this controller refreshes the stream list, selects all visible streams,
starts recording automatically, and stops recording during cleanup.

The current main experiment intentionally leaves the XDF filename/path under
operator control in the Lab Recorder GUI. The main script does not send a
filename command; it uses RCS only for update/select/start/stop. The optional
set_filename() and setup_and_start() methods remain available for standalone
automation but are not used by the current main workflow.

Late-joining streams can be captured by reselect_all(), which is scheduled
after recording begins. RCS must be enabled in Lab Recorder before running
the experiment. If RCS is unreachable, the experiment continues without XDF
and reports this explicitly.

Dependencies: Python standard library (socket, os, time)
"""

import socket
import time
import os


RCS_HOST    = 'localhost'
RCS_PORT    = 22345
RCS_TIMEOUT = 5.0


class LabRecorderController:

    def __init__(self,
                 study_root    = 'Experiment_Data',
                 subject_id    = '001',
                 subject_name  = '',
                 session       = '001',
                 task          = 'DeceptionDetection',
                 run           = '001',
                 host          = RCS_HOST,
                 port          = RCS_PORT,
                 subject_folder = None):
        """
        Parameters
        ----------
        subject_folder : str or None
            If provided, XDF is saved inside subject_folder/XDF/.
            This puts the XDF alongside Audio/, Behavioral/, etc.
            If None, falls back to study_root/sub-{id}_{name}_ses-{ses}/.
        """
        self.study_root     = os.path.abspath(study_root)
        self.subject_id     = subject_id
        self.subject_name   = subject_name.replace(' ', '_') if subject_name else ''
        self.session        = session
        self.task           = task
        self.run            = run
        self.host           = host
        self.port           = port
        self.subject_folder = subject_folder   # full path to subject data folder
        self._sock          = None
        self._connected     = False
        self._xdf_path      = None

    # ── Connection ────────────────────────────────────────────

    def connect(self):
        """Connect to Lab Recorder RCS. Returns True on success."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(RCS_TIMEOUT)
            self._sock.connect((self.host, self.port))
            self._connected = True
            print(f"[LabRecorder] Connected  {self.host}:{self.port}")
            return True
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            self._connected = False
            print(f"[LabRecorder] Cannot connect: {e}")
            return False

    def disconnect(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self._connected = False
        print("[LabRecorder] Disconnected.")

    # ── Low-level send ────────────────────────────────────────

    def _send(self, command: str):
        """Send one newline-terminated command. Returns reply string."""
        if not self._connected:
            print(f"[LabRecorder] Not connected — skipping: {command!r}")
            return ''
        try:
            self._sock.sendall((command + '\n').encode('utf-8'))
            try:
                reply = self._sock.recv(4096).decode('utf-8').strip()
            except socket.timeout:
                reply = ''
            if reply:
                print(f"[LabRecorder] reply: {reply}")
            return reply
        except Exception as e:
            print(f"[LabRecorder] Send error: {e}")
            return ''

    # ── Commands ──────────────────────────────────────────────

    def update(self):
        """Force Lab Recorder to refresh its stream list from the network."""
        self._send("update")

    def select_all(self):
        """Tick every currently visible stream."""
        self._send("select all")

    def set_filename(self):
        """
        Send full absolute XDF path to Lab Recorder.

        If subject_folder is provided (recommended):
          XDF saved inside subject_folder/XDF/ alongside Audio/, Behavioral/, etc.
          Example:
            Sub035_Ahmed_S001_20260524/
              Audio/  Behavioral/  EyeTracking/  Physiological/  Logs/
              XDF/
                sub-035_Ahmed_ses-001_task-DeceptionDetection_run-001.xdf

        If subject_folder is None (fallback):
          XDF saved in study_root/sub-{id}_{name}_ses-{ses}/

        REQUIRES: BIDS checkbox UNCHECKED in Lab Recorder.
        With BIDS ON, Lab Recorder ignores this command entirely.
        """
        # Build filename
        if self.subject_name:
            name_part = f"sub-{self.subject_id}_{self.subject_name}"
        else:
            name_part = f"sub-{self.subject_id}"

        xdf_name = (
            f"{name_part}"
            f"_ses-{self.session}"
            f"_task-{self.task}"
            f"_run-{self.run}.xdf"
        )

        # Choose folder — inside subject_folder/XDF/ if provided
        if self.subject_folder:
            xdf_folder = os.path.join(self.subject_folder, 'XDF')
        else:
            xdf_folder = os.path.join(
                self.study_root,
                f"{name_part}_ses-{self.session}"
            )

        self._xdf_path = os.path.join(xdf_folder, xdf_name)

        # Pre-create folder so Lab Recorder only needs to open the file
        try:
            os.makedirs(xdf_folder, exist_ok=True)
            print(f"[LabRecorder] XDF folder ready: {xdf_folder}")
        except Exception as e:
            print(f"[LabRecorder] Warning — folder pre-create failed: {e}")

        self._send(f"filename {self._xdf_path}")
        print(f"[LabRecorder] XDF path → {self._xdf_path}")

    def start(self):
        """Start XDF recording."""
        self._send("start")
        print("[LabRecorder] Recording STARTED.")

    def stop(self):
        """Stop and close XDF file."""
        self._send("stop")
        print(f"[LabRecorder] Recording STOPPED → {self._xdf_path}")

    def setup_and_start(self):
        """
        Full startup sequence:
          1. update()       — refresh Lab Recorder stream list
          2. select_all()   — tick all currently visible streams
          3. wait 2s        — let late-joining streams appear
          4. update()       — refresh again
          5. select_all()   — tick any new streams
          6. set_filename() — push custom XDF path (BIDS must be OFF)
          7. start()        — begin recording

        All PC2 local LSL streams must be created BEFORE calling this.
        """
        if not self._connected:
            print("[LabRecorder] Not connected.")
            return False

        print("[LabRecorder] Refreshing stream list (pass 1)...")
        self.update()
        time.sleep(0.5)
        self.select_all()

        print("[LabRecorder] Waiting 2s for late-joining streams...")
        time.sleep(2.0)

        print("[LabRecorder] Refreshing stream list (pass 2)...")
        self.update()
        time.sleep(0.5)
        self.select_all()

        time.sleep(0.3)
        self.set_filename()
        time.sleep(0.3)
        self.start()
        return True

    def reselect_all(self):
        """
        Call ~6s after start() to catch streams that appeared after
        recording began. Called via threading.Timer in main_experiment.py.
        """
        if not self._connected:
            return
        print("[LabRecorder] Re-selecting streams (catching late joiners)...")
        self.update()
        time.sleep(0.5)
        self.select_all()
        print("[LabRecorder] Stream re-selection complete.")

    def is_connected(self):
        return self._connected

    def xdf_path(self):
        return self._xdf_path


# ── Check-only (used by sensor preflight) ────────────────────────────────────

def check_lab_recorder(host=RCS_HOST, port=RCS_PORT):
    """Test reachability. Returns (ok: bool, message: str)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect((host, port))
        s.close()
        msg = f"Lab Recorder RCS reachable at {host}:{port}"
        print(f"[LabRecorder] OK   {msg}")
        return True, msg
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        msg = f"Lab Recorder NOT reachable ({e}). Open it and enable RCS."
        print(f"[LabRecorder] FAIL {msg}")
        return False, msg


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    print("Lab Recorder RCS test — Lab Recorder must be open with BIDS OFF.\n")

    lr = LabRecorderController(
        study_root   = 'Experiment_Data',
        subject_id   = '001',
        subject_name = 'Ahmed Mohamed',
        session      = '001',
        task         = 'DeceptionDetection',
        run          = '001',
    )
    if not lr.connect():
        sys.exit(1)

    lr.setup_and_start()
    print(f"\nRecording 10s → {lr.xdf_path()}")

    import threading
    threading.Timer(5.0, lr.reselect_all).start()

    time.sleep(10)
    lr.stop()
    lr.disconnect()
    print("Done.")
