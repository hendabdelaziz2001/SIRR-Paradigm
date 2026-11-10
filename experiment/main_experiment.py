"""
Main PsychoPy controller for the multimodal deception-detection protocol.
HCM Lab – Ain Shams University

Purpose
-------
Runs the participant GUI and task sequence while acquiring synchronized data
from the local sensors and LSL streams. It creates the subject folder,
starts/stops GSR, voice, and webcam eye tracking, sends experiment markers,
logs behavioral responses, and coordinates Lab Recorder through local RCS.

Acquisition architecture
------------------------
PC2 local sensors:
  GSR        Grove GSR V2.0 → Arduino serial → GSR_EDA/GSR_Markers
  Voice      Boya BY-M1/default input → VoiceAudio/VoiceMarkers + local WAV
  Eye       webcam + MediaPipe → EyeTracking_Data/EyeTrackingMarkers + MP4

PC1 network sensors:
  ANT Neuro EEG 1000 Hz
  Polar H10      HR/RR/HRV, ECG 130 Hz, ACC 25 Hz

The operator sets the desired XDF path/name manually in Lab Recorder. After
all locally created LSL outlets are available, this script automatically
refreshes/selects streams and sends Lab Recorder start/stop commands. A later
re-selection catches late network streams. If RCS is unavailable, the task
continues but no XDF is recorded.

Startup timing
--------------
The main marker outlet is created first. Voice and eye tracking start
immediately. GSR starts in a worker and needs approximately 2 s serial
stabilization plus 5 s rate measurement before its LSL outlet appears. The
controller waits for this GSR initialization before starting Lab Recorder.
EEG/Polar should already be streaming on PC1 before the pre-session check;
their availability is reported as warnings and missing streams must be fixed
manually before a participant session.

Protocol
--------
Block 0: 3-minute resting baseline.
Block 1: experimenter-controlled acquaintance speech.
Two sequence-dependent video rounds, each followed by 12 question trials.
The eye tracker has no participant-facing calibration and provides relative
webcam eye coordinates, not calibrated screen gaze or true pupil diameter.

Markers
-------
Global markers use 1–7, 200–211, and 999. Trial markers use
300 + trial_id*10 + phase, where phase 1–7 means fixation, stimulus, onset,
offset, truth, lie, and ITI respectively. This separate range prevents the
trial-20 collision that existed in the previous N*10+phase scheme.
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import os
import sys
import csv
import time
from datetime import datetime

import threading
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display

from psychopy import visual, core, event, gui, data
import pylsl

from gsr_module               import start_gsr,   stop_gsr,   send_gsr_marker,   get_latest_gsr, get_measured_rate
from voice_module             import start_voice, stop_voice, send_voice_marker,  check_microphone
from psychopy_eeg_eye_tracker import EEG_EyeTracker
from lab_recorder             import LabRecorderController, check_lab_recorder

# ── Suppress verbose pylsl internal INFO logs (netinterfaces.cpp etc.) ────
import os as _os
_os.environ.setdefault('LSLAPICFG_DEFAULT_LOG_LEVEL', 'error')

_eye_tracker = None
# ══════════════════════════════════════════════════════════════════════════════
# ── ARABIC TEXT HELPER ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def ar(text):
    try:
        lines = text.split('\n')
        reshaped_lines = []
        for line in lines:
            reshaped = arabic_reshaper.reshape(line)
            bidi_text = get_display(reshaped)
            reshaped_lines.append(bidi_text)
        return '\n'.join(reshaped_lines)
    except Exception:
        return text


# ══════════════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

CFG = dict(
    # Hardware — PC2 local
    gsr_port        = 'COM3',
    gsr_baud        = 115200,
    # Current laptop camera is OpenCV index 0; update this for the Logitech
    # camera when it is connected in the lab.
    camera_id       = 0,

    # Display
    screen_res      = (1920, 1080),
    fullscreen      = True,
    screen_id       = 0,

    # Timing (seconds)
    fixation_dur          = 1.0,
    iti_dur               = 4.0,     # 4s gives experimenter time to press T or L
    baseline_dur          = 180,     # 3 minutes resting baseline

    # Data
    data_root       = 'Experiment_Data',
    image_folder    = 'stimuli',
    video_a         = 'video_a.mp4',   # name of first video file
    video_b         = 'video_b.mp4',   # name of second video file

    # LSL
    lsl_stream_name = 'PsychoPy_Markers',
    lsl_source_id   = 'PC2_MainExperiment',

    # Lab Recorder RCS
    lab_recorder_host = 'localhost',
    lab_recorder_port = 22345,

    # PC1 expected LSL stream names (as shown in Lab Recorder stream list)
    eeg_stream_name         = 'EE224-000000-000101',
    polar_stream_names      = (
        'PolarH10_HR', 'PolarH10_RR', 'PolarH10_HRV',
        'PolarH10_ECG', 'PolarH10_ACC', 'PolarH10_Markers',
    ),
)

CFG['image_folder'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), CFG['image_folder'])
CFG['data_root'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), CFG['data_root'])
CFG['video_a'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), CFG['video_a'])
CFG['video_b'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), CFG['video_b'])


# ══════════════════════════════════════════════════════════════════════════════
# ── IMAGE HELPER ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

def find_image(folder, base_name):
    """Return full path to image or None. Accepts None base_name gracefully."""
    if not base_name:
        return None
    name_no_ext = os.path.splitext(base_name)[0]
    exact = os.path.join(folder, base_name)
    if os.path.isfile(exact):
        return exact
    for ext in SUPPORTED_EXTENSIONS:
        candidate = os.path.join(folder, name_no_ext + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ── STIMULUS DEFINITIONS ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── Block 1: Acquaintance Test ────────────────────────────────────────────────
# Single slide, all sub-questions shown together, 60s continuous voice recording.
ACQUAINTANCE_QUESTION = """عرّف بنفسك
اسمك — سنك — شغلتك
المحافظة اللي ساكن فيها
ترتيبك في العيلة — حالتك الاجتماعية"""

# ── Block 2: Crime Scene Questions (10 trials, some with image) ───────────────
# These will be reused for both Video A and Video B (with different trial IDs).
# The operator will later edit the questions themselves.
TRIALS_VIDEO_A = [
    {'id': 1,  'image': None, 'question': "احكيلي من البداية، وبالترتيب،\nإيه اللي حصل في الأوضة؟"},
    {'id': 2,  'image': None, 'question': "إيه الحاجة اللي اختفت من الأوضة؟"},
    {'id': 3,  'image': None, 'question': "الحاجة دي كانت موجودة فين\nقبل ما صاحبها يخرج من الأوضة؟"},
    {'id': 4,  'image': None, 'question': "إيه السبب اللي خلّى صاحب الحاجة\nيخرج من الأوضة في بداية الموقف؟"},
    {'id': 5,  'image': None, 'question': "مين الشخص اللي أخد الحاجة\nبعد ما صاحبها خرج؟"},
    {'id': 6,  'image': None, 'question': "بعد ما الشخص مسك الحاجة،\nعمل بيها إيه بعد كده مباشرة؟"},
    {'id': 7,  'image': None, 'question': "الشخص حط الحاجة فين بعد ما أخدها؟"},
    {'id': 8,  'image': None, 'question': "البنت اللي كانت موجودة في الأوضة\nكانت بتعمل إيه وقت ما الحاجة اتاخدت؟"},
    {'id': 9,  'image': None, 'question': "إيه آخر تصرف عمله الشخص اللي أخد الحاجة\nقبل ما يسيب الأوضة ويمشي؟"},
    {'id': 10, 'image': None, 'question': "بعد ما الشخص أخد الحاجة،\nخرج من الأوضة في أنهي اتجاه أو من أنهي باب؟"},
    {'id': 11, 'image': None, 'question': "في اللحظة اللي الشخص اللي أخد الحاجة خرج فيها،\nالحاجة دي كانت موجودة فين بالظبط؟"},
    {'id': 12, 'image': None, 'question': "مين كان آخر شخص مسك الحاجة\nقبل ما صاحبها يرجع الأوضة؟"}
]

TRIALS_VIDEO_B = [
    {'id': 13, 'image': None, 'question': "احكيلي من البداية، وبالترتيب،\nإيه اللي حصل في الأوضة؟"},
    {'id': 14, 'image': None, 'question': "إيه الحاجة اللي اختفت من الأوضة؟"},
    {'id': 15, 'image': None, 'question': "الحاجة دي كانت موجودة فين\nقبل ما صاحبها يخرج من الأوضة؟"},
    {'id': 16, 'image': None, 'question': "إيه السبب اللي خلّى صاحب الحاجة\nيخرج من الأوضة في بداية الموقف؟"},
    {'id': 17, 'image': None, 'question': "مين الشخص اللي أخد الحاجة\nبعد ما صاحبها خرج؟"},
    {'id': 18, 'image': None, 'question': "بعد ما الشخص مسك الحاجة،\nعمل بيها إيه بعد كده مباشرة؟"},
    {'id': 19, 'image': None, 'question': "الشخص حط الحاجة فين بعد ما أخدها؟"},
    {'id': 20, 'image': None, 'question': "البنت اللي كانت موجودة في الأوضة\nكانت بتعمل إيه وقت ما الحاجة اتاخدت؟"},
    {'id': 21, 'image': None, 'question': "إيه آخر تصرف عمله الشخص اللي أخد الحاجة\nقبل ما يسيب الأوضة ويمشي؟"},
    {'id': 22, 'image': None, 'question': "بعد ما الشخص أخد الحاجة،\nخرج من الأوضة في أنهي اتجاه أو من أنهي باب؟"},
    {'id': 23, 'image': None, 'question': "في اللحظة اللي الشخص اللي أخد الحاجة خرج فيها،\nالحاجة دي كانت موجودة فين بالظبط؟"},
    {'id': 24, 'image': None, 'question': "مين كان آخر شخص مسك الحاجة\nقبل ما صاحبها يرجع الأوضة؟"}
]


# ══════════════════════════════════════════════════════════════════════════════
# ── MARKER CODES ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def mk(trial_id, phase):
    # Keep trial markers separate from the global 200-211 video/instruction
    # marker range.  The old N*10+phase scheme collided at trial 20.
    return 300 + trial_id * 10 + phase

MK_FIXATION       = 1   # N1
MK_STIMULUS       = 2   # N2
MK_ONSET          = 3   # N3
MK_OFFSET         = 4   # N4
MK_TRUTH          = 5   # N5
MK_LIE            = 6   # N6
MK_ITI            = 7   # N7

MK_EXP_START      = 1
MK_BASELINE_START = 2
MK_BASELINE_END   = 3
MK_ACQ_START      = 4   # Block 1 acquaintance start
MK_ACQ_END        = 5   # Block 1 acquaintance end

# New markers for two‑video phase
MK_COOP_INSTRUCTION_START  = 200
MK_COOP_INSTRUCTION_END    = 201
MK_PROTECT_INSTRUCTION_START = 202
MK_PROTECT_INSTRUCTION_END   = 203
MK_VIDEO_A_START = 204
MK_VIDEO_A_END   = 205
MK_VIDEO_B_START = 206
MK_VIDEO_B_END   = 207
MK_BLOCK_A_START = 208
MK_BLOCK_A_END   = 209
MK_BLOCK_B_START = 210
MK_BLOCK_B_END   = 211
MK_EXP_END        = 999


# ══════════════════════════════════════════════════════════════════════════════
# ── MARKER SENDER ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_psychopy_outlet = None

def send_all_markers(code, label=''):
    t = pylsl.local_clock()
    _psychopy_outlet.push_sample([str(code)], timestamp=t)
    send_gsr_marker(label or str(code))
    send_voice_marker(code, label=label)
    if _eye_tracker is not None:
        _eye_tracker.send_marker(code, label)  
    print(f"  [MARKER] {code:>4}  {label}")


# ══════════════════════════════════════════════════════════════════════════════
# ── SENSOR CHECKLIST (pre-session) ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_sensor_check():
    """
    Check all sensors before the session starts.
    PC2 local: Microphone, Camera, GSR/Arduino, Lab Recorder RCS
    PC1 network: EEG stream, Polar HR stream

    Prints a checklist. Blocks (raises SystemExit) if any sensor fails.
    Returns dict of check results for use later (e.g. gsr_active).
    """
    import serial

    print("\n" + "=" * 60)
    print("  PRE-SESSION SENSOR CHECK")
    print("=" * 60)

    results = {}
    failures = []

    # ── Helper ────────────────────────────────────────────────
    def report(name, ok, msg):
        symbol = "OK  " if ok else "FAIL"
        print(f"  [{symbol}] {name:<28} {msg}")
        results[name] = ok
        if not ok:
            failures.append(f"{name}: {msg}")

    # ─────────────────────────────────────────────────────────
    # PC2 LOCAL SENSORS
    # ─────────────────────────────────────────────────────────
    print("\n  -- PC2 Local Sensors --")

    # 1. Microphone
    mic_ok, mic_msg, _ = check_microphone()
    report("Microphone (Boya M1)", mic_ok, mic_msg)

    # 2. Camera (eye tracker)
    try:
        import cv2
        cap = cv2.VideoCapture(CFG['camera_id'])
        cam_ok = cap.isOpened()
        if cam_ok:
            ret, _ = cap.read()
            cam_ok = ret
            cap.release()
        cam_msg = (f"Camera {CFG['camera_id']} OK"
                   if cam_ok else f"Camera {CFG['camera_id']} NOT accessible")
        report("Camera (Eye Tracker)", cam_ok, cam_msg)
    except Exception as e:
        report("Camera (Eye Tracker)", False, str(e))

    # 3. GSR / Arduino
    gsr_ok = False
    try:
        port = CFG['gsr_port']
        s = serial.Serial(port, CFG['gsr_baud'], timeout=0.5)
        s.close()
        gsr_ok = True
        report("GSR Arduino", True, f"Port {port} accessible")
    except Exception as e:
        report("GSR Arduino", False, f"Port {CFG['gsr_port']}: {e}")

    # 4. Lab Recorder RCS — WARN only, never blocks experiment
    lr_ok, lr_msg = check_lab_recorder(
        CFG['lab_recorder_host'], CFG['lab_recorder_port'])
    symbol = "OK  " if lr_ok else "WARN"
    print(f"  [{symbol}] {'Lab Recorder RCS':<28} {lr_msg}")
    results['Lab Recorder RCS'] = lr_ok
    warnings_list_lr = [] if lr_ok else [f"Lab Recorder RCS: {lr_msg}"]
    
    warnings_list = []
    
    # ─────────────────────────────────────────────────────────
    # PC1 NETWORK LSL STREAMS  (WARN only — never blocks)
    # ─────────────────────────────────────────────────────────
    print("\n  -- PC1 Network LSL Streams (warn only, 8s search) --")

    def report_warn(name, ok, msg):
        symbol = "OK  " if ok else "WARN"
        print(f"  [{symbol}] {name:<28} {msg}")
        results[name] = ok
        if not ok:
            warnings_list.append(f"{name}: {msg}")

    try:
        all_network_streams = pylsl.resolve_streams(wait_time=8.0)
        network_stream_names = {
            s.name().strip().lower() for s in all_network_streams
        }
    except Exception:
        network_stream_names = set()

    def has_lsl_stream(stream_name):
        target = stream_name.strip().lower()
        return target in network_stream_names

    # 5. EEG (ANT Neuro)
    eeg_ok = has_lsl_stream(CFG['eeg_stream_name'])
    report_warn(
        "EEG (ANT Neuro)", eeg_ok,
        f"Stream '{CFG['eeg_stream_name']}' "
        + ("found on network" if eeg_ok else "not seen by Python — verify in Lab Recorder"))

    # 6. Polar H10 streams (warn only, consistent with the PC1 EEG check)
    for polar_name in CFG['polar_stream_names']:
        polar_ok = has_lsl_stream(polar_name)
        report_warn(
            f"Polar ({polar_name})", polar_ok,
            f"Stream '{polar_name}' "
            + ("found on network" if polar_ok
               else "not seen by Python — verify Polar streamer/Lab Recorder"))

    # ─────────────────────────────────────────────────────────
    # RESULT
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)

    all_warnings = warnings_list_lr + warnings_list
    if all_warnings:
        print("  NOTE — warnings (experiment will still run):")
        for w in all_warnings:
            print(f"    ! {w}")
        if not lr_ok:
            print("  ! XDF recording will be SKIPPED — open Lab Recorder and")
            print("    enable RCS before next session to record XDF.")
        print()

    if failures:
        print("  SENSOR CHECK FAILED — fix these before starting:\n")
        for f in failures:
            print(f"    ✗ {f}")
        print("=" * 60)
        print("\nExperiment BLOCKED. Fix the above and restart.\n")
        sys.exit(1)
    else:
        print("  PC2 SENSORS OK — ready to start.")
        print("=" * 60 + "\n")

    results['gsr_active']    = gsr_ok
    results['lr_active']     = lr_ok
    return results


# ══════════════════════════════════════════════════════════════════════════════
# ── SETUP FUNCTIONS ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def collect_subject_info():
    exp_info = {
        'Subject ID':   '001',
        'Full Name':    '',
        'Age':          '',
        'Gender':       ['Male', 'Female', 'Other'],
        'Session':      '001',
        'Experimenter': '',
        'Sequence':     ['1', '2', '3', '4'],  # <-- NEW
        'Notes':        '',
    }

    dlg = gui.Dlg(title='Deception Detection — Participant Info')
    dlg.addField('Subject ID:',   exp_info['Subject ID'])
    dlg.addField('Full Name:',    exp_info['Full Name'])
    dlg.addField('Age:',          exp_info['Age'])
    dlg.addField('Gender:',       choices=exp_info['Gender'])
    dlg.addField('Session:',      exp_info['Session'])
    dlg.addField('Experimenter:', exp_info['Experimenter'])
    dlg.addField('Sequence:',     choices=exp_info['Sequence'])  # <-- NEW
    dlg.addField('Notes:',        exp_info['Notes'])
    dlg.show()

    if not dlg.OK:
        print("Experiment cancelled.")
        core.quit()

    exp_info['Subject ID']   = dlg.data[0]
    exp_info['Full Name']    = dlg.data[1]
    exp_info['Age']          = dlg.data[2]
    exp_info['Gender']       = dlg.data[3]
    exp_info['Session']      = dlg.data[4]
    exp_info['Experimenter'] = dlg.data[5]
    exp_info['Sequence']     = dlg.data[6]   # <-- NEW
    exp_info['Notes']        = dlg.data[7]
    exp_info['Date']         = data.getDateStr()
    return exp_info


def create_folders(root, subject_id, name, session):
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"Sub{subject_id}_{name.replace(' ', '_')}_S{session}_{ts}"
    subject_folder = os.path.join(root, folder_name)
    for sub in ['Audio', 'Behavioral', 'EyeTracking', 'Physiological', 'Logs']:
        os.makedirs(os.path.join(subject_folder, sub), exist_ok=True)
    print(f"[Setup] Data folder: {subject_folder}")
    return subject_folder


def setup_lsl():
    global _psychopy_outlet
    info = pylsl.StreamInfo(
        name=CFG['lsl_stream_name'],
        type='Markers',
        channel_count=1,
        nominal_srate=0,
        channel_format='string',
        source_id=CFG['lsl_source_id']
    )
    _psychopy_outlet = pylsl.StreamOutlet(info)
    print("[Setup] Main LSL marker outlet ready.")


def save_participant_info(subject_folder, exp_info, ts):
    path = os.path.join(subject_folder, 'Logs', f"participant_info_{ts}.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("PARTICIPANT INFORMATION\n")
        f.write("=" * 60 + "\n")
        for k, v in exp_info.items():
            f.write(f"{k}: {v}\n")
        f.write("=" * 60 + "\n")
        f.write("\nPROTOCOL STRUCTURE:\n")
        f.write("  Block 0  — Resting Baseline (2 min)\n")
        f.write("  Block 1  — Acquaintance Test (60s continuous voice)\n")
        f.write("  —        — Two‑Video Phase (sequence‑dependent)\n")
        f.write("               Round 1: Instruction (Cooperate/Protect) → Video → 10 questions\n")
        f.write("               Round 2: Instruction (opposite) → Video → 10 questions\n")
        f.write("\nMARKER SCHEME:\n")
        f.write("  MK_EXP_START      =   1\n")
        f.write("  MK_BASELINE_START =   2\n")
        f.write("  MK_BASELINE_END   =   3\n")
        f.write("  MK_ACQ_START      =   4\n")
        f.write("  MK_ACQ_END        =   5\n")
        f.write("  MK_COOP_INSTR_START  = 200\n")
        f.write("  MK_COOP_INSTR_END    = 201\n")
        f.write("  MK_PROTECT_INSTR_START = 202\n")
        f.write("  MK_PROTECT_INSTR_END   = 203\n")
        f.write("  MK_VIDEO_A_START  = 204\n")
        f.write("  MK_VIDEO_A_END    = 205\n")
        f.write("  MK_VIDEO_B_START  = 206\n")
        f.write("  MK_VIDEO_B_END    = 207\n")
        f.write("  MK_BLOCK_A_START  = 208\n")
        f.write("  MK_BLOCK_A_END    = 209\n")
        f.write("  MK_BLOCK_B_START  = 210\n")
        f.write("  MK_BLOCK_B_END    = 211\n")
        f.write("  MK_EXP_END        = 999\n")
        f.write("  Per trial N (N = trial ID):\n")
        f.write("    fixation  = 300 + N*10 + 1\n")
        f.write("    stimulus  = 300 + N*10 + 2\n")
        f.write("    onset     = 300 + N*10 + 3\n")
        f.write("    offset    = 300 + N*10 + 4\n")
        f.write("    truth     = 300 + N*10 + 5\n")
        f.write("    lie       = 300 + N*10 + 6\n")
        f.write("    iti       = 300 + N*10 + 7\n")
    print(f"[Setup] Participant info saved: {path}")
    return path


def validate_stimuli():
    """Check image files for trials that have images (image != None)."""
    print("\n[Stimuli] Checking image files...")
    missing = []
    found   = []
    # Combine both sets (they share the same image names)
    for trial in TRIALS_VIDEO_A + TRIALS_VIDEO_B:
        if trial['image'] is None:
            found.append(f"  --  Trial {trial['id']:>2}: (no image — text only)")
            continue
        path = find_image(CFG['image_folder'], trial['image'])
        if path:
            found.append(f"  OK  Trial {trial['id']:>2}: {os.path.basename(path)}")
        else:
            missing.append(
                f"  !!  Trial {trial['id']:>2}: {trial['image']} NOT FOUND "
                f"in {CFG['image_folder']}")
    for m in found:
        print(m)
    for m in missing:
        print(m)
    if missing:
        print(f"\n  {len(missing)} image(s) missing — will show text-only for those trials.")
    else:
        print(f"  All image checks passed.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ── EXPERIMENT PHASES ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def make_arabic_stim(win, text, height, pos, wrap=1.6, color='white'):
    return visual.TextStim(
        win, text=ar(text), font="Arial",
        height=height, color=color,
        wrapWidth=wrap, pos=pos,
        alignText="center", anchorHoriz="center",
    )


def show_instructions(win, exp_info):
    """
    Welcome screen shown before baseline.
    Experimenter presses any key to start baseline.
    """
    full_text = (
        f"مرحباً، {exp_info['Full Name']}!\n\n"
        "في هذه التجربة ستشاهد مقطعي فيديو،\n"
        "وبعد كل مقطع سيُطلب منك الإجابة عن بعض الأسئلة بصوت عالٍ.\n\n"
        "نرجو منك الحفاظ على ثبات جسمك ورأسك قدر الإمكان أثناء الجلوس.\n\n"
        "سنبدأ الآن بفترة راحة قصيرة.\n"
        "انظر إلى علامة (+) في وسط الشاشة وحاول الاسترخاء تماماً.\n\n"
        "[المجرب: اضغط أي مفتاح للبدء]"
    )
    visual.TextStim(
        win, text=ar(full_text), font="Arial",
        height=0.050, color='white',
        wrapWidth=1.6, pos=(0, 0),
        alignText="center", anchorHoriz="center",
    ).draw()
    win.flip()
    event.waitKeys()


def show_post_baseline(win):
    """After baseline, before acquaintance test."""
    full_text = (
        "انتهت فترة الراحة.\n\n"
        "سنبدأ الآن بسؤال تعارف عام.\n"
        "رجاءً أجب بصوت عالٍ وواضح.\n\n"
        "[المجرب: اضغط أي مفتاح للبدء]"
    )
    visual.TextStim(
        win, text=ar(full_text), font="Arial",
        height=0.055, color='white',
        wrapWidth=1.6, pos=(0, 0),
        alignText="center", anchorHoriz="center",
    ).draw()
    win.flip()
    event.waitKeys()


def show_acquaintance_transition(win):
    """After acquaintance test, before video phase."""
    full_text = (
        "شكراً لك.\n\n"
        "ستشاهد الآن الفيديوهات.\n"
        "سيتم عرض تعليمات تحدد دورك بدقة قبل كل فيديو.\n"
        "يرجى قراءتها بتركيز والانتباه جيداً للمطلوب منك.\n\n"
        "[المجرب: اضغط أي مفتاح للمتابعة]"
    )
    visual.TextStim(
        win, text=ar(full_text), font="Arial",
        height=0.055, color='white',
        wrapWidth=1.6, pos=(0, 0),
        alignText="center", anchorHoriz="center",
    ).draw()
    win.flip()
    event.waitKeys()


# ── Video playback ───────────────────────────────────────────────────────────

def play_movie(win, video_path, start_marker, end_marker):
    """
    Play a video fullscreen using PsychoPy MovieStim.
    Sends start_marker at beginning, end_marker automatically when video finishes.
    ESC aborts the experiment.
    Returns True if completed, False if aborted.
    """
    if not os.path.isfile(video_path):
        print(f"[Video] ERROR: Video not found: {video_path}")
        print("[Video] Skipping this video.")
        return True   # skip but don't abort

    print(f"[Video] Loading: {os.path.basename(video_path)}")

    try:
        movie = visual.MovieStim(
            win,
            filename   = video_path,
            size       = CFG['screen_res'],
            pos        = (0, 0),
            units      = 'pix',
            loop       = False,
            autoStart  = False,
        )
    except Exception as e:
        print(f"[Video] Could not load movie: {e}")
        print("[Video] Skipping this video.")
        return True

    win.flip()
    core.wait(0.5)

    send_all_markers(start_marker, f'video_start_{os.path.basename(video_path)}')
    movie.play()

    print("[Video] Playing... (ESC to abort)")

    video_clock = core.Clock()
    try:
        duration = movie.duration
        if not duration or duration <= 0:
            duration = 9999
        print(f"[Video] Duration: {duration:.1f}s")
    except Exception:
        duration = 9999

    while True:
        movie.draw()
        win.flip()

        if event.getKeys(['escape']):
            movie.stop()
            send_all_markers(end_marker, 'video_end_ABORTED')
            return False

        if movie.status == visual.FINISHED:
            break

        if video_clock.getTime() >= duration:
            break

    movie.stop()
    send_all_markers(end_marker, 'video_end')   # auto‑sent
    print("[Video] Playback complete.")

    win.flip()
    core.wait(0.5)
    return True


# ── Instruction + Video + Questions block ────────────────────────────────────

def run_instruction_video_questions(win, label_text, display_text, video_path, questions,
                                    behavioral_log, eye_tracker,
                                    instr_start_marker, instr_end_marker,
                                    video_start_marker, video_end_marker,
                                    block_start_marker, block_end_marker,
                                    block_num, image_stim, question_stim, fixation_stim):
    """
    Show instruction screen, wait for SPACE to start video,
    play video (auto‑ends), wait for SPACE to start question block,
    then run trials.
    Returns 'ok' or 'abort'.
    """
    # ── 1. Instruction ──────────────────────────────────────────────────────
    instr_stim = visual.TextStim(
        win, text=ar(display_text),
        height=0.065, color='white', pos=(0, 0), wrapWidth=1.6,
        alignText="center", anchorHoriz="center"
    )
    instr_stim.draw()
    win.flip()
    send_all_markers(instr_start_marker, f'{label_text}_instruction_start')

    keys = event.waitKeys(keyList=['space', 'escape'])
    if 'escape' in keys:
        return 'abort'
    send_all_markers(instr_end_marker, f'{label_text}_instruction_end')

    # ── 2. Video ────────────────────────────────────────────────────────────
    ok = play_movie(win, video_path, video_start_marker, video_end_marker)
    if not ok:
        return 'abort'

    # ── 3. Wait for SPACE to start questions ──────────────────────────────
    prompt_stim = visual.TextStim(
        win,
        text=ar("اضغط SPACE لبدء الأسئلة"),
        height=0.07, color='white', pos=(0, 0),
        alignText="center", anchorHoriz="center"
    )
    prompt_stim.draw()
    win.flip()

    keys = event.waitKeys(keyList=['space', 'escape'])
    if 'escape' in keys:
        return 'abort'

    # ── 4. Run questions ────────────────────────────────────────────────────
    send_all_markers(block_start_marker, f'block_{block_num}_start')
    ok = run_trial_block(
        win, image_stim, question_stim, fixation_stim,
        questions,
        block_num=block_num,
        block_label=f"Block {block_num} Questions",
        behavioral_log=behavioral_log,
        eye_tracker=eye_tracker,
    )
    if not ok:
        return 'abort'
    send_all_markers(block_end_marker, f'block_{block_num}_end')

    return 'ok'


def run_video_experiment_phase(win, exp_info, behavioral_log, eye_tracker,
                               image_stim, question_stim, fixation_stim):
    """
    Sequence‑dependent two‑video phase.
    Returns True if completed, False if aborted.
    """
    seq = int(exp_info.get('Sequence', '1'))

    # Mapping: (round1_instruction, round1_video, round1_questions,
    #           round2_instruction, round2_video, round2_questions)
    seq_map = {
        1: (
            "Cooperate",
            "مهمتك الآن هي التعاون مع المحقق.\n\nشاهد الفيديو القادم بدقة، وأجب عن الأسئلة بصدق تام.\nحاول أن تتذكر التفاصيل وتنقلها كما حدثت بالضبط.\n\nتذكر: هدفك هو مساعدة المحقق للوصول للحقيقة.",
            CFG["video_a"], TRIALS_VIDEO_A,
            "Protect",
            "مهمتك الآن هي حماية الشخص الذي قام بالسرقة.\n\nيجب عليك أن تكذب وتخفي المعلومات لتبرئته تماماً.\nلك مطلق الحرية في تأليف القصة أو الإجابات التي تراها مناسبة للدفاع عنه.\n\nتذكر: هدفك الأساسي هو إقناع المحقق ببراءة هذا الشخص وعدم كشف ما فعله.",
            CFG["video_b"], TRIALS_VIDEO_B
        ),
        2: (
            "Cooperate",
            "مهمتك الآن هي التعاون مع المحقق.\n\nشاهد الفيديو القادم بدقة، وأجب عن الأسئلة بصدق تام.\nحاول أن تتذكر التفاصيل وتنقلها كما حدثت بالضبط.\n\nتذكر: هدفك هو مساعدة المحقق للوصول للحقيقة.",
            CFG["video_b"], TRIALS_VIDEO_B,
            "Protect",
            "مهمتك الآن هي حماية الشخص الذي قام بالسرقة.\n\nيجب عليك أن تكذب وتخفي المعلومات لتبرئته تماماً.\nلك مطلق الحرية في تأليف القصة أو الإجابات التي تراها مناسبة للدفاع عنه.\n\nتذكر: هدفك الأساسي هو إقناع المحقق ببراءة هذا الشخص وعدم كشف ما فعله.",
            CFG["video_a"], TRIALS_VIDEO_A
        ),
        3: (
            "Protect",
            "مهمتك الآن هي حماية الشخص الذي قام بالسرقة.\n\nيجب عليك أن تكذب وتخفي المعلومات لتبرئته تماماً.\nلك مطلق الحرية في تأليف القصة أو الإجابات التي تراها مناسبة للدفاع عنه.\n\nتذكر: هدفك الأساسي هو إقناع المحقق ببراءة هذا الشخص وعدم كشف ما فعله.",
            CFG["video_a"], TRIALS_VIDEO_A,
            "Cooperate",
            "مهمتك الآن هي التعاون مع المحقق.\n\nشاهد الفيديو القادم بدقة، وأجب عن الأسئلة بصدق تام.\nحاول أن تتذكر التفاصيل وتنقلها كما حدثت بالضبط.\n\nتذكر: هدفك هو مساعدة المحقق للوصول للحقيقة.",
            CFG["video_b"], TRIALS_VIDEO_B
        ),
        4: (
            "Protect",
            "مهمتك الآن هي حماية الشخص الذي قام بالسرقة.\n\nيجب عليك أن تكذب وتخفي المعلومات لتبرئته تماماً.\nلك مطلق الحرية في تأليف القصة أو الإجابات التي تراها مناسبة للدفاع عنه.\n\nتذكر: هدفك الأساسي هو إقناع المحقق ببراءة هذا الشخص وعدم كشف ما فعله.",
            CFG["video_b"], TRIALS_VIDEO_B,
            "Cooperate",
            "مهمتك الآن هي التعاون مع المحقق.\n\nشاهد الفيديو القادم بدقة، وأجب عن الأسئلة بصدق تام.\nحاول أن تتذكر التفاصيل وتنقلها كما حدثت بالضبط.\n\nتذكر: هدفك هو مساعدة المحقق للوصول للحقيقة.",
            CFG["video_a"], TRIALS_VIDEO_A
        ),
    }

    r1_label, r1_display , r1_vid, r1_qs, r2_label, r2_display , r2_vid, r2_qs = seq_map.get(seq, seq_map[1])

    # Choose correct marker sets based on instruction / video
    def instr_markers(instr_text):
        if instr_text == 'Cooperate':
            return MK_COOP_INSTRUCTION_START, MK_COOP_INSTRUCTION_END
        else:
            return MK_PROTECT_INSTRUCTION_START, MK_PROTECT_INSTRUCTION_END

    def video_markers(vid_path):
        if vid_path == CFG['video_a']:
            return MK_VIDEO_A_START, MK_VIDEO_A_END
        else:
            return MK_VIDEO_B_START, MK_VIDEO_B_END

    # ── Round 1 ──────────────────────────────────────────────────────────────
    instr_s, instr_e = instr_markers(r1_label)
    vid_s, vid_e = video_markers(r1_vid)
    res = run_instruction_video_questions(
        win, r1_label, r1_display, r1_vid, r1_qs,
        behavioral_log, eye_tracker,
        instr_s, instr_e,
        vid_s, vid_e,
        MK_BLOCK_A_START, MK_BLOCK_A_END,
        block_num=2,
        image_stim=image_stim, question_stim=question_stim, fixation_stim=fixation_stim
    )
    if res == 'abort':
        return False

    # ── INTER-BLOCK BASELINE ─────────────────────────────────────────────────
    # Show transition screen before the second baseline
    transition_text = (
        "انتهى الجزء الأول بنجاح.\n\n"
        "سنأخذ الآن فترة راحة قصيرة قبل عرض الفيديو الثاني.\n"
        "انظر إلى علامة (+) في وسط الشاشة وحاول الاسترخاء تماماً.\n\n"
        "[المجرب: اضغط أي مفتاح لبدء الراحة]"
    )
    visual.TextStim(
        win, text=ar(transition_text), font="Arial",
        height=0.065, color='white',
        wrapWidth=1.6, pos=(0, 0),
        alignText="center", anchorHoriz="center",
    ).draw()
    win.flip()

    keys = event.waitKeys(keyList=['space', 'escape', 'return'])
    if 'escape' in keys:
        return False

    # Run the baseline (using the exact same duration as Block 0)
    ok = run_baseline(win, CFG['baseline_dur'])
    if not ok:
        return False

    # ── Round 2 ──────────────────────────────────────────────────────────────
    instr_s, instr_e = instr_markers(r2_label)
    vid_s, vid_e = video_markers(r2_vid)
    # Block B uses block start/end 210/211
    res = run_instruction_video_questions(
        win, r2_label,r2_display, r2_vid, r2_qs,
        behavioral_log, eye_tracker,
        instr_s, instr_e,
        vid_s, vid_e,
        MK_BLOCK_B_START, MK_BLOCK_B_END,
        block_num=3,
        image_stim=image_stim, question_stim=question_stim, fixation_stim=fixation_stim
    )
    return res != 'abort'


# ── BASELINE ──────────────────────────────────────────────────────────────────

def run_baseline(win, dur_seconds):
    """
    Show fixation cross for baseline period.
    Sends BASELINE_START and BASELINE_END markers.
    Returns False if aborted, True if completed.
    """
    print(f"\n[Baseline] Starting {dur_seconds}s resting baseline...")

    fixation = visual.TextStim(
        win, text='+', height=0.10, color='white', pos=(0, 0))

    send_all_markers(MK_BASELINE_START, 'baseline_start')
    timer = core.CountdownTimer(dur_seconds)

    last_print = dur_seconds
    while timer.getTime() > 0:
        fixation.draw()
        win.flip()

        remaining = int(timer.getTime())
        if remaining != last_print and remaining % 30 == 0:
            print(f"  [Baseline] {remaining}s remaining...")
            last_print = remaining

        if event.getKeys(['escape']):
            send_all_markers(MK_BASELINE_END, 'baseline_end_ABORTED')
            return False

    send_all_markers(MK_BASELINE_END, 'baseline_end')
    print("[Baseline] Complete.")
    return True


# ── ACQUAINTANCE TEST (Block 1) ───────────────────────────────────────────────

def run_acquaintance_block(win, behavioral_log):
    """
    Block 1 — Acquaintance Test.
    Single slide showing all sub-questions at once.
    Experimenter presses SPACE to start the 60s continuous voice recording,
    then SPACE again when participant finishes (or timer expires).
    ESC aborts.

    No truth/lie label for this block — it's always baseline/truthful speech.
    Returns 'ok' or 'abort'.
    """
    print("\n[Block 1] Acquaintance Test — single slide, continuous voice")

    send_all_markers(MK_ACQ_START, 'acquaintance_start')

    # Build display text
    question_text = ACQUAINTANCE_QUESTION
    question_stim = visual.TextStim(
        win, text=ar(question_text), font="Arial",
        height=0.060, color='white',
        pos=(0, 0.05), wrapWidth=1.5,
        alignText="center", anchorHoriz="center",
    )

    # ── Two separate prompts — one Arabic, one Latin-only ─────────────────────
    prompt_ar = visual.TextStim(
        win,
        text=ar('[SPACE = بدء التسجيل  |  ESC = إيقاف]'),
        font="Arial", pos=(0, -0.42), height=0.032, color=(0.6, 0.6, 0.6),
        alignText="center", anchorHoriz="center",
    )
    prompt_rec = visual.TextStim(
        win,
        text='[SPACE = stop  |  ESC = abort]',
        font="Arial", pos=(0, -0.42), height=0.032, color=(0.6, 0.6, 0.6),
        alignText="center", anchorHoriz="center",
    )

    # Phase 0: wait for SPACE (onset) — no time limit, experimenter controls
    t_onset  = None
    t_offset = None
    phase    = 0

    while True:
        question_stim.draw()
        if phase == 0:
            prompt_ar.draw()
        else:
            prompt_rec.draw()
        win.flip()

        keys = event.getKeys(['space', 'escape'])

        if 'escape' in keys:
            send_all_markers(MK_ACQ_END, 'acquaintance_end_ABORTED')
            return 'abort'

        if 'space' in keys:
            if phase == 0:
                # First SPACE: onset
                t_onset = pylsl.local_clock()
                send_all_markers(MK_ACQ_START + 1, 'acquaintance_onset')
                phase = 1
                print("  [Block 1] Recording started — experimenter controls duration")

            elif phase == 1:
                # Second SPACE: offset
                t_offset = pylsl.local_clock()
                break

    send_all_markers(MK_ACQ_END, 'acquaintance_end')
    dur = (t_offset - t_onset) if (t_onset and t_offset) else 0
    print(f"  [Block 1] Acquaintance recording duration: {dur:.2f}s")

    # Log entry (truth_lie = 'truth' by design — acquaintance is always truthful)
    entry = {
        'block':           1,
        'trial_id':        'ACQ',
        'image':           '',
        'image_path':      '',
        'question':        question_text,
        't_stimulus_on':   f"{t_onset:.6f}" if t_onset else '',
        't_onset':         f"{t_onset:.6f}" if t_onset else '',
        't_offset':        f"{t_offset:.6f}" if t_offset else '',
        'reaction_time_s': '0.0000',
        'speech_dur_s':    f"{dur:.4f}",
        'truth_lie':       'truth',
        'status':          'ok',
    }
    behavioral_log.append(entry)
    return 'ok'


# ── FIXATION ──────────────────────────────────────────────────────────────────

def run_fixation(win, fixation_stim, trial_id, dur):
    fixation_stim.draw()
    win.flip()
    send_all_markers(mk(trial_id, MK_FIXATION), f'trial_{trial_id}_fixation')
    timer = core.CountdownTimer(dur)
    while timer.getTime() > 0:
        if event.getKeys(['escape']):
            return False
    return True


# ── STIMULUS + RESPONSE (no timeout) ─────────────────────────────────────────

def run_stimulus_and_response(win, image_stim, question_stim, trial,
                               behavioral_log, block_num, eye_tracker=None):
    """
    Show question (and image if trial['image'] is not None).
    Waits indefinitely for experimenter input:
      SPACE (1st) → response ONSET marker
      SPACE (2nd) → response OFFSET marker → clears screen
      ESC         → abort experiment

    Returns: ('ok', entry) | ('abort', None)
    """
    trial_id = trial['id']

    # ── Image loading (only if trial has an image) ────────────────────────────
    image_path   = None
    image_loaded = False

    if trial.get('image'):
        image_path = find_image(CFG['image_folder'], trial['image'])
        if image_path:
            try:
                image_stim.image = image_path
                image_loaded = True
                print(f"  Image loaded: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"  Could not load image {image_path}: {e}")
        else:
            print(f"  Image not found for trial {trial_id}: '{trial['image']}'")
    else:
        print(f"  Trial {trial_id}: text-only (no image)")

    # ── Question position: centered if no image, lower if image present ───────
    if image_loaded:
        question_stim.pos = (0, -0.35)
    else:
        question_stim.pos = (0, 0.0)

    question_stim.text = ar(trial['question'])

    # Initial draw
    if image_loaded:
        image_stim.draw()
    question_stim.draw()
    win.flip()

    send_all_markers(mk(trial_id, MK_STIMULUS), f'trial_{trial_id}_stimulus_on')
    t_stimulus_on = pylsl.local_clock()

    response_phase = 0
    t_onset        = None
    t_offset       = None

    # Two pre-built Latin-only prompts — never mutate .text mid-loop
    prompt_wait = visual.TextStim(
        win, text='[SPACE = onset  |  ESC = abort]',
        pos=(0, -0.45), height=0.03, color=(0.6, 0.6, 0.6)
    )
    prompt_rec = visual.TextStim(
        win, text='[SPACE = offset when participant finishes  |  ESC = abort]',
        pos=(0, -0.45), height=0.03, color=(0.6, 0.6, 0.6)
    )

    # ── Wait loop — no timeout ─────────────────────────────────────────────────
    while True:
        if image_loaded:
            image_stim.draw()
        question_stim.draw()
        if response_phase == 0:
            prompt_wait.draw()
        else:
            prompt_rec.draw()
        win.flip()

        keys = event.getKeys(['space', 'escape'])

        if 'escape' in keys:
            image_stim.setAutoDraw(False)
            question_stim.setAutoDraw(False)
            return 'abort', None

        if 'space' in keys:
            if response_phase == 0:
                t_onset = pylsl.local_clock()
                send_all_markers(mk(trial_id, MK_ONSET),
                                 f'trial_{trial_id}_response_onset')
                response_phase = 1

            elif response_phase == 1:
                t_offset = pylsl.local_clock()
                send_all_markers(mk(trial_id, MK_OFFSET),
                                 f'trial_{trial_id}_response_offset')
                break

    image_stim.setAutoDraw(False)
    question_stim.setAutoDraw(False)

    rt  = t_onset  - t_stimulus_on
    dur = t_offset - t_onset
    print(f"  Trial {trial_id}: RT={rt:.3f}s  speech_dur={dur:.3f}s")

    entry = {
        'block':           block_num,
        'trial_id':        trial_id,
        'image':           trial['image'] or '',
        'image_path':      image_path or 'NO_IMAGE',
        'question':        trial['question'],
        't_stimulus_on':   f"{t_stimulus_on:.6f}",
        't_onset':         f"{t_onset:.6f}",
        't_offset':        f"{t_offset:.6f}",
        'reaction_time_s': f"{rt:.4f}",
        'speech_dur_s':    f"{dur:.4f}",
        'truth_lie':       '',       # filled by run_iti()
        'status':          'ok',
    }
    behavioral_log.append(entry)
    return 'ok', entry


# ── ITI + SILENT TRUTH/LIE CAPTURE ───────────────────────────────────────────

def run_iti(win, trial_id, dur, behavioral_entry=None):
    """
    Blank screen ITI. Sends ITI marker immediately.
    Experimenter presses T (truth) or L (lie) silently during the blank.
    No annotation shown on screen. Marker fires the instant key is pressed.
    ESC aborts. T/L logged to behavioral_entry['truth_lie'] and CSV.
    """
    win.flip()   # blank screen
    send_all_markers(mk(trial_id, MK_ITI), f'trial_{trial_id}_iti')

    timer = core.CountdownTimer(dur)
    while timer.getTime() > 0:
        keys = event.getKeys(['t', 'l', 'escape'])

        if 'escape' in keys:
            return False

        if 't' in keys and behavioral_entry is not None:
            send_all_markers(mk(trial_id, MK_TRUTH), f'trial_{trial_id}_TRUTH')
            behavioral_entry['truth_lie'] = 'truth'
            print(f"  [Trial {trial_id}] Labeled: TRUTH (silent, during ITI)")

        if 'l' in keys and behavioral_entry is not None:
            send_all_markers(mk(trial_id, MK_LIE), f'trial_{trial_id}_LIE')
            behavioral_entry['truth_lie'] = 'lie'
            print(f"  [Trial {trial_id}] Labeled: LIE (silent, during ITI)")

    return True


# ══════════════════════════════════════════════════════════════════════════════
# ── TRIAL BLOCK RUNNER ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_trial_block(win, image_stim, question_stim, fixation_stim,
                    trials, block_num, block_label, behavioral_log, eye_tracker=None):
    """
    Generic runner for a list of trials (Block 2 or Block 3).
    Returns True if completed, False if aborted.
    """
    print(f"\n{'=' * 60}")
    print(f"  {block_label} — {len(trials)} trials")
    print("=" * 60)
    print("  SPACE (1st) = Response onset")
    print("  SPACE (2nd) = Response offset")
    print("  T / L       = Truth/Lie label (silent, during ITI)")
    print("  ESC         = Abort\n")

    for trial in trials:
        trial_id = trial['id']
        print(f"\n-- {block_label} Trial {trial_id}: "
              f"{'[img] ' if trial.get('image') else '[text]'} "
              f"{trial['image'] or '(no image)'}")

        # Fixation
        ok = run_fixation(win, fixation_stim, trial_id, CFG['fixation_dur'])
        if not ok:
            return False

        # Stimulus + Response
        result, entry = run_stimulus_and_response(
            win, image_stim, question_stim, trial,
            behavioral_log, block_num, eye_tracker)
        if result == 'abort':
            return False

        # ITI with silent T/L labeling
        ok = run_iti(win, trial_id, CFG['iti_dur'], behavioral_entry=entry)
        if not ok:
            return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
# ── MAIN ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main():

    # ── 1. Sensor check ────────────────────────────────────────────────────────
    sensor_results = run_sensor_check()
    gsr_active     = sensor_results.get('gsr_active', False)
    lr_active      = sensor_results.get('lr_active',  False)
    lr_recording_started = False
    lr_reselect_timer = None

    # ── 2. Collect participant info ────────────────────────────────────────────
    exp_info   = collect_subject_info()
    subject_id = exp_info['Subject ID']
    session    = exp_info['Session']
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 60)
    print("  DECEPTION DETECTION EXPERIMENT — HCM Lab")
    print("=" * 60)
    print(f"  Subject : {exp_info['Full Name']} (ID: {subject_id})")
    print(f"  Session : {session}")
    print(f"  Sequence: {exp_info['Sequence']}")
    print("=" * 60 + "\n")

    # ── 3. Create folder structure ─────────────────────────────────────────────
    os.makedirs(CFG['data_root'], exist_ok=True)
    subject_folder = create_folders(
        CFG['data_root'], subject_id,
        exp_info['Full Name'], session)
    audio_folder  = os.path.join(subject_folder, 'Audio')
    physio_folder = os.path.join(subject_folder, 'Physiological')
    behav_folder  = os.path.join(subject_folder, 'Behavioral')

    save_participant_info(subject_folder, exp_info, ts)
    validate_stimuli()

    # ── 4. LSL main outlet ─────────────────────────────────────────────────────
    setup_lsl()
    print("[Setup] PsychoPy_Markers LSL outlet created.")

    # ── 5. Start background sensors ────────────────────────────────────────────
    print("\n[Setup] Starting sensors...")

    if gsr_active:
        start_gsr(port=CFG['gsr_port'], baud=CFG['gsr_baud'],
                  subject_id=subject_id, output_folder=physio_folder)

    start_voice(subject_id=subject_id, output_folder=audio_folder)

    eye_tracker = EEG_EyeTracker(
        subject_id         = subject_id,
        subject_name       = exp_info['Full Name'],
        session            = session,
        experiment_name    = 'DeceptionDetection',
        eyetracking_folder = os.path.join(subject_folder, 'EyeTracking'),
        camera_id          = CFG['camera_id'],
        show_preview       = True,
        save_video         = False,
        record_raw_video   = True,
        screen_width       = CFG['screen_res'][0],
        screen_height      = CFG['screen_res'][1],
    )
    global _eye_tracker
    _eye_tracker = eye_tracker
    eye_tracker.initialize_camera()
    # Start tracking immediately so video recording begins now
    eye_tracker.start_tracking()
    
    print("[Setup] All sensors started (including eye tracker video). Stabilizing 3s...")
    time.sleep(3)

    # GSR creates its LSL outlet only after serial stabilization plus the
    # five-second Arduino rate measurement. Wait for that outlet before the
    # first Lab Recorder stream selection, so the normal session does not
    # begin with a missing GSR stream.
    if gsr_active:
        print("[Setup] Waiting for GSR rate measurement before Lab Recorder...")
        gsr_deadline = time.monotonic() + 10.0
        while (get_measured_rate() is None and
               time.monotonic() < gsr_deadline):
            time.sleep(0.1)
        if get_measured_rate() is None:
            print("[Setup] WARNING: GSR LSL stream is not ready yet; "
                  "late-stream reselection will try again.")
        else:
            print(f"[Setup] GSR LSL ready ({get_measured_rate():.1f} Hz).")

    # ── 6. Lab Recorder ────────────────────────────────────────────────────────
    # Lab Recorder path/name is intentionally configured manually in the
    # Lab Recorder UI. RCS is used only for refresh/select/start/stop.
    lr = LabRecorderController(
        host = CFG['lab_recorder_host'],
        port = CFG['lab_recorder_port'],
    )
    if lr_active:
        if lr.connect():
            print("[LabRecorder] Refreshing stream list (pass 1)...")
            lr.update()
            time.sleep(0.5)
            lr.select_all()
            print("[LabRecorder] Waiting 2s for late-joining streams...")
            time.sleep(2.0)
            print("[LabRecorder] Refreshing stream list (pass 2)...")
            lr.update()
            time.sleep(0.5)
            lr.select_all()
            time.sleep(0.3)
            lr.start()
            lr_recording_started = True
            print("[Setup] Lab Recorder recording STARTED — XDF path is manual.")

            def _reselect():
                print("[LabRecorder] Scheduled re-selection (6s)...")
                lr.reselect_all()
            lr_reselect_timer = threading.Timer(6.0, _reselect)
            lr_reselect_timer.daemon = True
            lr_reselect_timer.start()
        else:
            print("[Setup] Lab Recorder could not be connected — XDF recording skipped.")
    else:
        print("[Setup] Lab Recorder not reachable — XDF recording skipped.")

    # ── IMPORTANT REMINDER ─────────────────────────────────────────────────
    print("\n" + "!" * 60)
    print("  !! REMINDER: If you made ANY code changes, you MUST:")
    print("     1. Re-select ALL streams in Lab Recorder manually")
    print("     2. Verify all sensors are running properly")
    print("     3. THEN start Lab Recorder recording")
    print("  !! New/changed streams won't appear automatically!")
    print("!" * 60 + "\n")

    time.sleep(1)

    # ── 7. PsychoPy window ────────────────────────────────────────────────────
    win = visual.Window(
        size     = CFG['screen_res'],
        fullscr  = CFG['fullscreen'],
        screen   = CFG['screen_id'],
        color    = (-1, -1, -1),
        colorSpace = 'rgb',
        units    = 'height',
        winType  = 'pyglet',
        allowGUI = False,
    )

    # ── Shared stimuli objects ────────────────────────────────────────────────
    fixation_stim = visual.TextStim(
        win, text='+', height=0.08, color='white', pos=(0, 0))

    image_stim = visual.ImageStim(
        win, pos=(0, 0.1), size=(0.55, 0.55), units='height')

    question_stim = visual.TextStim(
        win, text='', font="Arial",
        height=0.065, color='white',
        pos=(0, -0.35), wrapWidth=1.5,
        alignText="center", anchorHoriz="center",
    )

    behavioral_log = []
    aborted        = False
    experiment_started = False

    # ─────────────────────────────────────────────────────────────────────────
    # ── PHASE 1: Welcome screen ───────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    show_instructions(win, exp_info)

    # ── Experiment START marker ───────────────────────────────────────────────
    send_all_markers(MK_EXP_START, 'experiment_start')
    experiment_started = True

    # ─────────────────────────────────────────────────────────────────────────
    # ── PHASE 2: Block 0 — Resting Baseline ──────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    if not aborted:
        print(f"\n[Block 0] Resting Baseline — {CFG['baseline_dur']}s")
        ok = run_baseline(win, CFG['baseline_dur'])
        if not ok:
            aborted = True

    # ─────────────────────────────────────────────────────────────────────────
    # ── PHASE 3: Block 1 — Acquaintance Test ─────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    if not aborted:
        show_post_baseline(win)
        result = run_acquaintance_block(win, behavioral_log)
        if result == 'abort':
            aborted = True

    # ─────────────────────────────────────────────────────────────────────────
    # ── PHASE 4: Two‑Video Phase (replaces old crime video + Block 2/3) ─────
    # ─────────────────────────────────────────────────────────────────────────
    if not aborted:
        show_acquaintance_transition(win)
        aborted = not run_video_experiment_phase(
            win, exp_info, behavioral_log, eye_tracker,
            image_stim, question_stim, fixation_stim
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ── CLEANUP ───────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────

    if experiment_started:
        send_all_markers(MK_EXP_END, 'experiment_end')

    # Stop Lab Recorder
    if lr_reselect_timer is not None:
        lr_reselect_timer.cancel()
    if lr_recording_started and lr.is_connected():
        time.sleep(1.0)
        lr.stop()
        lr.disconnect()
        print("[Cleanup] XDF recording stopped.")

    # Stop sensors
    print("\n[Cleanup] Stopping sensors...")
    eye_tracker.close()
    stop_voice()
    if gsr_active:
        stop_gsr()

    
    # Save behavioral log
    if behavioral_log:
        behav_file = os.path.join(
            behav_folder, f"behavioral_{subject_id}_{ts}.csv")
        with open(behav_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=behavioral_log[0].keys())
            writer.writeheader()
            writer.writerows(behavioral_log)
        print(f"[Save] Behavioral log: {behav_file}")

    # Completion screen
    n_ok   = sum(1 for r in behavioral_log if r['status'] == 'ok')
    n_true = sum(1 for r in behavioral_log if r.get('truth_lie') == 'truth')
    n_lie  = sum(1 for r in behavioral_log if r.get('truth_lie') == 'lie')
    total  = len(TRIALS_VIDEO_A) + len(TRIALS_VIDEO_B)

    completion_text = (
        "تم إيقاف التجربة!\n\n" if aborted else "اكتملت التجربة!\n\n"
    )
    completion_text += (
        f"المحاولات المكتملة: {n_ok} / {total}\n\n"
        "تم حفظ جميع البيانات.\n\n"
        "اضغط أي مفتاح للخروج."
    )

    visual.TextStim(
        win, text=ar(completion_text), font="Arial",
        height=0.055,
        color='green' if not aborted else 'yellow',
        wrapWidth=1.5, pos=(0, 0),
        alignText="center", anchorHoriz="center",
    ).draw()
    win.flip()
    event.waitKeys()

    # Final summary
    print("\n" + "=" * 60)
    print("  ALL DATA SAVED")
    print("=" * 60)
    print(f"  Subject folder : {subject_folder}")
    print(f"  Trials OK      : {n_ok} / {total}")
    print(f"  Truth labels   : {n_true}")
    print(f"  Lie labels     : {n_lie}")
    print("  XDF (Lab Rec.) : manually configured/saved in Lab Recorder"
          if lr_recording_started else
          "  XDF (Lab Rec.) : not recorded")
    print("=" * 60)

    win.close()
    core.quit()


if __name__ == '__main__':
    main()
