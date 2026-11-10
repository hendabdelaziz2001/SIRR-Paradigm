"""
Polar H10 — Full LSL Streamer (PC1)
HCM Lab – Ain Shams University | Deception Detection Protocol

Streams ALL data the Polar H10 can produce:
  HR  + RR + HRV  — beat-driven, irregular
  ECG             — 130 Hz, raw electrode signal in µV
  ACC             — 25 Hz, 3-axis accelerometer in mg

CHANGES (2026-05):
  1. ECG stream added  — 130 Hz, per-sample timestamps from hardware clock
  2. ACC stream added  — 25 Hz, 3 channels (X,Y,Z mg)
  3. Subject folder created per session containing 3 CSV files:
       sub-{ID}_{Name}_{ts}/
         PolarH10_{ID}_HR_{ts}.csv
         PolarH10_{ID}_ECG_{ts}.csv
         PolarH10_{ID}_ACC_{ts}.csv
  4. ECG timestamps: hardware nanosecond clock converted to LSL clock via
     one-time offset computed at first packet — per-sample accuracy ±7ms
  5. ACC timestamps: same hardware-clock approach as ECG
  6. HR srate=0 (irregular, already fixed previously)
  7. PolarH10_Markers outlet silent (no conflict with PC2 markers)

LSL Streams published:
  PolarH10_HR      — BPM,     irregular (srate=0), float32
  PolarH10_RR      — ms,      irregular (srate=0), float32
  PolarH10_HRV     — ms RMSSD,irregular (srate=0), float32
  PolarH10_ECG     — µV,      130 Hz   (srate=130),float32
  PolarH10_ACC     — mg XYZ,  25 Hz    (srate=25), float32, 3 channels
  PolarH10_Markers — string,  irregular, silent outlet

USAGE:
  python polar_streamer_pc1.py
  python polar_streamer_pc1.py 001 SubjectName

INSTALL on PC1 (run once):
  pip install polar-python pylsl bleak

PAIRING (before every session):
  1. Windows Settings → Bluetooth & devices
  2. Find "Polar H10 XXXXXXXX" → Remove device
  3. Wet H10 electrodes, put on participant
  4. Add device → Bluetooth → Polar H10 → wait "Connected"
  5. Run this script
"""

import asyncio
import math
import csv
import os
import sys
import signal
from collections import deque
from datetime import datetime

# ── Dependency checks ─────────────────────────────────────────────────────────

try:
    from polar_python import PolarDevice
except ImportError:
    raise ImportError("pip install polar-python")

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
except ImportError:
    raise ImportError("pip install pylsl")

try:
    from bleak import BleakScanner
except ImportError:
    raise ImportError("pip install bleak")


# ══════════════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

DEVICE_ADDRESS  = "24:AC:AC:13:DE:CC"   # ← Your Polar H10 MAC address
RR_BUFFER_SIZE  = 10                     # RR intervals kept for RMSSD

ECG_SAMPLE_RATE = 130                    # Hz — only valid rate for H10
ECG_RESOLUTION  = 14                     # bits — only valid resolution
ACC_SAMPLE_RATE = 25                     # Hz — 25/50/100/200 available
ACC_RESOLUTION  = 16                     # bits
ACC_RANGE       = 2                      # g — 2/4/8 available

STREAM_HR_NAME  = "PolarH10_HR"
STREAM_RR_NAME  = "PolarH10_RR"
STREAM_HRV_NAME = "PolarH10_HRV"
STREAM_ECG_NAME = "PolarH10_ECG"
STREAM_ACC_NAME = "PolarH10_ACC"
STREAM_MRK_NAME = "PolarH10_Markers"
SOURCE_ID       = "PolarH10_HCMLab"

# Base output folder — subject subfolder created inside
BASE_OUTPUT     = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "polar_data")


# ══════════════════════════════════════════════════════════════════════════════
# ── HRV CALCULATION ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def compute_rmssd(rr_buffer):
    if len(rr_buffer) < 2:
        return None
    rr    = list(rr_buffer)
    diffs = [rr[i+1] - rr[i] for i in range(len(rr) - 1)]
    return math.sqrt(sum(d**2 for d in diffs) / len(diffs))


# ══════════════════════════════════════════════════════════════════════════════
# ── LSL STREAMS ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def create_lsl_streams(subject_id):
    """
    Create all 6 LSL outlets.

    HR / RR / HRV / Markers: srate=0 (irregular — beat-driven)
    ECG:                      srate=130 (fixed hardware clock)
    ACC:                      srate=25  (fixed hardware clock)
    """

    def _info(name, stype, channels, srate, fmt, sid_suffix):
        info = StreamInfo(name, stype, channels, srate, fmt,
                          SOURCE_ID + sid_suffix)
        info.desc().append_child_value("manufacturer", "Polar")
        info.desc().append_child_value("device",       "H10")
        info.desc().append_child_value("subject_id",   subject_id)
        return info

    # ── HR — irregular ────────────────────────────────────────────────────────
    hr_info = _info(STREAM_HR_NAME,  "HR",      1, 0,   "float32", "_HR")
    hr_ch   = hr_info.desc().append_child("channels").append_child("channel")
    hr_ch.append_child_value("label", "HR_bpm")
    hr_ch.append_child_value("unit",  "bpm")
    hr_ch.append_child_value("note",  "one sample per detected heartbeat")

    # ── RR — irregular ────────────────────────────────────────────────────────
    rr_info = _info(STREAM_RR_NAME,  "RR",      1, 0,   "float32", "_RR")
    rr_ch   = rr_info.desc().append_child("channels").append_child("channel")
    rr_ch.append_child_value("label", "RR_ms")
    rr_ch.append_child_value("unit",  "ms")
    rr_ch.append_child_value("note",  "timestamp back-calculated per beat")

    # ── HRV — irregular ───────────────────────────────────────────────────────
    hrv_info = _info(STREAM_HRV_NAME, "HRV",    1, 0,   "float32", "_HRV")
    hrv_ch   = hrv_info.desc().append_child("channels").append_child("channel")
    hrv_ch.append_child_value("label",      "RMSSD_ms")
    hrv_ch.append_child_value("unit",       "ms")
    hrv_ch.append_child_value("hrv_metric", "RMSSD")
    hrv_ch.append_child_value("window",     str(RR_BUFFER_SIZE) + "_beats")

    # ── ECG — 130 Hz fixed ────────────────────────────────────────────────────
    ecg_info = _info(STREAM_ECG_NAME, "ECG",    1, ECG_SAMPLE_RATE,
                     "float32", "_ECG")
    ecg_ch   = ecg_info.desc().append_child("channels").append_child("channel")
    ecg_ch.append_child_value("label",      "ECG_uV")
    ecg_ch.append_child_value("unit",       "microvolts")
    ecg_ch.append_child_value("resolution", str(ECG_RESOLUTION) + "bit")
    ecg_ch.append_child_value("note",
        "per-sample LSL timestamp via hardware-clock offset")

    # ── ACC — 25 Hz fixed, 3 channels ─────────────────────────────────────────
    acc_info = _info(STREAM_ACC_NAME, "ACC",    3, ACC_SAMPLE_RATE,
                     "float32", "_ACC")
    acc_chs  = acc_info.desc().append_child("channels")
    for label in ["ACC_X_mg", "ACC_Y_mg", "ACC_Z_mg"]:
        ch = acc_chs.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit",  "mg")
    acc_info.desc().append_child_value("range_g", str(ACC_RANGE))

    # ── Markers — silent outlet ───────────────────────────────────────────────
    mrk_info = _info(STREAM_MRK_NAME, "Markers", 1, 0,  "string",  "_MRK")
    mrk_info.desc().append_child_value("note",
        "outlet present for Lab Recorder; no markers pushed from PC1 "
        "to avoid conflict with PC2 PsychoPy_Markers")

    outlets = {
        'hr':  StreamOutlet(hr_info),
        'rr':  StreamOutlet(rr_info),
        'hrv': StreamOutlet(hrv_info),
        'ecg': StreamOutlet(ecg_info, chunk_size=13, max_buffered=360),
        'acc': StreamOutlet(acc_info, chunk_size=5,  max_buffered=60),
        'mrk': StreamOutlet(mrk_info),
    }

    print("[Polar] LSL streams ready:")
    rows = [
        (STREAM_HR_NAME,  "irregular", "1 ch  BPM"),
        (STREAM_RR_NAME,  "irregular", "1 ch  ms"),
        (STREAM_HRV_NAME, "irregular", "1 ch  ms RMSSD"),
        (STREAM_ECG_NAME, "130 Hz   ", "1 ch  µV float32"),
        (STREAM_ACC_NAME, "25 Hz    ", "3 ch  mg float32"),
        (STREAM_MRK_NAME, "irregular", "silent outlet"),
    ]
    for name, rate, desc in rows:
        print(f"         {name:<22} {rate}  {desc}")
    print()
    return outlets


# ══════════════════════════════════════════════════════════════════════════════
# ── SUBJECT FOLDER + CSV FILES ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def create_subject_folder(subject_id, subject_name):
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = subject_name.replace(' ', '_') if subject_name else 'unknown'
    folder    = os.path.join(
        BASE_OUTPUT,
        f"sub-{subject_id}_{safe_name}_{ts}"
    )
    os.makedirs(folder, exist_ok=True)
    print(f"[Polar] Subject folder: {folder}")
    return folder, ts


def open_hr_csv(folder, subject_id, ts):
    path   = os.path.join(folder, f"PolarH10_{subject_id}_HR_{ts}.csv")
    f      = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "wall_time", "elapsed_s", "lsl_clock",
        "hr_bpm", "rr_ms", "rr_lsl_clock", "rmssd_ms"
    ])
    print(f"[Polar] HR  CSV → {os.path.basename(path)}")
    return f, writer


def open_ecg_csv(folder, subject_id, ts):
    path   = os.path.join(folder, f"PolarH10_{subject_id}_ECG_{ts}.csv")
    f      = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "wall_time", "elapsed_s",
        "sensor_ts_ns", "lsl_clock", "ecg_uv"
    ])
    print(f"[Polar] ECG CSV → {os.path.basename(path)}")
    return f, writer


def open_acc_csv(folder, subject_id, ts):
    path   = os.path.join(folder, f"PolarH10_{subject_id}_ACC_{ts}.csv")
    f      = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "wall_time", "elapsed_s",
        "sensor_ts_ns", "lsl_clock",
        "acc_x_mg", "acc_y_mg", "acc_z_mg"
    ])
    print(f"[Polar] ACC CSV → {os.path.basename(path)}")
    return f, writer


# ══════════════════════════════════════════════════════════════════════════════
# ── DEVICE DISCOVERY ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

async def find_polar():
    print("[Polar] Scanning for Polar H10 (10s)...")
    devices = await BleakScanner.discover(timeout=10.0)
    for d in devices:
        if d.name and "Polar" in d.name:
            print(f"[Polar] Found: {d.name} [{d.address}]")
            return d.address
    raise RuntimeError(
        "No Polar device found.\n"
        "  Make sure H10 is on the participant and paired in Windows Bluetooth."
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── MAIN ASYNC STREAM LOOP ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

async def run_stream(address, subject_id, subject_name, stop_event):

    # ── Create subject folder and CSV files ───────────────────────────────────
    folder, ts = create_subject_folder(subject_id, subject_name)
    hr_f,  hr_w  = open_hr_csv( folder, subject_id, ts)
    ecg_f, ecg_w = open_ecg_csv(folder, subject_id, ts)
    acc_f, acc_w = open_acc_csv(folder, subject_id, ts)

    # ── Create LSL streams ────────────────────────────────────────────────────
    outlets = create_lsl_streams(subject_id)

    # ── State ─────────────────────────────────────────────────────────────────
    rr_buffer  = deque(maxlen=RR_BUFFER_SIZE)
    beat_count = 0
    ecg_count  = 0
    acc_count  = 0
    start_time = datetime.now()

    # Hardware-clock → LSL-clock offset (computed on first ECG/ACC packet)
    # Polar sensor timestamps are in nanoseconds on the sensor's own clock.
    # We compute offset = local_clock() - sensor_time_s at first packet,
    # then: lsl_t = sensor_time_s + offset  for every subsequent sample.
    ecg_offset = [None]   # list so closure can write to it
    acc_offset = [None]

    if address is None:
        address = await find_polar()

    print(f"\n[Polar] Connecting to {address} ...")

    try:
        device = PolarDevice(address)

        async with device as polar_dev:
            print("[Polar] Connected!")
            print("[Polar] Starting HR + ECG + ACC streams simultaneously...")
            print("[Polar] Press Ctrl+C when the experiment session ends.\n")

            # ── HR callback ───────────────────────────────────────────────────
            def on_hr(data):
                nonlocal beat_count

                t       = local_clock()
                now     = datetime.now()
                elapsed = (now - start_time).total_seconds()
                hr      = data.heartrate

                # Push HR — one sample per packet, irregular
                outlets['hr'].push_sample([float(hr)], t)

                # Back-calculate per-beat RR timestamps
                rr_list       = list(data.rr_intervals)
                cumulative_ms = 0.0
                rr_timestamps = []
                for rr_ms in reversed(rr_list):
                    cumulative_ms += rr_ms
                    rr_timestamps.insert(0, t - cumulative_ms / 1000.0)

                for rr_ms, rr_t in zip(rr_list, rr_timestamps):
                    rr_buffer.append(rr_ms)
                    outlets['rr'].push_sample([float(rr_ms)], rr_t)
                    beat_count += 1

                    rmssd = compute_rmssd(rr_buffer)
                    if rmssd is not None:
                        outlets['hrv'].push_sample([rmssd], rr_t)

                    hr_w.writerow([
                        now.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        f"{elapsed:.3f}",
                        f"{t:.6f}",
                        hr,
                        f"{rr_ms:.1f}",
                        f"{rr_t:.6f}",
                        f"{rmssd:.2f}" if rmssd else ""
                    ])
                hr_f.flush()

                if beat_count % 5 == 0:
                    rmssd = compute_rmssd(rr_buffer)
                    rmssd_str = f"  RMSSD={rmssd:.1f}ms" if rmssd else ""
                    print(f"  HR={hr}bpm  RR={rr_list[-1]:.0f}ms  "
                          f"beats={beat_count}  "
                          f"ECG={ecg_count}  ACC={acc_count}"
                          f"{rmssd_str}")

            # ── ECG callback ──────────────────────────────────────────────────
            def on_ecg(data):
                nonlocal ecg_count

                now     = datetime.now()
                elapsed = (now - start_time).total_seconds()

                # data.timestamp = sensor hardware time in nanoseconds
                sensor_s = data.timestamp / 1_000_000_000.0

                # Compute one-time offset on first packet
                if ecg_offset[0] is None:
                    ecg_offset[0] = local_clock() - sensor_s
                    print(f"[Polar] ECG clock offset computed: "
                          f"{ecg_offset[0]:.3f}s")

                offset   = ecg_offset[0]

                # ── polar-python API compatibility ────────────────────────────
                # v0.x:  data.samples  → list of int/float µV values
                # v1.x:  data.samples  → list of ECGSample objects  with .voltage
                #        (ECGData is a dataclass; inspect on first packet)
                if hasattr(data, 'samples'):
                    raw = data.samples
                    # v1.x ECGSample objects have a .voltage attribute
                    if raw and hasattr(raw[0], 'voltage'):
                        samples = [s.voltage for s in raw]
                    else:
                        samples = [float(v) for v in raw]  # v0.x plain numbers
                elif hasattr(data, 'ecg_data'):
                    raw = data.ecg_data
                    if raw and hasattr(raw[0], 'voltage'):
                        samples = [s.voltage for s in raw]
                    else:
                        samples = [float(v) for v in raw]
                elif hasattr(data, 'data'):
                    # v1.x streaming JSON shows {"data": [[val], [val], ...]}
                    raw = data.data
                    # data.data may be list-of-lists or list-of-scalars
                    if raw and isinstance(raw[0], (list, tuple)):
                        samples = [float(v[0]) for v in raw]
                    else:
                        samples = [float(v) for v in raw]
                else:
                    # Last resort: print the object so you can see its attributes
                    print(f"[Polar][ECG DEBUG] Unknown ECGData attrs: {vars(data)}")
                    return
                # ─────────────────────────────────────────────────────────────

                n        = len(samples)
                interval = 1.0 / ECG_SAMPLE_RATE  # 7.692ms per sample

                # Back-calculate per-sample timestamp:
                # data.timestamp = time of LAST sample in packet
                # earlier samples: subtract intervals going backwards
                lsl_chunk  = []
                csv_rows   = []
                for i, uv in enumerate(samples):
                    # i=0 is oldest, i=n-1 is newest (=data.timestamp)
                    sample_sensor_s = sensor_s - (n - 1 - i) * interval
                    sample_lsl_t    = sample_sensor_s + offset
                    lsl_chunk.append([float(uv)])
                    csv_rows.append([
                        now.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        f"{elapsed:.6f}",
                        data.timestamp,
                        f"{sample_lsl_t:.6f}",
                        f"{uv:.2f}"
                    ])
                    ecg_count += 1

                # push_chunk with first sample timestamp
                # LSL interpolates the rest at nominal 130 Hz
                first_t = (sensor_s - (n - 1) * interval) + offset
                outlets['ecg'].push_chunk(lsl_chunk, first_t)

                ecg_w.writerows(csv_rows)
                ecg_f.flush()

            # ── ACC callback ──────────────────────────────────────────────────
            def on_acc(data):
                nonlocal acc_count

                now     = datetime.now()
                elapsed = (now - start_time).total_seconds()

                # data.timestamp = sensor hardware time in nanoseconds
                sensor_s = data.timestamp / 1_000_000_000.0

                # Compute one-time offset on first packet
                if acc_offset[0] is None:
                    acc_offset[0] = local_clock() - sensor_s
                    print(f"[Polar] ACC clock offset computed: "
                          f"{acc_offset[0]:.3f}s")

                offset   = acc_offset[0]
                samples  = data.data              # list of [x, y, z] in mg
                n        = len(samples)
                interval = 1.0 / ACC_SAMPLE_RATE  # 40ms per sample at 25 Hz

                lsl_chunk = []
                csv_rows  = []
                for i, (x, y, z) in enumerate(samples):
                    sample_sensor_s = sensor_s - (n - 1 - i) * interval
                    sample_lsl_t    = sample_sensor_s + offset
                    lsl_chunk.append([float(x), float(y), float(z)])
                    csv_rows.append([
                        now.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        f"{elapsed:.6f}",
                        data.timestamp,
                        f"{sample_lsl_t:.6f}",
                        x, y, z
                    ])
                    acc_count += 1

                first_t = (sensor_s - (n - 1) * interval) + offset
                outlets['acc'].push_chunk(lsl_chunk, first_t)

                acc_w.writerows(csv_rows)
                acc_f.flush()

            # ── Start all three streams simultaneously ─────────────────────────
            await polar_dev.start_hr_stream(hr_callback=on_hr)
            print("[Polar] HR  stream started  (irregular, beat-driven)")

            await polar_dev.start_ecg_stream(
                ecg_callback=on_ecg,
                sample_rate=ECG_SAMPLE_RATE,
                resolution=ECG_RESOLUTION
            )
            print(f"[Polar] ECG stream started  ({ECG_SAMPLE_RATE} Hz, "
                  f"{ECG_RESOLUTION}-bit)")

            await polar_dev.start_acc_stream(
                acc_callback=on_acc,
                sample_rate=ACC_SAMPLE_RATE,
                resolution=ACC_RESOLUTION,
                range=ACC_RANGE
            )
            print(f"[Polar] ACC stream started  ({ACC_SAMPLE_RATE} Hz, "
                  f"±{ACC_RANGE}g)\n")

            print("[Polar] All streams active. Press Ctrl+C to stop.\n")
            print(f"  {'Stream':<12}  {'Live data'}")
            print("  " + "-" * 50)

            # ── Wait until stop ────────────────────────────────────────────────
            while not stop_event.is_set():
                await asyncio.sleep(0.2)

            # ── Stop all streams ───────────────────────────────────────────────
            print("\n[Polar] Stopping streams...")
            await polar_dev.stop_hr_stream()
            await polar_dev.stop_ecg_stream()
            await polar_dev.stop_acc_stream()
            print("[Polar] All streams stopped.")

    except Exception as e:
        print(f"\n[Polar] Error: {e}")

        if "Insufficient Authentication" in str(e):
            print("\n[Polar] FIX — Re-pair the H10 in Windows Bluetooth:")
            print("   1. Settings → Bluetooth → Remove 'Polar H10...'")
            print("   2. Wet electrodes, put on participant")
            print("   3. Add device → Bluetooth → Polar H10")
            print("   4. Wait for 'Connected', re-run this script\n")

        elif "not found" in str(e).lower() or "was not found" in str(e):
            print("[Polar] FIX — Check Bluetooth is ON and H10 is paired.")
            print(f"[Polar]       MAC address: {address}")
            print("[Polar]       Run with no args to auto-scan.\n")

    finally:
        # ── Close all CSV files ────────────────────────────────────────────────
        hr_f.close()
        ecg_f.close()
        acc_f.close()

        elapsed_total = (datetime.now() - start_time).total_seconds()
        print(f"\n[Polar] Session complete.")
        print(f"[Polar] Duration : {elapsed_total:.1f}s")
        print(f"[Polar] Beats    : {beat_count}")
        print(f"[Polar] ECG pts  : {ecg_count}  "
              f"(expected ~{int(elapsed_total * ECG_SAMPLE_RATE)})")
        print(f"[Polar] ACC pts  : {acc_count}  "
              f"(expected ~{int(elapsed_total * ACC_SAMPLE_RATE)})")
        print(f"[Polar] Folder   : {folder}")
        print("[Polar] Copy folder to subject's Physiological/ folder on PC2.")


# ══════════════════════════════════════════════════════════════════════════════
# ── ENTRY POINT ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main():
    subject_id   = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    subject_name = sys.argv[2] if len(sys.argv) > 2 else ""
    address      = DEVICE_ADDRESS   # set to None to auto-scan

    print("=" * 60)
    print("  POLAR H10 FULL STREAMER — PC1")
    print("  HCM Lab, Ain Shams University")
    print("=" * 60)
    print(f"  Subject ID   : {subject_id}")
    print(f"  Subject Name : {subject_name or '(not provided)'}")
    print(f"  MAC address  : {address}  (None = auto-scan)")
    print(f"  Output root  : {BASE_OUTPUT}")
    print(f"  Streams      : HR + RR + HRV + ECG ({ECG_SAMPLE_RATE}Hz)"
          f" + ACC ({ACC_SAMPLE_RATE}Hz)")
    print("=" * 60)
    print()
    print("  BEFORE RUNNING:")
    print("  [ ] H10 is on participant (electrodes wet)")
    print("  [ ] H10 paired in Windows Bluetooth on PC1")
    print("  [ ] Lab Recorder is open on PC2")
    print("  [ ] Ethernet connects PC1 <-> PC2")
    print()
    print("  Ctrl+C = stop all streams cleanly")
    print("=" * 60 + "\n")

    # ── FIX: Create loop first, then asyncio.Event ─────────────────────────
    loop       = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop_event = asyncio.Event()

    # ── Clean Ctrl+C — POSIX first, Windows fallback ──────────────────────
    try:
        def _on_sigint():
            print("\n[Polar] Ctrl+C — stopping streams...")
            loop.call_soon_threadsafe(stop_event.set)
        loop.add_signal_handler(signal.SIGINT, _on_sigint)
        use_kbi = False
    except NotImplementedError:
        use_kbi = True   # Windows

    async def _windows_wrapper():
        task = asyncio.ensure_future(
            run_stream(address, subject_id, subject_name, stop_event))
        try:
            await task
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n[Polar] Interrupted — stopping streams...")
            stop_event.set()
            await asyncio.sleep(2.0)   # allow stop_*_stream() to complete

    try:
        if use_kbi:
            loop.run_until_complete(_windows_wrapper())
        else:
            loop.run_until_complete(
                run_stream(address, subject_id, subject_name, stop_event))
    except KeyboardInterrupt:
        print("\n[Polar] Interrupted.")
        stop_event.set()
        loop.run_until_complete(asyncio.sleep(1.5))
    finally:
        try:
            loop.close()
        except Exception:
            pass

    print("\n[Polar] Streamer exited.")


if __name__ == "__main__":
    main()
