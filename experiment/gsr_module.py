"""
GSR acquisition module for the multimodal deception-detection protocol.
HCM Lab – Ain Shams University

Purpose
-------
Reads the Grove GSR Sensor V2.0 through Arduino analog pin A0 and USB
serial, then records the raw 10-bit ADC value to CSV and publishes two LSL
streams:
  GSR_EDA       : one raw ADC channel at the measured Arduino rate
  GSR_Markers   : event-driven string markers

The module deliberately stores raw ADC values. It does not convert them to
skin resistance or conductance; those transformations belong in offline
analysis after the circuit/sensor configuration is finalized.

Startup/timing
--------------
start_gsr() returns immediately and starts a background worker. The worker
waits 2 s for serial stabilization, measures the Arduino rate for 5 s, and
only then creates the LSL stream and CSV. Consequently GSR may appear about
7 s after the other local sensors. Queued markers are delivered when the GSR
stream becomes available.

Current hardware defaults: COM3 at 115200 baud. The standalone test accepts
another port as its first command-line argument.

Dependencies: pyserial, pylsl
"""

import serial
import threading
import csv
import queue
import time
from datetime import datetime
from pylsl import StreamInfo, StreamOutlet, local_clock


# ── Configuration ─────────────────────────────────────────────────────────────

GSR_PORT        = 'COM3'
GSR_BAUD        = 115200
GSR_LSL_NAME    = 'GSR_EDA'
GSR_SOURCE_ID   = 'Arduino_GSR_HCMLab'
RATE_MEASURE_S  = 5      # seconds to measure actual sample rate at startup
FLUSH_EVERY     = 20     # flush CSV every N samples (~1s at 20Hz)
RATE_REPORT_S   = 10     # report live rate to console every N seconds


# ── Module-level state ────────────────────────────────────────────────────────

_thread         = None
_stop_event     = threading.Event()
_data_queue     = queue.Queue()
_marker_queue   = queue.Queue()

_csv_file       = None
_csv_writer     = None
_csv_filename   = None
_gsr_outlet     = None
_mrk_outlet     = None
_sample_count   = 0
_start_lsl_t    = None
_measured_rate  = None   # actual Hz measured at startup


# ── LSL Setup ─────────────────────────────────────────────────────────────────

def _create_lsl_streams(measured_hz):
    """
    Create GSR_EDA and GSR_Markers LSL outlets.
    nominal_srate is set from the measured Arduino rate (not hardcoded).
    """
    global _gsr_outlet, _mrk_outlet

    # Round to nearest integer Hz for nominal_srate
    # (LSL nominal_srate must be a number, 0 = irregular)
    nominal = round(measured_hz) if measured_hz and measured_hz > 1 else 0

    gsr_info = StreamInfo(
        name           = GSR_LSL_NAME,
        type           = 'EDA',
        channel_count  = 1,
        nominal_srate  = nominal,
        channel_format = 'float32',
        source_id      = GSR_SOURCE_ID
    )
    ch = gsr_info.desc().append_child('channels').append_child('channel')
    ch.append_child_value('label',         'GSR_Raw')
    ch.append_child_value('unit',          'ADC_0-1023')
    ch.append_child_value('type',          'EDA')
    ch.append_child_value('sensor',        'Grove_GSR_V2.0')
    ch.append_child_value('measured_hz',   f"{measured_hz:.1f}" if measured_hz else 'unknown')
    ch.append_child_value('note',
        'Raw 10-bit ADC from Arduino analogRead(A0). '
        'Sensor measures RESISTANCE not conductance. '
        'Higher value = lower skin conductance.')
    _gsr_outlet = StreamOutlet(gsr_info)

    mrk_info = StreamInfo(
        name           = 'GSR_Markers',
        type           = 'Markers',
        channel_count  = 1,
        nominal_srate  = 0,
        channel_format = 'string',
        source_id      = GSR_SOURCE_ID + '_MRK'
    )
    _mrk_outlet = StreamOutlet(mrk_info)

    print(f"[GSR] LSL streams ready: '{GSR_LSL_NAME}' "
          f"(srate={nominal} Hz), 'GSR_Markers'")


# ── CSV Setup ─────────────────────────────────────────────────────────────────

def _open_csv(subject_id, output_folder):
    global _csv_file, _csv_writer, _csv_filename

    ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
    _csv_filename = f"{output_folder}/GSR_{subject_id}_{ts}.csv"
    _csv_file     = open(_csv_filename, 'w', newline='')
    _csv_writer   = csv.writer(_csv_file)
    _csv_writer.writerow([
        'wall_time', 'lsl_clock', 'elapsed_s',
        'gsr_raw', 'marker'
    ])
    print(f"[GSR] CSV → {_csv_filename}")


# ── Rate Measurement ──────────────────────────────────────────────────────────

def _measure_rate(port, duration_s=RATE_MEASURE_S):
    """
    Read from port for `duration_s` seconds and count valid samples.
    Returns measured Hz. Does NOT store data — just counting.
    Port must already be open and flushed.
    """
    print(f"[GSR] Measuring Arduino sample rate ({duration_s}s)...")
    count      = 0
    t_start    = time.time()
    t_end      = t_start + duration_s

    while time.time() < t_end:
        try:
            raw = port.readline().decode('utf-8').strip()
            if raw.lstrip('-').replace('.', '', 1).isdigit():
                count += 1
        except (UnicodeDecodeError, ValueError):
            pass

    elapsed = time.time() - t_start
    hz      = count / elapsed if elapsed > 0 else 0
    print(f"[GSR] Measured rate: {count} samples in {elapsed:.1f}s "
          f"= {hz:.1f} Hz")
    return hz


# ── Background Thread ─────────────────────────────────────────────────────────

def _gsr_thread(port_name, baud, subject_id, output_folder):
    global _sample_count, _start_lsl_t, _measured_rate

    # ── Open serial port ──────────────────────────────────────────────────────
    try:
        port = serial.Serial(port_name, baud, timeout=1)
    except serial.SerialException as e:
        print(f"[GSR] Cannot open {port_name}: {e}")
        return

    time.sleep(2)        # Let Arduino boot / stabilize
    port.reset_input_buffer()

    # ── FIX 1: Measure actual sample rate before creating LSL stream ──────────
    measured_hz    = _measure_rate(port, RATE_MEASURE_S)
    _measured_rate = measured_hz
    port.reset_input_buffer()   # clear any backlog from measurement period

    # ── Create LSL streams with measured rate ─────────────────────────────────
    _create_lsl_streams(measured_hz)
    _open_csv(subject_id, output_folder)

    _start_lsl_t  = local_clock()
    _sample_count = 0
    flush_counter = 0
    last_rate_t   = time.time()
    rate_count    = 0

    print(f"[GSR] Recording started on {port_name} @ {baud} baud "
          f"({measured_hz:.1f} Hz)")

    # ── FIX 2: Blocking readline loop (no in_waiting spinning) ───────────────
    while not _stop_event.is_set():

        # ── Drain marker queue first ──────────────────────────────────────────
        while not _marker_queue.empty():
            try:
                label   = _marker_queue.get_nowait()
                t       = local_clock()
                elapsed = t - _start_lsl_t
                _mrk_outlet.push_sample([label], t)
                _csv_writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    f"{t:.6f}", f"{elapsed:.3f}",
                    '', f"MARKER:{label}"
                ])
                _csv_file.flush()
                print(f"[GSR] Marker: {label}")
            except queue.Empty:
                break

        # ── Read one line (blocks up to timeout=1s) ───────────────────────────
        try:
            raw = port.readline().decode('utf-8').strip()
        except (UnicodeDecodeError, serial.SerialException):
            continue

        if not raw:
            continue   # timeout with no data

        if not raw.lstrip('-').replace('.', '', 1).isdigit():
            continue   # skip non-numeric / partial lines

        gsr_value = float(raw)
        t         = local_clock()
        elapsed   = t - _start_lsl_t
        #Voltage = (gsr_value*5)/1023
        #R_skin = ((1024+2*gsr_value)*10000)/(512-gsr_value)
        #C_skin = 1/R_skin

        # ── Push to LSL ───────────────────────────────────────────────────────
        _gsr_outlet.push_sample([gsr_value], t)

        # ── Write to CSV ──────────────────────────────────────────────────────
        _csv_writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            f"{t:.6f}", f"{elapsed:.3f}",
            f"{gsr_value:.1f}", ''
        ])

        # ── FIX 3: Flush every FLUSH_EVERY samples ────────────────────────────
        flush_counter += 1
        if flush_counter >= FLUSH_EVERY:
            _csv_file.flush()
            flush_counter = 0

        # ── Update counters ───────────────────────────────────────────────────
        _sample_count += 1
        rate_count    += 1
        _data_queue.put({'gsr': gsr_value, 'elapsed': elapsed, 'lsl_t': t})

        # ── FIX 4: Report live rate every RATE_REPORT_S seconds ───────────────
        now = time.time()
        if now - last_rate_t >= RATE_REPORT_S:
            live_hz     = rate_count / (now - last_rate_t)
            last_rate_t = now
            rate_count  = 0
            print(f"[GSR] Live: GSR={gsr_value:.0f}  "
                  f"rate={live_hz:.1f}Hz  "
                  f"elapsed={elapsed:.0f}s  "
                  f"samples={_sample_count}"
            
                  )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    port.close()
    if _csv_file:
        _csv_file.flush()
        _csv_file.close()
    print(f"[GSR] Stopped. {_sample_count} samples @ "
          f"{_measured_rate:.1f}Hz → {_csv_filename}")


# ── Public API ────────────────────────────────────────────────────────────────

def start_gsr(port=GSR_PORT, baud=GSR_BAUD,
              subject_id='unknown', output_folder='.'):
    """
    Start GSR recording in a background thread.

    Measures actual Arduino sample rate for the first 5 seconds,
    then creates the LSL stream with the correct nominal_srate.

    Parameters
    ----------
    port          : str  — Serial port (e.g. 'COM3')
    baud          : int  — Baud rate (default 9600)
    subject_id    : str  — Used for CSV filename
    output_folder : str  — Where to save the CSV
    """
    global _thread, _sample_count, _start_lsl_t, _measured_rate
    global _gsr_outlet, _mrk_outlet, _csv_file, _csv_writer, _csv_filename

    # ── FIX 5: Full state reset so start_gsr() is safe to call again ─────────
    _stop_event.clear()
    _sample_count  = 0
    _start_lsl_t   = None
    _measured_rate = None
    _gsr_outlet    = None
    _mrk_outlet    = None
    _csv_file      = None
    _csv_writer    = None
    _csv_filename  = None

    # Drain any leftover items from previous session
    while not _data_queue.empty():
        try:
            _data_queue.get_nowait()
        except queue.Empty:
            break
    while not _marker_queue.empty():
        try:
            _marker_queue.get_nowait()
        except queue.Empty:
            break

    _thread = threading.Thread(
        target=_gsr_thread,
        args=(port, baud, subject_id, output_folder),
        daemon=True
    )
    _thread.start()
    print(f"[GSR] Background thread started (port={port}, baud={baud}).")
    print(f"[GSR] Measuring Arduino rate for {RATE_MEASURE_S}s — please wait...")


def stop_gsr():
    """Stop GSR recording and close resources."""
    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=5)
    print("[GSR] Stopped.")


def send_gsr_marker(label: str):
    """
    Send a string marker synchronized on the LSL clock.
    Thread-safe — safe to call from main experiment loop.

    Parameters
    ----------
    label : str — e.g. 'trial_1_stimulus_on'
    """
    _marker_queue.put(label)


def get_latest_gsr():
    """
    Non-blocking. Returns most recent GSR sample dict or None.

    Returns
    -------
    dict with keys: 'gsr' (ADC 0-1023), 'elapsed' (s), 'lsl_t'
    or None if no new data.
    """
    latest = None
    while not _data_queue.empty():
        try:
            latest = _data_queue.get_nowait()
        except queue.Empty:
            break
    return latest


def get_measured_rate():
    """Returns the measured Arduino sample rate in Hz, or None if not yet measured."""
    return _measured_rate


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    port = sys.argv[1] if len(sys.argv) > 1 else GSR_PORT
    print(f"GSR Module standalone test — port={port}")
    print("Will measure rate for 5s then stream for 15s.\n")

    start_gsr(port=port, subject_id='TEST', output_folder='.')

    # Wait for rate measurement to complete (~7s total: 2s boot + 5s measure)
    time.sleep(8)
    print(f"\nMeasured rate: {get_measured_rate()} Hz\n")

    send_gsr_marker('test_start')

    try:
        for i in range(5):
            time.sleep(3)
            send_gsr_marker(f'test_marker_{i+1}')
            d = get_latest_gsr()
            if d:
                print(f"  GSR={d['gsr']:.0f}  elapsed={d['elapsed']:.2f}s")
            else:
                print("  (no data yet)")
    except KeyboardInterrupt:
        pass

    send_gsr_marker('test_end')
    stop_gsr()
