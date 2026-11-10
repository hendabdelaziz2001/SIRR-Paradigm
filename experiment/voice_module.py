"""
Voice acquisition module for the multimodal deception-detection protocol.
HCM Lab – Ain Shams University

Purpose
-------
Checks the selected/default microphone, captures mono speech at 16 kHz, saves
the local microphone signal as a mono PCM-16 WAV, and publishes synchronized
LSL streams:
  VoiceAudio    : float32, one channel, 16,000 Hz
  VoiceMarkers  : string, irregular event markers

The local WAV is the preferred input for offline speech analysis. Marker
timing is stored in VoiceMarkers_<subject>_<timestamp>.csv, including the
microphone sample position and LSL clock. The LSL audio stream remains active
for Lab Recorder/XDF synchronization and is not replaced by the local file.

start_voice() starts immediately after the pre-session microphone check.
The module uses the configured/default input device; for reproducible data
collection the Boya microphone should be selected as the Windows default or
assigned an explicit device ID.

Dependencies: sounddevice, soundfile, pylsl, numpy
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import csv
import threading
import os
from datetime import datetime
from pylsl import StreamInfo, StreamOutlet, local_clock


# ── Configuration ─────────────────────────────────────────────────────────────

SAMPLE_RATE  = 16000      # 16 kHz — sufficient for speech, compact XDF
BLOCK_SIZE   = 512        # Frames per callback (~32 ms at 16 kHz)
CHANNELS     = 1          # Mono mic
DEVICE_ID    = None       # None = system default mic


# ── Module-level state ────────────────────────────────────────────────────────

_audio_frames      = []
_sample_counter    = [0]
_marker_log        = []

_stream            = None
_lsl_marker_outlet = None
_lsl_audio_outlet  = None
_recording         = False
_lock              = threading.Lock()

_wav_filename      = None
_csv_filename      = None
_start_lsl_t       = None


# ── Audio Callback ────────────────────────────────────────────────────────────

def _audio_callback(indata, frames, time_info, status):
    """Capture microphone audio and push the same samples to LSL."""
    if status:
        print(f"[Voice] Audio status: {status}")

    block = indata.copy()

    with _lock:
        _audio_frames.append(block)
        _sample_counter[0] += frames

    # Push raw audio to LSL — Lab Recorder saves this into the XDF
    if _lsl_audio_outlet is not None:
        try:
            _lsl_audio_outlet.push_chunk(block.tolist())
        except Exception:
            pass


# ── Microphone Check ──────────────────────────────────────────────────────────

def check_microphone(device=DEVICE_ID, test_seconds=1.0):
    """
    Auto-detect and test the default microphone.

    Returns
    -------
    ok         : bool
    message    : str   — human-readable result
    device_info: dict or None
    """
    import time

    # 1. Query devices
    try:
        devices       = sd.query_devices()
        default_input = sd.default.device[0]
    except Exception as e:
        return False, f"sounddevice error: {e}", None

    if default_input is None or default_input < 0:
        return False, "No default input device found on this PC.", None

    dev_info = devices[default_input] if default_input < len(devices) else {}
    dev_name = dev_info.get('name', 'Unknown')

    # 2. Try opening the stream briefly
    try:
        test_buf = []

        def _test_cb(indata, frames, t, status):
            test_buf.append(indata.copy())

        s = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32',
            callback=_test_cb,
            blocksize=BLOCK_SIZE,
            device=device
        )
        s.start()
        time.sleep(test_seconds)
        s.stop()
        s.close()
    except Exception as e:
        return False, f"Cannot open mic '{dev_name}': {e}", None

    # 3. Check signal is not completely silent
    if not test_buf:
        return False, f"Mic '{dev_name}' opened but captured no data.", None

    audio = np.concatenate(test_buf, axis=0)
    rms   = float(np.sqrt(np.mean(audio ** 2)))
    peak  = float(np.max(np.abs(audio)))

    # Threshold 0.000005 — handles very quiet mics like Boya M1 at rest.
    # Pure silence (disconnected/muted) gives RMS < 0.000001.
    if rms < 0.000005:
        msg = (f"Mic '{dev_name}' connected but SILENT "
               f"(RMS={rms:.8f}). Check cable/mute/Windows privacy settings.")
        return False, msg, dev_info

    msg = (f"'{dev_name}' | {SAMPLE_RATE} Hz | "
           f"RMS={rms:.6f} | Peak={peak:.4f}")
    print(f"[Voice] Mic OK — {msg}")
    return True, msg, dev_info


# ── LSL Setup ─────────────────────────────────────────────────────────────────

def _create_lsl_streams(subject_id):
    global _lsl_marker_outlet, _lsl_audio_outlet

    # Marker stream — event-driven, irregular rate
    mrk_info = StreamInfo(
        name='VoiceMarkers',
        type='Markers',
        channel_count=1,
        nominal_srate=0,
        channel_format='string',
        source_id=f'Voice_HCMLab_{subject_id}'
    )
    mrk_info.desc().append_child_value("subject_id", subject_id)
    _lsl_marker_outlet = StreamOutlet(mrk_info)
    print("[Voice] LSL outlet ready: 'VoiceMarkers'")

    # Audio stream — continuous 16 kHz, recorded into XDF by Lab Recorder
    aud_info = StreamInfo(
        name='VoiceAudio',
        type='Audio',
        channel_count=1,
        nominal_srate=SAMPLE_RATE,
        channel_format='float32',
        source_id=f'VoiceAudio_HCMLab_{subject_id}'
    )
    ch = aud_info.desc().append_child('channels').append_child('channel')
    ch.append_child_value('label', 'Microphone')
    ch.append_child_value('unit',  'normalized')
    ch.append_child_value('type',  'Audio')
    aud_info.desc().append_child_value("subject_id",  subject_id)
    aud_info.desc().append_child_value("sample_rate", str(SAMPLE_RATE))
    aud_info.desc().append_child_value("mic_model",   "Boya_M1")
    _lsl_audio_outlet = StreamOutlet(
        aud_info, chunk_size=BLOCK_SIZE, max_buffered=30)
    print(f"[Voice] LSL outlet ready: 'VoiceAudio' ({SAMPLE_RATE} Hz → XDF)")


# ── Public API ────────────────────────────────────────────────────────────────

def start_voice(subject_id='unknown', output_folder='.', device=DEVICE_ID):
    """
    Start microphone recording.
    Streams audio to LSL (VoiceAudio) AND saves WAV locally.

    Parameters
    ----------
    subject_id    : str          — used for output filenames
    output_folder : str          — where to save WAV and CSV
    device        : int or None  — sounddevice device index (None = default)
    """
    global _stream, _recording, _wav_filename, _csv_filename, _start_lsl_t

    os.makedirs(output_folder, exist_ok=True)

    _audio_frames.clear()
    _marker_log.clear()
    _sample_counter[0] = 0

    _create_lsl_streams(subject_id)

    ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
    _wav_filename = os.path.join(
        output_folder, f"Voice_{subject_id}_{ts}.wav")
    _csv_filename = os.path.join(
        output_folder, f"VoiceMarkers_{subject_id}_{ts}.csv")

    # Set timing/state before opening the stream so the first callback cannot
    # arrive with an unset start time.
    _start_lsl_t = local_clock()
    _recording   = True

    try:
        _stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32',
            callback=_audio_callback,
            blocksize=BLOCK_SIZE,
            device=device
        )
        _stream.start()
    except Exception:
        _recording = False
        _stream = None
        raise

    print(f"[Voice] Recording started @ {SAMPLE_RATE} Hz → {_wav_filename}")


def stop_voice():
    """
    Stop recording.
    Saves mono WAV microphone audio and a CSV marker log.
    """
    global _stream, _recording
    global _lsl_audio_outlet, _lsl_marker_outlet

    if not _recording:
        print("[Voice] Not recording — nothing to stop.")
        return

    _recording = False
    if _stream:
        _stream.stop()
        _stream.close()
        _stream = None

    # ── Save mono microphone WAV ────────────────────────────────
    if _audio_frames:
        with _lock:
            audio_data  = np.concatenate(_audio_frames,  axis=0)

        # Keep the analysis input one-dimensional/mono. Marker timing is
        # already available in VoiceMarkers.csv and must not be mixed into
        # the microphone waveform.
        sf.write(_wav_filename, audio_data, SAMPLE_RATE, subtype='PCM_16')
        duration = len(audio_data) / SAMPLE_RATE
        print(f"[Voice] Mono WAV saved: {_wav_filename}  ({duration:.1f}s)")
    else:
        print("[Voice] No audio data captured.")

    # ── Save marker CSV ────────────────────────────────────────
    marker_fields = [
        'wall_time', 'lsl_clock', 'elapsed_s', 'sample_pos',
        'time_in_audio', 'code', 'label'
    ]
    with open(_csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=marker_fields)
        writer.writeheader()
        writer.writerows(_marker_log)
    print(f"[Voice] Markers saved: {_csv_filename} ({len(_marker_log)} events)")

    # Releasing the outlet references makes the stopped streams disappear
    # cleanly before a possible next session.
    _lsl_audio_outlet = None
    _lsl_marker_outlet = None

    print("[Voice] Recording stopped.")


def send_voice_marker(code, label=''):
    """
    Send a marker via LSL and record its microphone sample position.

    Parameters
    ----------
    code  : int — numeric marker code (1-999). Amplitude = code/1000
    label : str — human-readable label for CSV log
    """
    if not _recording:
        print(f"[Voice] Warning: send_voice_marker({code}) called before start_voice()")
        return

    lsl_t = local_clock()

    if _lsl_marker_outlet:
        _lsl_marker_outlet.push_sample([str(code)], timestamp=lsl_t)

    with _lock:
        sample_pos = _sample_counter[0]

    entry = {
        'wall_time':     datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        'lsl_clock':     f"{lsl_t:.6f}",
        'elapsed_s':     f"{(lsl_t - _start_lsl_t):.3f}" if _start_lsl_t else '',
        'sample_pos':    sample_pos,
        'time_in_audio': f"{sample_pos / SAMPLE_RATE:.4f}",
        'code':          code,
        'label':         label
    }
    _marker_log.append(entry)

    print(f"[Voice] Marker {code:>4} ({label}) | LSL={lsl_t:.4f} | "
          f"audio={sample_pos / SAMPLE_RATE:.3f}s")


def is_recording():
    """Returns True if currently recording."""
    return _recording


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time

    print("=" * 55)
    print("Voice Module — Mic Check + LSL Stream Test")
    print("=" * 55)

    ok, msg, info = check_microphone()
    print(f"  {'OK' if ok else 'FAIL'} — {msg}")
    if not ok:
        print("  Fix microphone and retry.")
        exit(1)

    print("\nRecording 10 seconds (WAV + LSL audio stream)...")
    start_voice(subject_id='TEST', output_folder='.')
    time.sleep(1)

    send_voice_marker(1,   label='experiment_start')
    time.sleep(2)
    send_voice_marker(10,  label='fixation_on')
    time.sleep(1)
    send_voice_marker(20,  label='stimulus_on')
    time.sleep(3)
    send_voice_marker(30,  label='response_onset')
    time.sleep(2)
    send_voice_marker(40,  label='response_offset')
    time.sleep(1)
    send_voice_marker(999, label='experiment_end')

    stop_voice()
    print("Done.")
