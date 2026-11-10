"""
Real-time webcam eye-tracking module for the multimodal protocol.
HCM Lab – Ain Shams University

Purpose
-------
Uses MediaPipe Face Mesh with iris refinement to extract webcam-relative eye
features in real time. It publishes the continuous EyeTracking_Data LSL
stream and the event-driven EyeTrackingMarkers stream, writes frame/event
CSVs, and saves the original camera video for later Py-Feat facial analysis.

EyeTracking_Data v2 channels
----------------------------
0 gaze_relative_x       normalized iris position within the eye
1 gaze_relative_y       normalized iris position within the eye
2 iris_diam_proxy       iris diameter / eye width, not true pupil diameter
3 avg_ear               eye aspect ratio
4 blink                 binary blink state
5 fixation              webcam-relative I-DT fixation estimate
6 face_detected         binary face-detection quality flag
7 gaze_velocity_rel_s   relative gaze velocity per second

The gaze values are relative eye coordinates, not calibrated screen gaze.
The iris diameter is a scale-normalized proxy, not a physical pupil diameter.
Blink detection uses a fixed EAR threshold; there is no participant-facing
open-eye or point-screen calibration step.

Outputs
-------
The raw unmirrored MP4 is intended for offline Py-Feat processing. The raw
video frame-map CSV links saved video frames to capture-frame numbers and
LSL timestamps. The per-frame CSV includes rows even when no face is found.
The Preview window is unmirrored and is for quality control only.

Running this file directly starts a standalone camera/LSL/CSV/video test.
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import queue
from datetime import datetime
import os
import math
import sys
from collections import deque
import threading
import pylsl


# PsychoPy may launch this module with a Windows cp1252 console.  The module
# uses useful status symbols in its logs, so make console output tolerant of
# either UTF-8 or legacy Windows encodings instead of failing during startup.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        pass


# ── Tuning constants ──────────────────────────────────────────────────────────

BLINK_MIN_MS          = 40
BLINK_MAX_MS          = 500
BLINK_REFRACTORY_S    = 0.15

FIXATION_DISPERSION   = 0.04
FIXATION_MIN_MS       = 100
GAZE_SMOOTHING_FRAMES = 3
QC_REPORT_S           = 10
MAX_CAMERA_READ_FAILURES = 30

# Video settings
VIDEO_CODEC           = 'mp4v'
VIDEO_FPS             = 30
VIDEO_QUEUE_MAXSIZE   = 120    # max frames buffered (~4s at 30fps)


class EEG_EyeTracker:
    """
    Eye tracker for multimodal deception detection experiments.

    Usage in main_experiment.py:
        eye_tracker = EEG_EyeTracker(
            subject_id         = subject_id,
            subject_name       = exp_info['Full Name'],
            session            = exp_info['Session'],
            eyetracking_folder = os.path.join(subject_folder, 'EyeTracking'),
            camera_id          = CFG['camera_id'],
            show_preview       = True,
            save_video         = True,
            screen_width       = CFG['screen_res'][0],
            screen_height      = CFG['screen_res'][1],
        )
    """

    def __init__(self, subject_id, subject_name, session,
                 experiment_name  = "DeceptionDetection",
                 eyetracking_folder = "EyeTracking",
                 camera_id        = 0,
                 show_preview     = True,
                 save_video       = False,
                 record_raw_video = True,
                 screen_width     = 1920,
                 screen_height    = 1080):

        self.subject_id          = subject_id
        self.subject_name        = subject_name
        self.session             = session
        self.experiment_name     = experiment_name
        self.eyetracking_folder  = eyetracking_folder
        self.show_preview        = show_preview
        self.save_video          = save_video
        # ── Raw video writer (background thread, no overlay) ──────────────
        self._raw_video_writer   = None
        self._raw_video_queue    = queue.Queue(maxsize=VIDEO_QUEUE_MAXSIZE)
        self._raw_video_thread   = None
        self._raw_video_path     = None
        self._raw_video_map_path = None
        self._raw_video_map_csv  = None
        self._raw_video_map_writer = None
        self._raw_video_frames   = 0
        self._raw_video_enqueued = 0
        self._raw_video_dropped  = 0
        self._raw_video_errors   = 0
        self.record_raw_video    = record_raw_video   # store the flag
        # The preview uses the same orientation as the saved raw video and
        # the processed coordinates.  Mirroring reverses overlay text and is
        # undesirable for an analysis/QA preview.
        self.preview_mirrored     = False
        self.screen_width        = screen_width
        self.screen_height       = screen_height
        # ── Video Markers CSV (separate file, only user-sent markers) ──
        self._video_markers_path   = None
        self._video_markers_csv    = None
        self._video_markers_writer = None

        os.makedirs(self.eyetracking_folder, exist_ok=True)

        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh    = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.7, min_tracking_confidence=0.7)

        # Landmark groups
        self.LEFT_EYE    = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
        self.RIGHT_EYE   = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
        self.LEFT_IRIS   = [474,475,476,477]
        self.RIGHT_IRIS  = [469,470,471,472]
        self.LEFT_EAR_PTS  = [386, 374, 362, 263]
        self.RIGHT_EAR_PTS = [159, 145, 33,  133]

        # Gaze smoothing
        self.gaze_history = deque(maxlen=GAZE_SMOOTHING_FRAMES)

        # Blink state
        self.blink_threshold_left  = 0.15
        self.blink_threshold_right = 0.15
        self.blink_threshold       = 0.15
        # No open-eye calibration is used.  These fixed defaults are explicit
        # so the standalone module and the full experiment behave identically.
        self._calibrated           = False
        self._in_blink       = False
        self._blink_t0       = None
        self.total_blinks    = 0
        self.last_blink_time = None

        # Fixation state
        self._fix_window   = deque()
        self._fix_id       = 0
        self._in_fixation  = False
        self._fix_t0       = None
        self._fix_centroid = (0.5, 0.5)

        # Live measurements
        self.current_measurements = dict(
            gaze_relative_x=math.nan, gaze_relative_y=math.nan,
            gaze_relative_x_raw=math.nan, gaze_relative_y_raw=math.nan,
            left_iris=(0, 0), right_iris=(0, 0),
            left_iris_diam_proxy=math.nan,
            right_iris_diam_proxy=math.nan,
            avg_iris_diam_proxy=math.nan,
            left_ear=0.0, right_ear=0.0, avg_ear=0.0,
            blink_detected=False, fixation_active=False,
            face_detected=False, fixation_stability=0.0,
            gaze_velocity_rel_s=math.nan,
        )

        # Camera
        self.camera_id     = camera_id
        self.cap           = None
        self.camera_width  = 640
        self.camera_height = 480
        self.camera_fps    = VIDEO_FPS
        self.camera_read_failures = 0
        self.face_detected_frames = 0
        self.face_lost_frames     = 0
        self._last_qc_t           = 0.0
        self._last_capture_lsl_t  = None
        self._last_elapsed        = 0.0

        # Tracking thread
        self.tracking_active = False
        self.tracking_thread = None
        self.latest_frame    = None
        self.frame_lock      = threading.Lock()
        self._preview_quit_requested = False

        # ── Video writer (background thread) ──────────────────────────────
        self._video_writer   = None
        self._video_queue    = queue.Queue(maxsize=VIDEO_QUEUE_MAXSIZE)
        self._video_thread   = None
        self._video_path     = None
        self._video_frames   = 0
        self._video_dropped  = 0

        # LSL
        self.lsl_marker_outlet = None
        self.lsl_data_outlet   = None
        self._setup_lsl()

        # CSV
        self._csv_lock = threading.Lock()
        self._setup_csv()

        # Counters
        self.start_time   = None
        self.start_lsl_time = None
        self.frame_number = 0

        print(f"\n✓ Eye tracker ready  →  {self.eyetracking_folder}")

    # ─────────────────────────────────────────────────────────────────────────
    # LSL
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_lsl(self):
        try:
            info = pylsl.StreamInfo(
                'EyeTrackingMarkers', 'Markers', 1, 0, 'string',
                f'eyetrack_mrk_{self.subject_id}')
            info.desc().append_child_value("subject_id", self.subject_id)
            self.lsl_marker_outlet = pylsl.StreamOutlet(info)
            print("✓ LSL: EyeTrackingMarkers")
        except Exception as e:
            print(f"⚠ LSL marker stream failed: {e}")

        try:
            info = pylsl.StreamInfo(
                'EyeTracking_Data', 'Gaze', 8, 30, 'float32',
                f'eyetrack_data_v2_{self.subject_id}')
            chns = info.desc().append_child('channels')
            channel_specs = [
                ('gaze_relative_x', 'normalized_eye_coordinate'),
                ('gaze_relative_y', 'normalized_eye_coordinate'),
                ('iris_diam_proxy', 'iris_diameter_over_eye_width'),
                ('avg_ear', 'ratio'),
                ('blink', 'boolean'),
                ('fixation', 'boolean_relative_gaze'),
                ('face_detected', 'boolean'),
                ('gaze_velocity_rel_s', 'normalized_eye_coordinate_per_s'),
            ]
            for lbl, unit in channel_specs:
                c = chns.append_child('channel')
                c.append_child_value('label', lbl)
                c.append_child_value('unit', unit)
            info.desc().append_child_value("subject_id", self.subject_id)
            info.desc().append_child_value("session",    self.session)
            info.desc().append_child_value(
                "note",
                "Webcam-relative eye tracking. This stream does not provide "
                "calibrated screen gaze or true pupil diameter.")
            info.desc().append_child_value("version", "2")
            self.lsl_data_outlet = pylsl.StreamOutlet(info)
            print("✓ LSL: EyeTracking_Data v2  (8 ch, 30 Hz → XDF)")
        except Exception as e:
            print(f"⚠ LSL data stream failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # CSV
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_csv(self):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{self.subject_id}_S{self.session}_{ts}"

        # ── Frames CSV ──────────────────────────────────────────────
        fp = os.path.join(self.eyetracking_folder, f"{base}_frames.csv")
        self._frames_csv    = open(fp, 'w', newline='')
        self._frames_writer = csv.writer(self._frames_csv)
        self._frames_writer.writerow([
            'wall_time','lsl_clock','frame','elapsed_s','face_detected',
            'raw_video_enqueued','raw_video_dropped_total',
            'gaze_relative_x_raw','gaze_relative_y_raw',
            'gaze_relative_x','gaze_relative_y',
            'left_iris_x','left_iris_y','right_iris_x','right_iris_y',
            'left_iris_diam_proxy','right_iris_diam_proxy','avg_iris_diam_proxy',
            'left_ear','right_ear','avg_ear',
            'blink_detected','fixation_active',
            'fixation_stability','gaze_velocity_rel_s',
            'blink_threshold_left','blink_threshold_right','calibrated',
        ])

        # ── Events CSV (blinks, fixations, markers – legacy) ─────
        ep = os.path.join(self.eyetracking_folder, f"{base}_events.csv")
        self._events_csv    = open(ep, 'w', newline='')
        self._events_writer = csv.writer(self._events_csv)
        self._events_writer.writerow([
            'event_type','event_id',
            'wall_time','lsl_clock','elapsed_s','duration_ms',
            'gaze_relative_x','gaze_relative_y',
            'left_ear','right_ear','avg_ear','face_detected',
            'marker_code','note',
        ])
        self._events_csv.flush()

        # ── NEW: Video Markers CSV (dedicated, only user markers) ──
        vmp = os.path.join(self.eyetracking_folder, f"VideoMarkers_{base}.csv")
        self._video_markers_path = vmp
        self._video_markers_csv = open(vmp, 'w', newline='')
        self._video_markers_writer = csv.writer(self._video_markers_csv)
        self._video_markers_writer.writerow([
            'wall_time', 'lsl_clock', 'elapsed_s', 'capture_frame',
            'last_capture_lsl_clock', 'raw_video_saved_frames',
            'raw_video_dropped_total',
            'marker_code', 'marker_label',
            'gaze_relative_x', 'gaze_relative_y',
            'left_ear', 'right_ear', 'avg_ear', 'face_detected',
            'note'
        ])
        self._video_markers_csv.flush()

        print(f"  frames → {os.path.basename(fp)}")
        print(f"  events → {os.path.basename(ep)}")
        print(f"  video markers → {os.path.basename(vmp)}")

    # ─────────────────────────────────────────────────────────────────────────
    # VIDEO — setup and background writer thread
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_video(self):
        """
        Create VideoWriter and start the background writer thread.
        Called from start_tracking() after camera is confirmed open.
        Saves overlay video (with landmarks + EAR + blink/fixation indicators).
        """
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{self.subject_id}_S{self.session}_{ts}_eyevideo.mp4"
        self._video_path = os.path.join(self.eyetracking_folder, name)

        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        self._video_writer = cv2.VideoWriter(
            self._video_path,
            fourcc,
            self.camera_fps,
            (self.camera_width, self.camera_height)
        )

        if not self._video_writer.isOpened():
            print(f"⚠ VideoWriter could not open: {self._video_path}")
            self._video_writer = None
            return

        # Start background writer thread
        self._video_thread = threading.Thread(
            target=self._video_write_loop, daemon=True)
        self._video_thread.start()

        print(f"✓ Video recording → {os.path.basename(self._video_path)}")

    # Save Raw Video Helper Functions
    def _setup_raw_video(self):
        """
        Create VideoWriter and start background writer thread for the
        raw (unannotated) camera feed.
        """
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{self.subject_id}_S{self.session}_{ts}_raw_original.mp4"
        self._raw_video_path = os.path.join(self.eyetracking_folder, name)
        self._raw_video_map_path = os.path.join(
            self.eyetracking_folder,
            f"{self.subject_id}_S{self.session}_{ts}_raw_video_frames.csv")

        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        self._raw_video_writer = cv2.VideoWriter(
            self._raw_video_path,
            fourcc,
            self.camera_fps,
            (self.camera_width, self.camera_height)
        )

        if not self._raw_video_writer.isOpened():
            print(f"⚠ Raw VideoWriter could not open: {self._raw_video_path}")
            self._raw_video_writer = None
            return

        self._raw_video_map_csv = open(self._raw_video_map_path, 'w', newline='')
        self._raw_video_map_writer = csv.writer(self._raw_video_map_csv)
        self._raw_video_map_writer.writerow([
            'saved_video_frame', 'capture_frame', 'wall_time',
            'lsl_clock', 'elapsed_s'
        ])
        self._raw_video_map_csv.flush()

        self._raw_video_thread = threading.Thread(
            target=self._raw_video_write_loop, daemon=True)
        self._raw_video_thread.start()
        print(f"✓ Raw original video → {os.path.basename(self._raw_video_path)}")
        print(f"  video frame map → {os.path.basename(self._raw_video_map_path)}")

    def _raw_video_write_loop(self):
        """Background thread: drains original frames and records their mapping."""
        while True:
            item = self._raw_video_queue.get()
            if item is None:
                break
            capture_frame, wall_time, lsl_t, elapsed, frame = item
            try:
                if self._raw_video_writer and self._raw_video_writer.isOpened():
                    self._raw_video_writer.write(frame)
                    self._raw_video_frames += 1
                    if self._raw_video_map_writer:
                        self._raw_video_map_writer.writerow([
                            self._raw_video_frames, capture_frame, wall_time,
                            f"{lsl_t:.6f}", f"{elapsed:.6f}"
                        ])
                        if self._raw_video_frames % 30 == 0:
                            self._raw_video_map_csv.flush()
                else:
                    self._raw_video_errors += 1
            except Exception as e:
                self._raw_video_errors += 1
                print(f"⚠ Raw video write error: {e}")
        if self._raw_video_writer:
            self._raw_video_writer.release()
            self._raw_video_writer = None
        if self._raw_video_map_csv:
            self._raw_video_map_csv.flush()
            self._raw_video_map_csv.close()
            self._raw_video_map_csv = None

    def _enqueue_raw_frame(self, capture_frame, wall_time, lsl_t, elapsed, frame):
        """Non-blocking enqueue for original frames. Returns enqueue success."""
        try:
            self._raw_video_queue.put_nowait(
                (capture_frame, wall_time, lsl_t, elapsed, frame))
            self._raw_video_enqueued += 1
            return True
        except queue.Full:
            self._raw_video_dropped += 1
            return False

    def _video_write_loop(self):
        """
        Background thread: drains the video queue and writes frames to disk.
        Receives None as sentinel to stop.
        """
        while True:
            frame = self._video_queue.get()
            if frame is None:
                break
            if self._video_writer and self._video_writer.isOpened():
                self._video_writer.write(frame)
                self._video_frames += 1
        # Flush and release
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None

    def _enqueue_frame(self, frame):
        """
        Put an overlay frame in the video queue.
        If queue is full (writer falling behind), drop the frame silently
        rather than blocking the tracking loop.
        """
        try:
            self._video_queue.put_nowait(frame)
        except queue.Full:
            self._video_dropped += 1

    # ─────────────────────────────────────────────────────────────────────────
    # CAMERA
    # ─────────────────────────────────────────────────────────────────────────

    def initialize_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.camera_width = actual_w if actual_w > 0 else self.camera_width
        self.camera_height = actual_h if actual_h > 0 else self.camera_height
        self.camera_fps = actual_fps if 1.0 <= actual_fps <= 120.0 else VIDEO_FPS
        print(f"✓ Camera {self.camera_id}: {self.camera_width}×{self.camera_height} "
              f"@ {self.camera_fps:.1f} FPS")
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # START / STOP / CLOSE
    # ─────────────────────────────────────────────────────────────────────────

    def start_tracking(self):
        if self.tracking_active:
            return
        self.tracking_active = True
        self.start_time      = time.monotonic()
        self.start_lsl_time  = pylsl.local_clock()
        self.frame_number    = 0
        self.camera_read_failures = 0
        self.face_detected_frames = 0
        self.face_lost_frames = 0
        self._last_qc_t = 0.0
        self._last_capture_lsl_t = None
        self._last_elapsed = 0.0
        self._preview_quit_requested = False

        # Start annotated video writer (if requested)
        if self.save_video:
            self._setup_video()

        # Start raw video writer (if requested)   <-- NEW
        if self.record_raw_video:
            self._setup_raw_video()

        self.tracking_thread = threading.Thread(
            target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        print("✓ Eye tracking started")

        if self.show_preview:
            threading.Thread(target=self._preview_loop, daemon=True).start()
            print("✓ Preview window started")

    def stop_tracking(self):
        self.tracking_active = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=2.0)

        # Persist an active fixation rather than silently losing its final event.
        if self._in_fixation:
            self._finalize_fixation(
                self._last_elapsed, self._last_capture_lsl_t,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                note='tracking_stop')

        # Stop annotated video writer thread (if it was started)
        if self.save_video and self._video_thread is not None:
            self._video_queue.put(None, timeout=5.0)
            self._video_thread.join(timeout=10.0)
            print(f"✓ Annotated video saved: {self._video_frames} frames "
                f"→ {os.path.basename(self._video_path)}")

        # ── NEW: Stop raw video writer thread ──
        if self.record_raw_video and self._raw_video_thread is not None:
            self._raw_video_queue.put(None, timeout=5.0)
            self._raw_video_thread.join(timeout=10.0)
            print(f"✓ Raw original video saved: {self._raw_video_frames} frames "
                f"→ {os.path.basename(self._raw_video_path)}")

        print("✓ Eye tracking stopped")

    def close(self):
        self.stop_tracking()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

        try:
            self.face_mesh.close()
        except Exception:
            pass

        self._frames_csv.close()
        self._events_csv.close()
        if self._video_markers_csv:
            self._video_markers_csv.close()

        # Release raw video writer if still open (safety)
        if self._raw_video_writer:
            self._raw_video_writer.release()
        if self._raw_video_map_csv:
            self._raw_video_map_csv.close()

        print(f"\n{'='*55}")
        print(f"EYE TRACKING SAVED  →  {self.eyetracking_folder}")
        print(f"  Frames    : {self.frame_number}")
        print(f"  Face frames / lost : {self.face_detected_frames} / {self.face_lost_frames}")
        print(f"  Blinks    : {self.total_blinks}")
        print(f"  Fixations : {self._fix_id}")
        print(f"  EAR thr L/R : {self.blink_threshold_left:.4f} / "
              f"{self.blink_threshold_right:.4f}")
        if self.save_video and self._video_path:
            print(f"  Annotated video : {os.path.basename(self._video_path)}")
            print(f"  Ann. vid frames : {self._video_frames}")
        if self.record_raw_video and self._raw_video_path:
            print(f"  Raw video       : {os.path.basename(self._raw_video_path)}")
            print(f"  Raw vid frames  : {self._raw_video_frames}")
            print(f"  Raw vid dropped : {self._raw_video_dropped}")
            print(f"  Raw vid errors  : {self._raw_video_errors}")
            if self._raw_video_map_path:
                print(f"  Raw frame map   : {os.path.basename(self._raw_video_map_path)}")
        if self._video_markers_path:
            print(f"  Video markers   : {os.path.basename(self._video_markers_path)}")
        print(f"{'='*55}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # TRACKING LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def _tracking_loop(self):
        while self.tracking_active:
            if self.cap is None or not self.cap.isOpened():
                break
            ret, original_frame = self.cap.read()
            if not ret:
                self.camera_read_failures += 1
                if self.camera_read_failures >= MAX_CAMERA_READ_FAILURES:
                    print("⚠ Eye camera read failed repeatedly — stopping tracking.")
                    self.tracking_active = False
                    break
                time.sleep(0.01)
                continue

            self.camera_read_failures = 0
            self.frame_number += 1
            capture_lsl_t = pylsl.local_clock()
            elapsed = (capture_lsl_t - self.start_lsl_time
                       if self.start_lsl_time else 0.0)
            wall_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            self._last_capture_lsl_t = capture_lsl_t
            self._last_elapsed = elapsed
            fh, fw = original_frame.shape[:2]

            # Save the actual camera frame before any mirror/overlay processing.
            raw_video_enqueued = False
            if self.record_raw_video:
                raw_video_enqueued = self._enqueue_raw_frame(
                    self.frame_number, wall_time, capture_lsl_t, elapsed,
                    original_frame.copy())

            result = self.face_mesh.process(
                cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))

            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                m  = self._compute_measurements(lm, fw, fh, elapsed)
                self.current_measurements.update(m)
                self._update_blink(
                    m['left_ear'], m['right_ear'], elapsed,
                    capture_lsl_t, wall_time)
                self._update_fixation(
                    m['gaze_relative_x'], m['gaze_relative_y'], elapsed,
                    capture_lsl_t, wall_time)

                # State updates occur after measurement extraction. Store the
                # updated flags in this same frame, not one frame later.
                m['blink_detected'] = self._in_blink
                m['fixation_active'] = self._in_fixation
                m['face_detected'] = True
                self.current_measurements.update(m)
                self.face_detected_frames += 1
                self._push_lsl(m, capture_lsl_t)
                self._write_frame(
                    wall_time, capture_lsl_t, elapsed, m, raw_video_enqueued)
                overlay = self._draw_overlay(original_frame.copy(), lm, fw, fh)
            else:
                self.face_lost_frames += 1
                self._handle_face_lost(elapsed, capture_lsl_t, wall_time)
                m = self._no_face_measurements()
                self.current_measurements.update(m)
                self._push_lsl(m, capture_lsl_t)
                self._write_frame(
                    wall_time, capture_lsl_t, elapsed, m, raw_video_enqueued)
                overlay = original_frame

            # Share with preview
            with self.frame_lock:
                self.latest_frame = overlay

            # Send to video writer queue (non-blocking)
            if self.save_video:
                self._enqueue_frame(overlay.copy())

            self._report_qc(elapsed)

    # ─────────────────────────────────────────────────────────────────────────
    # COMPUTE MEASUREMENTS
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_measurements(self, lm, fw, fh, elapsed):
        l_iris = self._iris_center(lm, self.LEFT_IRIS,  fw, fh)
        r_iris = self._iris_center(lm, self.RIGHT_IRIS, fw, fh)

        l_ear   = self._ear(lm, self.LEFT_EAR_PTS,  fw, fh)
        r_ear   = self._ear(lm, self.RIGHT_EAR_PTS, fw, fh)
        avg_ear = (l_ear + r_ear) / 2

        # Gaze uses fixed eye corners and eyelid landmarks. It is a relative
        # eye coordinate, not a calibrated point of regard on the screen.
        lrx, lry = self._gaze_ratio(
            l_iris,
            self._landmark_xy(lm, 362, fw, fh),
            self._landmark_xy(lm, 263, fw, fh),
            self._landmark_xy(lm, 386, fw, fh),
            self._landmark_xy(lm, 374, fw, fh))
        rrx, rry = self._gaze_ratio(
            r_iris,
            self._landmark_xy(lm, 33, fw, fh),
            self._landmark_xy(lm, 133, fw, fh),
            self._landmark_xy(lm, 159, fw, fh),
            self._landmark_xy(lm, 145, fw, fh))
        gx_raw, gy_raw = (lrx + rrx) / 2, (lry + rry) / 2
        self.gaze_history.append((gx_raw, gy_raw))
        gx = float(np.mean([g[0] for g in self.gaze_history]))
        gy = float(np.mean([g[1] for g in self.gaze_history]))

        l_iris_proxy = self._iris_diameter_proxy(
            lm, self.LEFT_IRIS, fw, fh,
            self._landmark_xy(lm, 362, fw, fh),
            self._landmark_xy(lm, 263, fw, fh))
        r_iris_proxy = self._iris_diameter_proxy(
            lm, self.RIGHT_IRIS, fw, fh,
            self._landmark_xy(lm, 33, fw, fh),
            self._landmark_xy(lm, 133, fw, fh))

        return dict(
            gaze_relative_x=gx, gaze_relative_y=gy,
            gaze_relative_x_raw=gx_raw, gaze_relative_y_raw=gy_raw,
            left_iris=l_iris, right_iris=r_iris,
            left_iris_diam_proxy=l_iris_proxy,
            right_iris_diam_proxy=r_iris_proxy,
            avg_iris_diam_proxy=(l_iris_proxy + r_iris_proxy) / 2,
            left_ear=l_ear, right_ear=r_ear, avg_ear=avg_ear,
            blink_detected=False,
            fixation_active=False,
            face_detected=True,
            fixation_stability=self._fixation_stability(),
            gaze_velocity_rel_s=self._gaze_velocity(gx, gy, elapsed),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # EAR
    # ─────────────────────────────────────────────────────────────────────────

    def _ear(self, lm, pts, fw, fh):
        top    = np.array([lm[pts[0]].x*fw, lm[pts[0]].y*fh])
        bottom = np.array([lm[pts[1]].x*fw, lm[pts[1]].y*fh])
        left   = np.array([lm[pts[2]].x*fw, lm[pts[2]].y*fh])
        right  = np.array([lm[pts[3]].x*fw, lm[pts[3]].y*fh])
        vert   = np.linalg.norm(top - bottom)
        horiz  = np.linalg.norm(left - right)
        return float(vert / horiz) if horiz > 1 else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # BLINK
    # ─────────────────────────────────────────────────────────────────────────

    def _update_blink(self, left_ear, right_ear, elapsed, lsl_t, wall_time):
        is_closed = (left_ear < self.blink_threshold_left and
                     right_ear < self.blink_threshold_right)

        if is_closed and not self._in_blink:
            self._in_blink = True
            self._blink_t0 = elapsed
        elif not is_closed and self._in_blink:
            dur_ms = (elapsed - self._blink_t0) * 1000
            ref_ok = (self.last_blink_time is None or
                      elapsed - self.last_blink_time > BLINK_REFRACTORY_S)
            if BLINK_MIN_MS <= dur_ms <= BLINK_MAX_MS and ref_ok:
                self.total_blinks += 1
                self.last_blink_time = elapsed
                m = self.current_measurements
                self._write_event([
                    'blink', self.total_blinks, wall_time, f"{lsl_t:.6f}",
                    f"{elapsed:.6f}", f"{dur_ms:.1f}",
                    self._csv_float(m['gaze_relative_x']),
                    self._csv_float(m['gaze_relative_y']),
                    self._csv_float(left_ear), self._csv_float(right_ear),
                    self._csv_float((left_ear + right_ear) / 2), 1,
                    '', f"EAR_onset={self._blink_t0:.6f}",
                ])
            self._in_blink = False
            self._blink_t0 = None

    # ─────────────────────────────────────────────────────────────────────────
    # FIXATION (webcam-relative I-DT)
    # ─────────────────────────────────────────────────────────────────────────

    def _window_dispersion(self):
        if len(self._fix_window) < 2:
            return 0.0
        pts = np.asarray([(p[1], p[2]) for p in self._fix_window])
        return float((pts.max(axis=0) - pts.min(axis=0)).max())

    def _finalize_fixation(self, end_elapsed, lsl_t, wall_time, note):
        if not self._in_fixation or len(self._fix_window) < 2:
            self._in_fixation = False
            self._fix_window.clear()
            return
        start_elapsed = self._fix_window[0][0]
        dur_ms = max(0.0, (end_elapsed - start_elapsed) * 1000)
        if dur_ms >= FIXATION_MIN_MS:
            pts = np.asarray([(p[1], p[2]) for p in self._fix_window])
            cx, cy = np.mean(pts, axis=0)
            self._fix_id += 1
            m = self.current_measurements
            self._write_event([
                'fixation', self._fix_id, wall_time,
                f"{lsl_t:.6f}" if lsl_t is not None else '',
                f"{start_elapsed:.6f}", f"{dur_ms:.1f}",
                self._csv_float(cx), self._csv_float(cy),
                self._csv_float(m['left_ear']), self._csv_float(m['right_ear']),
                self._csv_float(m['avg_ear']), 1, '', note,
            ])
        self._in_fixation = False
        self._fix_window.clear()

    def _update_fixation(self, gx, gy, elapsed, lsl_t, wall_time):
        point = (elapsed, gx, gy)
        if not self._in_fixation:
            self._fix_window.append(point)
            # Shrink the candidate until it is spatially coherent.
            while len(self._fix_window) > 1 and \
                    self._window_dispersion() > FIXATION_DISPERSION:
                self._fix_window.popleft()
            dur_ms = (self._fix_window[-1][0] - self._fix_window[0][0]) * 1000
            if dur_ms >= FIXATION_MIN_MS:
                self._in_fixation = True
                self._fix_t0 = self._fix_window[0][0]
                self._fix_centroid = tuple(np.mean(
                    [(p[1], p[2]) for p in self._fix_window], axis=0))
            return

        self._fix_window.append(point)
        if self._window_dispersion() > FIXATION_DISPERSION:
            # The latest point belongs to the next candidate; close the
            # previous fixation without contaminating it with that point.
            next_candidate = self._fix_window.pop()
            end_t = self._fix_window[-1][0]
            self._finalize_fixation(end_t, lsl_t, wall_time, 'gaze_shift')
            self._fix_window.append(next_candidate)

    def _handle_face_lost(self, elapsed, lsl_t, wall_time):
        if self._in_fixation:
            self._finalize_fixation(elapsed, lsl_t, wall_time, 'face_lost')
        self._in_blink = False
        self._blink_t0 = None
        self._fix_window.clear()
        self.gaze_history.clear()
        if hasattr(self, '_pg'):
            del self._pg
            del self._pt

    def _no_face_measurements(self):
        return dict(
            gaze_relative_x=math.nan, gaze_relative_y=math.nan,
            gaze_relative_x_raw=math.nan, gaze_relative_y_raw=math.nan,
            left_iris=(0, 0), right_iris=(0, 0),
            left_iris_diam_proxy=math.nan,
            right_iris_diam_proxy=math.nan,
            avg_iris_diam_proxy=math.nan,
            left_ear=math.nan, right_ear=math.nan, avg_ear=math.nan,
            blink_detected=False, fixation_active=False,
            face_detected=False, fixation_stability=math.nan,
            gaze_velocity_rel_s=math.nan,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # LSL + MARKERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _finite_or_zero(value):
        try:
            value = float(value)
            return value if math.isfinite(value) else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _push_lsl(self, m, capture_lsl_t):
        if self.lsl_data_outlet:
            self.lsl_data_outlet.push_sample([
                self._finite_or_zero(m['gaze_relative_x']),
                self._finite_or_zero(m['gaze_relative_y']),
                self._finite_or_zero(m['avg_iris_diam_proxy']),
                self._finite_or_zero(m['avg_ear']),
                1.0 if m['blink_detected'] else 0.0,
                1.0 if m['fixation_active'] else 0.0,
                1.0 if m['face_detected'] else 0.0,
                self._finite_or_zero(m['gaze_velocity_rel_s']),
            ], timestamp=capture_lsl_t)

    def send_marker(self, marker_code, label=''):
        """Send an event marker and save its relation to webcam timing."""
        lsl_time = pylsl.local_clock()
        if self.lsl_marker_outlet:
            self.lsl_marker_outlet.push_sample([str(marker_code)], timestamp=lsl_time)

        elapsed = (lsl_time - self.start_lsl_time
                   if self.start_lsl_time is not None else 0.0)
        m = self.current_measurements.copy()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        self._write_event([
            'marker', '', now, f"{lsl_time:.6f}", f"{elapsed:.6f}", '',
            self._csv_float(m['gaze_relative_x']),
            self._csv_float(m['gaze_relative_y']),
            self._csv_float(m['left_ear']), self._csv_float(m['right_ear']),
            self._csv_float(m['avg_ear']), 1 if m['face_detected'] else 0,
            marker_code, label,
        ])

        if self._video_markers_writer is not None:
            with self._csv_lock:
                self._video_markers_writer.writerow([
                    now, f"{lsl_time:.6f}", f"{elapsed:.6f}",
                    self.frame_number,
                    (f"{self._last_capture_lsl_t:.6f}"
                     if self._last_capture_lsl_t is not None else ''),
                    self._raw_video_frames, self._raw_video_dropped,
                    marker_code, label,
                    self._csv_float(m['gaze_relative_x']),
                    self._csv_float(m['gaze_relative_y']),
                    self._csv_float(m['left_ear']),
                    self._csv_float(m['right_ear']),
                    self._csv_float(m['avg_ear']),
                    1 if m['face_detected'] else 0,
                    '',
                ])
                self._video_markers_csv.flush()

        print(f"  [Eye] Marker {marker_code} ({label})  "
              f"face={int(bool(m['face_detected']))}  "
              f"frame={self.frame_number}  elapsed={elapsed:.3f}s")

    # ─────────────────────────────────────────────────────────────────────────
    # CSV FRAME WRITE
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _csv_float(value, digits=6):
        """Return a finite numeric value for CSV, or an empty field."""
        try:
            value = float(value)
            return f"{value:.{digits}f}" if math.isfinite(value) else ''
        except (TypeError, ValueError):
            return ''

    def _write_event(self, row):
        """Serialize event writes from the tracker and experiment threads."""
        with self._csv_lock:
            self._events_writer.writerow(row)
            self._events_csv.flush()

    def _write_frame(self, wall_time, lsl_t, elapsed, m,
                     raw_video_enqueued):
        li = m.get('left_iris', (0, 0))
        ri = m.get('right_iris', (0, 0))
        row = [
            wall_time, f"{lsl_t:.6f}", self.frame_number,
            f"{elapsed:.6f}",
            1 if m.get('face_detected') else 0,
            1 if raw_video_enqueued else 0,
            self._raw_video_dropped,
            self._csv_float(m.get('gaze_relative_x_raw')),
            self._csv_float(m.get('gaze_relative_y_raw')),
            self._csv_float(m.get('gaze_relative_x')),
            self._csv_float(m.get('gaze_relative_y')),
            self._csv_float(li[0]), self._csv_float(li[1]),
            self._csv_float(ri[0]), self._csv_float(ri[1]),
            self._csv_float(m.get('left_iris_diam_proxy')),
            self._csv_float(m.get('right_iris_diam_proxy')),
            self._csv_float(m.get('avg_iris_diam_proxy')),
            self._csv_float(m.get('left_ear')),
            self._csv_float(m.get('right_ear')),
            self._csv_float(m.get('avg_ear')),
            1 if m.get('blink_detected') else 0,
            1 if m.get('fixation_active') else 0,
            self._csv_float(m.get('fixation_stability')),
            self._csv_float(m.get('gaze_velocity_rel_s')),
            self._csv_float(self.blink_threshold_left),
            self._csv_float(self.blink_threshold_right),
            1 if self._calibrated else 0,
        ]
        with self._csv_lock:
            self._frames_writer.writerow(row)
            if self.frame_number % 30 == 0:
                self._frames_csv.flush()

    def _report_qc(self, elapsed):
        if elapsed - self._last_qc_t < QC_REPORT_S:
            return
        self._last_qc_t = elapsed
        total = self.face_detected_frames + self.face_lost_frames
        face_pct = (100.0 * self.face_detected_frames / total
                    if total else 0.0)
        print(f"[Eye] {self.frame_number} frames | "
              f"face={face_pct:.1f}% | lost={self.face_lost_frames} | "
              f"raw_drop={self._raw_video_dropped} | "
              f"blinks={self.total_blinks} | fix={self._fix_id}")

    # ─────────────────────────────────────────────────────────────────────────
    # GEOMETRY HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _eye_points(self, lm, idx, fw, fh):
        pts = [(int(round(lm[i].x * fw)), int(round(lm[i].y * fh)))
               for i in idx]
        return tuple(np.mean(pts, axis=0).astype(int)), pts

    def _iris_center(self, lm, idx, fw, fh):
        pts = [(lm[i].x*fw, lm[i].y*fh) for i in idx]
        return tuple(np.mean(pts, axis=0).astype(float))

    def _landmark_xy(self, lm, idx, fw, fh):
        return (float(lm[idx].x * fw), float(lm[idx].y * fh))

    def _gaze_ratio(self, iris, corner_a, corner_b, lid_a, lid_b):
        """Normalize iris position inside one eye; not screen gaze."""
        x0, x1 = sorted((corner_a[0], corner_b[0]))
        y0, y1 = sorted((lid_a[1], lid_b[1]))
        ew = x1 - x0
        eh = y1 - y0
        rx = (iris[0] - x0) / ew if ew > 1.0 else 0.5
        ry = (iris[1] - y0) / eh if eh > 1.0 else 0.5
        return float(np.clip(rx,0,1)), float(np.clip(ry,0,1))

    def _iris_diameter_proxy(self, lm, iris_idx, fw, fh,
                             corner_a, corner_b):
        """Estimate iris diameter normalized by eye width, not pupil diameter."""
        pts = np.asarray([(lm[i].x * fw, lm[i].y * fh) for i in iris_idx])
        iris_width = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        eye_width = float(np.linalg.norm(
            np.asarray(corner_a) - np.asarray(corner_b)))
        return iris_width / eye_width if eye_width > 1.0 else math.nan

    def _fixation_stability(self):
        if len(self.gaze_history) < 3:
            return 0.0
        p    = np.array(list(self.gaze_history))
        disp = np.max(np.linalg.norm(p-p.mean(axis=0), axis=1))
        return float(1.0/(1.0+disp*10))

    def _head_distance(self, lep, rep):
        lc  = np.mean(lep, axis=0)
        rc  = np.mean(rep, axis=0)
        ied = np.linalg.norm(rc-lc)
        return float((120.0/ied)*60.0) if ied>0 else 0.0

    def _gaze_velocity(self, gx, gy, elapsed):
        if not hasattr(self, '_pg'):
            self._pg = (gx, gy); self._pt = elapsed; return 0.0
        dt  = elapsed - self._pt
        vel = float(np.hypot(gx-self._pg[0], gy-self._pg[1])/dt) if dt>0 else 0.0
        self._pg = (gx, gy); self._pt = elapsed
        return vel

    # ─────────────────────────────────────────────────────────────────────────
    # OVERLAY + PREVIEW
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_overlay(self, frame, lm, fw, fh):
        _, lep = self._eye_points(lm, self.LEFT_EYE,  fw, fh)
        _, rep = self._eye_points(lm, self.RIGHT_EYE, fw, fh)
        cv2.polylines(frame, [np.array(lep)], True, (0,255,0), 1)
        cv2.polylines(frame, [np.array(rep)], True, (0,255,0), 1)
        li = self._iris_center(lm, self.LEFT_IRIS,  fw, fh)
        ri = self._iris_center(lm, self.RIGHT_IRIS, fw, fh)
        li = (int(round(li[0])), int(round(li[1])))
        ri = (int(round(ri[0])), int(round(ri[1])))
        cv2.circle(frame, li, 15, (255,0,0), 2)
        cv2.circle(frame, li,  3, (0,255,255), -1)
        cv2.circle(frame, ri, 15, (255,0,0), 2)
        cv2.circle(frame, ri,  3, (0,255,255), -1)

        m   = self.current_measurements
        thr = (f"fixed blink thr L/R={self.blink_threshold_left:.3f}/"
               f"{self.blink_threshold_right:.3f}")
        for i, txt in enumerate([
            f"Sub:{self.subject_id}  Fr:{self.frame_number}",
            f"Gaze rel:({m['gaze_relative_x']:.3f},"
            f"{m['gaze_relative_y']:.3f})",
            f"EAR L={m['left_ear']:.3f} R={m['right_ear']:.3f}  {thr}",
            f"Iris proxy={m['avg_iris_diam_proxy']:.3f}",
            f"Blinks:{self.total_blinks}  Fix:{self._fix_id}",
            f"Face:{int(bool(m['face_detected']))}",
        ]):
            cv2.putText(frame, txt, (8, 22+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,255,0), 1)

        if self._in_blink:
            cv2.putText(frame, "BLINK", (fw-95, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        if self._in_fixation:
            cv2.putText(frame, "FIX", (fw-68, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
        return frame

    def _preview_loop(self):
        cv2.namedWindow('Eye Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Eye Tracking', 800, 600)
        while self.tracking_active:
            with self.frame_lock:
                if self.latest_frame is None:
                    time.sleep(0.01)
                    continue
                f = self.latest_frame.copy()
            cv2.imshow('Eye Tracking', f)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                self._preview_quit_requested = True
                break
            try:
                if cv2.getWindowProperty(
                        'Eye Tracking', cv2.WND_PROP_VISIBLE) < 1:
                    self._preview_quit_requested = True
                    break
            except cv2.error:
                self._preview_quit_requested = True
                break
        cv2.destroyAllWindows()


def _standalone_main():
    """Run the eye tracker by itself for camera/LSL/CSV/video verification."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Standalone MediaPipe eye-tracking and LSL test')
    parser.add_argument(
        '--camera', type=int, default=0,
        help='OpenCV camera index (default: 0 for the current laptop camera)')
    parser.add_argument(
        '--duration', type=float, default=0.0,
        help='Recording duration in seconds; 0 means until q/Ctrl+C')
    parser.add_argument(
        '--output', default=None,
        help='Output folder (default: EyeTracking_Standalone beside this file)')
    parser.add_argument(
        '--no-preview', action='store_true',
        help='Disable the OpenCV preview window')
    args = parser.parse_args()

    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'EyeTracking_Standalone')
    session = datetime.now().strftime('%Y%m%d_%H%M%S')
    tracker = EEG_EyeTracker(
        subject_id='standalone',
        subject_name='standalone_camera_test',
        session=session,
        experiment_name='EyeTrackingStandalone',
        eyetracking_folder=output,
        camera_id=args.camera,
        show_preview=not args.no_preview,
        save_video=False,
        record_raw_video=True,
    )

    try:
        tracker.initialize_camera()
        tracker.start_tracking()
        print(f"[Eye] Standalone recording → {output}")
        print("[Eye] Press q/ESC in the preview or Ctrl+C to stop.")
        t0 = time.monotonic()
        while tracker.tracking_active and not tracker._preview_quit_requested:
            if args.duration > 0 and time.monotonic() - t0 >= args.duration:
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[Eye] Stop requested by user.")
    finally:
        tracker.close()


if __name__ == '__main__':
    _standalone_main()
