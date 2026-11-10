# SIRR: Synchronized Integrated Recording with Role-based Responding

Code for the SIRR paradigm and its acquisition pipeline.

SIRR is a within-subject deception paradigm. Each participant watches two AI-generated theft videos and answers
the same 12 on-screen questions aloud (in Egyptian Arabic) under two roles:
- **Cooperate**: answer honestly.
- **Protect**: protect the person who committed the theft, deciding freely which answers to falsify.

Each answer is later labelled by participant self-report as truthful, deceptive or memory failure.
EEG, ECG/heart rate, GSR, facial video with webcam-based eye tracking, and audio are recorded on two computers
synchronized with the [Lab Streaming Layer (LSL)](https://github.com/sccn/labstreaminglayer).

> **No participant data are included in this repository.** The dataset (HCM-SIRR) will be released separately
> in a data descriptor.

## Repository structure

```
experiment/                    task and acquisition code (as used for data collection)
  main_experiment.py           PsychoPy task, runs on PC2 (role instructions, videos, question trials, markers)
  psychopy_eeg_eye_tracker.py  webcam eye/face module (MediaPipe Face Mesh, 8-channel LSL stream)
  gsr_module.py                Grove GSR v2.0 via Arduino serial, LSL stream
  voice_module.py              lavalier microphone, 16 kHz LSL audio stream
  lab_recorder.py              remote control of LabRecorder (start/stop/refresh)
  polar_streamer_pc1.py        Polar H10 (ECG, HR, R-R, RMSSD, ACC) to LSL, runs on PC1
```

## Hardware

| Computer | Device | Stream |
|---|---|---|
| PC1 | ANT Neuro eego EE-224 amplifier, 64-ch ANT Neuro waveguard cap (CPz reference), ANT Neuro eego software 1.9.2 | EEG, 1000 Hz (LSL) |
| PC1 | Polar H10 chest strap (Bluetooth LE) | ECG 130 Hz, HR, R-R, ACC 25 Hz |
| PC2 | Grove GSR v2.0 + Arduino (USB serial, 115200 baud) | GSR, ~98 Hz |
| PC2 | Logitech Brio 300 webcam | 8-channel eye/face stream, 30 Hz, plus raw video |
| PC2 | Boya BY-M1 lavalier microphone | audio, 16 kHz |

The two computers are on the same local Wi-Fi network. All streams are recorded into one XDF file by
LabRecorder on PC2.

## Running the experiment

1. Install the dependencies: `pip install -r requirements-experiment.txt`.
2. On PC1, start the ANT Neuro eego LSL stream and run `python experiment/polar_streamer_pc1.py`.
   Set `DEVICE_ADDRESS` to your Polar H10's address first.
3. On PC2, set the GSR serial port in `main_experiment.py` (default `COM3`). Place the two stimulus videos
   (`video_a.mp4`, `video_b.mp4`) next to the script, start LabRecorder, and run
   `python experiment/main_experiment.py`.
4. Select the counterbalancing sequence (1-4) at start-up:
   - S1: Cooperate-A then Protect-B
   - S2: Cooperate-B then Protect-A
   - S3: Protect-A then Cooperate-B
   - S4: Protect-B then Cooperate-A

### Event markers

Each marker is pushed to four LSL marker streams: task, voice, eye tracking and GSR.

| Code | Event |
|---|---|
| 1 | experiment start |
| 2 / 3 | baseline start / end (the recovery baseline reuses these codes) |
| 4 / 5 | acquaintance start / end |
| 200-211 | role instructions, videos and question blocks |
| 300 + 10 x trial + phase | trial events |
| 999 | end of experiment |

In trial events, trial is 1-12 for Video A and 13-24 for Video B. Phase is one of:

| Phase | Event |
|---|---|
| 1 | fixation |
| 2 | question onset |
| 3 | response onset |
| 4 | response offset |
| 5 | truthful judgement |
| 6 | deceptive judgement |
| 7 | ITI onset |

## Ethics

Data collection was approved by the Institutional Review Board of the Faculty of Engineering, Ain Shams
University (IRB-ASU-26-038). All participants gave written informed consent.

## License

MIT (see `LICENSE`). Developed at the Human Centered Mechatronics (HCM) Lab, Ain Shams University.
