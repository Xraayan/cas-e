# CAS-E Desktop Guide

CAS-E is a live conversational guide for the Electronics and Communication Department at Rajiv Gandhi Institute of Technology, Kottayam. It runs as a desktop voice assistant, answers department and college questions from local knowledge sources, can search the web when needed, and can show the class timetable on screen when visitors ask for it.

## What It Does

- Runs a realtime voice assistant using Google GenAI Live.
- Uses local department knowledge from `casie_direct/KMS`.
- Answers EC department FAQs, college details, and website-context questions through tools.
- Displays the timetable image from `casie_direct/KMS/timetable.jpg`, `.jpeg`, or `.png`.
- Provides a fullscreen desktop UI with live transcript lanes, audio level visualization, wake/end controls, and idle auto-sleep.
- Includes `run_casie_desktop.sh` for launching the app on Linux or Raspberry Pi style desktop sessions.

## Project Layout

```text
casie_direct/
  agent.py              # realtime voice session and audio pipeline
  desktop_ui.py         # Tkinter fullscreen desktop experience
  tools.py              # assistant tools for FAQ/search/timetable display
  prompts.py            # assistant persona and session instructions
  requirements.txt      # Python dependencies
  KMS/                  # local knowledge, cache, timetable assets
run_casie_desktop.sh    # Linux desktop launcher
```

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r casie_direct/requirements.txt
```

Create a local `.env` file from `.env.example` and fill in the required keys for your machine. Keep `.env` private; it is intentionally ignored by git.

## Run

From the repo root:

```bash
source .venv/bin/activate
cd casie_direct
python3 desktop_ui.py
```

On Linux desktop devices, you can use the launcher:

```bash
./run_casie_desktop.sh
```

The launcher writes logs to `logs/casie_desktop.log`, waits briefly for the display server, activates `.venv` or `venv` when present, and starts `desktop_ui.py`.

## Timetable

Place the current timetable image in `casie_direct/KMS` using one of these names:

- `timetable.jpg`
- `timetable.jpeg`
- `timetable.png`

When a user asks for the timetable, CAS-E emits a UI event and the desktop app shows the image for the configured display duration.

## Useful Environment Settings

Most runtime behavior is controlled through `.env`. Common settings include:

- `GOOGLE_API_KEY`
- `CASIE_MODEL`
- `CASIE_INPUT_DEVICE`
- `CASIE_OUTPUT_DEVICE`
- `CASIE_UI_AUTO_SLEEP_IDLE_MS`
- `CASIE_TIMETABLE_SELF_TEST`
- `CASIE_TIMETABLE_MARGIN_PX`

Do not commit local keys, device names, or machine-specific values.

## Validation

Run a quick syntax check before pushing:

```bash
python3 -m compileall casie_direct
```
