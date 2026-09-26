<div align="center">

<img src="docs/images/architecture-overview.png" alt="Visilytics detection pipeline: RTSP camera to YOLOv8 + BoT-SORT tracking to direction filter to AWS Rekognition to REST API upload" width="100%"/>

# 👁️ Visilytics

### Real-time retail person detection — live across **45+ stores**

Turns the cameras a store already has into a clean, front-facing visitor feed.<br/>
**RTSP in → local YOLOv8 tracking → smart filtering → one cheap AWS check → analytics API.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111F68?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-RTSP%20%2B%20vision-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-Rekognition-FF9900?style=flat-square&logo=amazonaws&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-Ubuntu-E95420?style=flat-square&logo=ubuntu&logoColor=white)
![Status](https://img.shields.io/badge/status-live%20in%20production-2ea44f?style=flat-square)

[**Overview**](#-overview) ·
[**Screenshots**](#-screenshots) ·
[**The problem**](#-the-problem-it-solves) ·
[**Architecture**](#-architecture) ·
[**Live view**](#-reading-the-live-view) ·
[**Design decisions**](#-design-decisions) ·
[**Configuration**](#-configuration) ·
[**Add a store**](#-adding-a-new-store) ·
[**Production**](#-running-in-production) ·
[**Quick start**](#-quick-start) ·
[**FAQ**](#-faq)

<br/>

<table>
<tr>
<td align="center" width="25%"><h2>45+</h2>store locations<br/>live</td>
<td align="center" width="25%"><h2>99.9%</h2>uptime</td>
<td align="center" width="25%"><h2>~$4</h2>AWS cost per<br/>store / month</td>
<td align="center" width="25%"><h2>2</h2>max uploads<br/>per visitor</td>
</tr>
</table>

</div>

---

## 📌 Overview

**Visilytics** is a distributed computer vision system deployed across **45+ Ashley HomeStore retail locations** in the United States (client: **iConnect Group US**). It connects to the IP cameras already installed at each store entrance, detects customers walking in, discards people who are leaving or facing away, and uploads one clean, front-facing image per visitor to the iConnect analytics platform.

Every store and camera is described by **a single JSON file**, so onboarding a new location is a configuration change — not a code change.

| | |
|---|---|
| 🎯 **Goal** | Accurate, de-duplicated visitor counts and photos from existing CCTV — no new hardware |
| 🧠 **Detection** | YOLOv8 + BoT-SORT tracking, run **locally** (person class only) |
| ☁️ **AWS's role** | Rekognition is called **once per candidate person**, after tracking and filtering — never per frame |
| 💸 **Cost** | ~**$4 / month / store** in AWS spend |
| 🔌 **Integration** | REST upload to the client's backend, with a local backup queue and automatic retry |
| 🟢 **Status** | Live in production |

---

## 📸 Screenshots

> 🔒 All customer faces are blurred. Screenshots come from production stores; overlay labels vary slightly between builds (for example `PENDING`).

<table>
<tr>
<td width="50%" valign="top">
<img src="docs/images/live-detection-two-cameras.png" alt="Live detection window showing two entrance cameras with tracked people, red door-threshold line and upload/filter counters"/>
<br/><sub><b>Two entrance cameras, one store.</b> Each tracked person gets an ID and a status; the red line is the door threshold; the top bar shows door state, uploads and filtered detections.</sub>
</td>
<td width="50%" valign="top">
<img src="docs/images/live-detection-tracking.png" alt="Live detection window with two people tracked and labelled with IDs and track state"/>
<br/><sub><b>Multi-person tracking.</b> BoT-SORT keeps a stable ID per person across frames, so each visitor is captured once — not once per frame.</sub>
</td>
</tr>
</table>

<div align="center">
<img src="docs/images/client-portal-captures.png" alt="Client portal grid of captured visitor images with faces blurred" width="92%"/>
<br/><sub><b>The output.</b> Cropped visitor captures landing in the client portal, ready for review, merging and reporting.</sub>
</div>

---

## 🧭 The problem it solves

Counting people at a store entrance sounds simple until it meets real cameras and real traffic.

| Challenge | What goes wrong | How Visilytics handles it |
|---|---|---|
| 💸 **Cloud vision is billed per call** | Analysing every frame from every camera gets expensive fast | Detection and tracking run locally; AWS is called once per already-filtered candidate |
| 🔁 **One person, many frames** | A visitor is visible for dozens of frames and gets counted again and again | Persistent track IDs, a best-frame window and an upload cap per person |
| 🔙 **Backs of heads and people leaving** | Useless photos, inflated counts | Direction filter plus an AWS face-angle check |
| 📷 **Every camera is different** | Angle, height, door position and lighting change from store to store | Per-camera JSON: ROI, door line, orientation flag, model path |
| 🌐 **Networks and cameras drop** | Streams freeze, uploads fail | Automatic stream reconnect, local backup queue, hourly retry |
| 🚪 **Empty stores waste compute** | Running detection on a quiet entrance all day | Motion gate skips frames when nothing is happening at the door |

---

## 🧩 Architecture

```mermaid
flowchart TD
    A["📹 RTSP camera<br/>(store entrance)"] --> B["🖼️ Frame capture<br/>OpenCV · auto-reconnect"]
    B --> C["🚪 Door motion gate<br/>skip idle frames"]
    C --> D["🧠 YOLOv8 + BoT-SORT<br/>detect and track people"]
    D --> E{"🧭 Moving toward<br/>the exit?"}
    E -- yes --> X["⏭️ Skip<br/>no back-of-head shots"]
    E -- no --> F["📏 Position + quality filters<br/>door line · size · frame edges"]
    F --> G["⏱️ Best-frame selection<br/>~1 s window, up to 10 candidates"]
    G --> H["☁️ AWS Rekognition<br/>1 call per candidate<br/>reject extreme back / look-down"]
    H --> I["📤 Upload to client API<br/>image + visitor record"]
    I -- "on failure" --> J[("💾 Local backup<br/>hourly auto-retry")]
    J -.-> I
```

### The pipeline, stage by stage

| # | Stage | What it does | Why it matters |
|---|-------|--------------|----------------|
| 1 | 📹 **Capture** | Reads the RTSP stream with OpenCV; if a read fails the capture is released and re-opened automatically | Cameras drop — the system heals without a restart |
| 2 | 🚪 **Door motion gate** | Frame-differences a thin strip across the door; detection runs only while there is motion (plus a 4 s cool-down) | Saves compute; ignores empty-store frames |
| 3 | 🧠 **Detect + track** | YOLOv8 (`classes=0`, conf ≥ 0.35) with BoT-SORT gives every person a persistent ID | Stable IDs are what make de-duplication possible |
| 4 | 🧭 **Direction filter** | Tracks each ID's vertical movement (≈5 px average, 3-frame consistency); people heading for the exit are skipped | No photos of people leaving |
| 5 | 📏 **Position + quality** | Feet must be on the *inside* of the door line, box large enough, and inside the central 80% of the frame | Filters partial, distant and edge-of-frame detections |
| 6 | ⏱️ **Best frame** | Collects candidates for ~1 s and keeps the best one | One good photo instead of a burst of blurry ones |
| 7 | ☁️ **AWS pose check** | Rekognition `DetectFaces` on the cropped person: rejects extreme back-of-head (yaw > 70°) or looking-down (pitch > 85°) shots; *no face detected* is accepted | Lenient by design — missing a customer is worse than an imperfect photo |
| 8 | 📤 **Upload** | Resizes to 200×400, uploads the image and a visitor record; **max 2 uploads per person**, counter resets after 60 s | Prevents duplicate counts |
| 9 | 💾 **Backup + retry** | Failed uploads are written to `image_backup/`; a background thread retries **every hour** and prunes after 15 days | A network blip never silently loses a visitor |

### 🚶 One visitor's journey

```mermaid
sequenceDiagram
    autonumber
    participant Cam as 📹 Camera
    participant Eng as 🖥️ Visilytics engine
    participant AWS as ☁️ AWS Rekognition
    participant API as 🏢 Client API
    participant Disk as 💾 Backup queue

    Cam->>Eng: video frames over RTSP
    Eng->>Eng: door motion gate, YOLOv8 + BoT-SORT, filters
    Note over Eng: person tracked - about 1 s of candidate frames collected
    Eng->>Eng: pick the best frame
    Eng->>AWS: DetectFaces - one call on the cropped person
    AWS-->>Eng: face angle and quality
    alt extreme back-of-head or looking down
        Eng->>Eng: reject - nothing is uploaded
    else accepted - including no face found
        Eng->>API: upload image and visitor record
        alt upload fails
            Eng->>Disk: save image locally
            Disk-->>API: retry every hour
        end
    end
```

### 🔄 The life of a tracked person

```mermaid
stateDiagram-v2
    [*] --> NEW: person first tracked
    NEW --> TRACK: passes position and quality filters
    TRACK --> DONE_1: about 1 s later - best frame, AWS check, upload
    DONE_1 --> TRACK: still in view and upload budget left
    TRACK --> DONE_2: second upload
    DONE_2 --> [*]: budget used up
    NEW --> EXIT: moving toward the exit
    TRACK --> EXIT: moving toward the exit
    EXIT --> [*]: skipped and never uploaded

    NEW : Green box - NEW
    TRACK : Yellow box - TRACK n
    DONE_1 : Blue box - DONE 1 of 2
    DONE_2 : Blue box - DONE 2 of 2
    EXIT : Red box - EXIT
```

---

## ✨ Key features

- 🧠 **Local YOLOv8 + BoT-SORT tracking** on live RTSP streams — no cloud round-trip for detection
- 💸 **Cost-aware AWS usage** — Rekognition is a single, targeted call per candidate, not a per-frame service
- 🧾 **Config-driven fleet** — one JSON per store/camera; camera angle, door line, ROI and model path are all data
- 🧭 **Direction awareness** — entering vs. exiting, with a per-camera switch for cameras mounted the other way round
- ⏱️ **Best-frame capture** — a short observation window picks the strongest image per person
- 🔁 **Duplicate control** — persistent track IDs plus an upload cap per person
- 🚪 **Door-aware processing** — a motion gate keeps the pipeline idle when nothing is happening
- 💾 **Resilient delivery** — local backup queue, hourly retry, automatic stream reconnection
- 🖥️ **Live operator view** — annotated window with track state, door threshold and running counters

---

## 👀 Reading the live view

<table>
<tr>
<td width="46%" valign="top">
<img src="docs/images/overlay-schematic.png" alt="Schematic of the detection overlay: person box with confidence, blurred face, entering arrow and status line"/>
<br/><sub><i>Illustrative schematic — no real footage.</i></sub>
</td>
<td width="54%" valign="top">

| Overlay | Meaning |
|---|---|
| 🟩 **Green** box · `NEW` | Newly tracked, nothing collected yet |
| 🟨 **Yellow** box · `TRACK:n` | Collecting candidate frames (*n* so far) |
| 🟦 **Blue** box · `DONE (x/2)` | Already uploaded *x* of the max 2 |
| 🟥 **Red** box · `EXIT` | Moving toward the exit → skipped |
| ➖ **Red horizontal line** | The **door threshold** (see below) |
| 🔢 **Top bar** | `Door: OPEN/CLOSED · Uploaded · Filtered · AWS` |

</td>
</tr>
</table>

**The red line is functional, not decorative.** It is drawn at `door_sy_ratio` from the camera's config, and the quality filter uses the *same* value: a person only counts once their **feet** have crossed to the inside of the line. Cameras mounted the opposite way set `"invert_door_threshold": true`, which flips both the rule and the on-screen label (`must be below` ↔ `must be above`).

---

## 🧠 Design decisions

| Decision | Why |
|---|---|
| **Detect locally, not in the cloud** | No per-frame API bill and no network round-trip; only cropped candidates ever leave the store |
| **Call AWS once, after tracking and filtering** | By then only a handful of real candidates remain, which is what keeps spend near $4 per store per month |
| **Lenient AWS thresholds** | Only *extreme* angles are rejected (yaw > 70°, pitch > 85°) and a missing face is accepted — a lost customer costs more than an imperfect photo |
| **Motion-gate the door** | The pipeline stays idle on an empty entrance instead of running detection all day |
| **Process every frame** | Frame skipping is off (`process_every_n_frames = 1`) so a quick walk-through is never missed; the motion gate keeps that affordable |
| **Judge position by the feet** | A person "inside" the store is decided by the bottom of the box crossing the door line, with a head check when very close to it |
| **Best frame over first frame** | A ~1 s observation window yields a sharper, better-framed image |
| **Track IDs plus an upload cap** | Stable IDs and a maximum of two uploads per person (reset after 60 s) stop one visitor becoming ten counts |
| **One JSON per camera** | Everything that differs between sites is data, so the same code runs unchanged everywhere |
| **Local backup + hourly retry** | Uploads survive outages; nothing is silently dropped |

---

## 🔧 Configuration

Each camera is described by one JSON file. Start from a template:

| Template | Use it for |
|---|---|
| [`config_template_standard.json`](config_template_standard.json) | Cameras at a normal distance from the entrance |
| [`config_template_close_camera.json`](config_template_close_camera.json) | Cameras very close to the door (higher door line, stricter motion threshold) |

The engine reads this minimum:

```json
{
  "store_info": {
    "store_name": "STORE_NAME",
    "client_code": "CLIENT_CODE",
    "company_id": "COMPANY_ID",
    "store_id": "STORE_ID"
  },
  "cameras": [{
    "camera_id": "1",
    "camera_name": "Main Entrance",
    "rtsp_url": "rtsp://admin:YOUR_PASSWORD@192.168.1.100:554",
    "roi": { "start_width": 0.1, "end_width": 0.9, "start_height": 0.2, "end_height": 0.8 },
    "door_detection": {
      "door_sx_ratio": 0.0, "door_ex_ratio": 1.0,
      "door_sy_ratio": 0.5, "door_ey_ratio": 0.8,
      "invert_door_threshold": false
    }
  }],
  "processing": { "yolo_model_path": "FinalModel.pt", "use_cuda": true }
}
```

| Block | Controls |
|---|---|
| `store_info` | Routes every upload to the right store record in the client's backend |
| `cameras[].rtsp_url` | The stream to read |
| `cameras[].roi` | Crop of the frame to analyse (fractions of width / height) |
| `cameras[].door_detection` | Where the door strip and threshold line sit, and which side counts as "inside" |
| `processing` | Model weights and whether to use the GPU |

<details>
<summary><b>🎛️ Tuning constants and troubleshooting</b> (click to expand)</summary>

<br/>

Behavioural constants live at the top of `LiveDetectionSystem.__init__` in [`live_detection_aws.py`](live_detection_aws.py):

| Setting | Default | Effect |
|---|---|---|
| Detection confidence | `0.35` | Deliberately low — biased towards catching everyone |
| Minimum body size | `50 × 25 px` | Ignores tiny / distant detections |
| Best-frame window | `1.0 s`, up to `10` frames | How long each person is observed before upload |
| Uploads per person | `2`, reset after `60 s` | Duplicate control |
| Door motion threshold | `800` changed pixels, `4 s` cool-down | How much motion counts as "door active" |
| Exit detection | `5 px` average, `3`-frame consistency | Direction filter sensitivity |
| AWS thresholds | yaw `70°`, pitch `85°` | Only extreme angles are rejected |
| Backup retention | `15 days`, retry hourly | Failed-upload queue |

| Symptom | First thing to try |
|---|---|
| ❌ People are being missed | Move the door line (`door_sy_ratio`) so it sits where people actually cross; check the ROI |
| ❌ Too many duplicates | Lower the per-person upload cap or lengthen the best-frame window |
| ❌ Door never reads `OPEN` | Widen the door strip in the config or lower the motion threshold |
| ❌ Grey / frozen window | Check camera connectivity and test the RTSP URL in VLC |
| 📏 Camera very close to the door | Use the close-camera template and see [`README_AWS.md`](README_AWS.md) for the code-level adjustments |

</details>

---

## ➕ Adding a new store

No code changes — just a config file and a process per camera.

1. 📄 **Copy a template** → `config_<store>_cam<N>.json` (use the close-camera template if the camera is right by the door).
2. 🏷️ **Fill in `store_info`** — these IDs decide which store record the uploads land in.
3. 📹 **Set `rtsp_url`** and confirm the stream plays in VLC first.
4. 🚪 **Place the door line.** Run it and watch the red line: set `door_sy_ratio` where people cross as they step inside. If the camera looks the other way, set `invert_door_threshold` to `true`.
5. ✂️ **Trim the ROI** to the entrance area so the rest of the store is ignored.
6. 📊 **Watch the counters** (`Uploaded` vs. `Filtered`) for a few minutes and adjust the door line if real visitors are being filtered.
7. 🔁 **Install it as a service** (below). A store with two cameras simply runs two processes with two configs.

---

## 💻 Running in production

The engine is **one process per camera**, so a two-camera store runs two services, each pointed at its own config. On Linux a `systemd` unit keeps each one alive across crashes and reboots:

```ini
# /etc/systemd/system/visilytics-store01-cam1.service   (example — adapt the paths)
[Unit]
Description=Visilytics - store01 camera 1
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/opt/visilytics
EnvironmentFile=/etc/visilytics/aws.env        # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
ExecStart=/opt/visilytics/venv/bin/python live_detection_aws.py /opt/visilytics/configs/store01_cam1.json
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now visilytics-store01-cam1     # start now and on every boot
sudo systemctl status visilytics-store01-cam1           # is it running?
journalctl -u visilytics-store01-cam1 -f                # follow the logs
```

> 💡 The engine draws a live preview window (`cv2.imshow`) for operators. On a machine with no display, run it under a virtual display, for example `xvfb-run -a python live_detection_aws.py <config>`.

**What to watch:** the `Stats` log line (every 300 frames) reports detected / uploaded / filtered / AWS-rejected counts, and `image_backup/` should stay empty — files piling up there means uploads are failing.

---

## 🚀 Quick start

```bash
git clone https://github.com/Mushibhai47/Visilytics-python.git
cd Visilytics-python

python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# AWS credentials come from the environment — never hardcode them
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Describe your camera, then run it
cp config_template_standard.json config_mystore_cam1.json
python live_detection_aws.py config_mystore_cam1.json      # press q in the window to quit
```

**Requirements:** Python 3, an RTSP camera, YOLOv8 weights that detect people (a COCO `yolov8n.pt` is enough to try; production uses weights tuned per store), an NVIDIA GPU with CUDA for real-time use (it falls back to CPU otherwise), and an AWS account with Rekognition access.

> ℹ️ **Not included in this repository:** the client-specific integration modules the engine imports (the upload API wrapper, the local face-analysis client and a small resize helper) and the production model weights. They are tied to the client's backend and network, so you will need to provide your own equivalents to run the full pipeline end to end.

---

## 📊 Deployment at a glance

| Metric | Value |
|--------|-------|
| 🏬 Active stores | **45+** |
| 🗺️ States | TX, VA, NY, NM, AZ, OH and more |
| 📷 Cameras per store | 1–2 |
| ⏱️ Uptime | **99.9%** |
| 💸 AWS cost | **~$4 / month / store** |
| 👤 Uploads per visitor | max **2** (counter resets after 60 s) |

---

## 🧰 Tech stack

| Layer | Technology |
|-------|------------|
| Language | Python |
| Detection + tracking | Ultralytics YOLOv8 · BoT-SORT |
| Video I/O + vision | OpenCV (RTSP capture, motion gate, overlay) |
| Runtime | PyTorch · CUDA |
| Cloud | AWS Rekognition (`boto3`) |
| Delivery | REST upload · local backup queue · background retry thread |
| Ops | Linux (Ubuntu) · one process per camera · systemd · JSON configuration |

---

## 🗂️ Project structure

```
Visilytics-python/
├── live_detection_aws.py              # Detection engine (LiveDetectionSystem)
├── config_template_standard.json      # Config template: normal-distance camera
├── config_template_close_camera.json  # Config template: camera very close to the door
├── requirements.txt                   # Python dependencies
├── README_AWS.md                      # Operations notes: tuning, backups, troubleshooting
└── docs/images/                       # Architecture diagram + anonymised screenshots
```

<details>
<summary><b>🔎 Where to look in the code</b> (click to expand)</summary>

<br/>

| Function | Responsibility |
|---|---|
| `check_door_open()` | Motion gate over the door strip |
| `is_good_quality_detection()` | Size, door-line position and frame-edge filtering |
| `is_person_exiting()` | Direction filter from recent vertical movement |
| `should_upload()` | Per-person upload cap and 60 s reset |
| `add_frame_candidate()` / `get_best_frame()` | ~1 s best-frame collection and selection |
| `detect_face_with_aws()` | The single Rekognition `DetectFaces` call |
| `process_and_upload()` | Orchestrates filter → AWS check → resize → upload → backup |
| `retry_backup_uploads()` | Hourly background retry of failed uploads |
| `run()` | Main loop: read frame → gate → track → draw overlay → reconnect on failure |

</details>

---

## ❓ FAQ

<details>
<summary><b>Why isn't AWS doing the detection?</b></summary>

<br/>

Calling a cloud vision API on every frame from every camera is slow (network round-trip) and billed per call. Visilytics detects and tracks people locally with YOLOv8 + BoT-SORT and only sends AWS a single cropped image per candidate, after it has already passed direction, position and quality checks.

</details>

<details>
<summary><b>Why does it accept people when no face is detected?</b></summary>

<br/>

By design. Someone looking at their phone, wearing a hat or caught side-on still counts as a visitor. Only clearly bad angles (yaw > 70° or pitch > 85°) are rejected, because a missed customer is worse than an imperfect photo.

</details>

<details>
<summary><b>How are duplicate counts avoided?</b></summary>

<br/>

BoT-SORT gives each person a persistent ID, a ~1 s window selects one best frame, and a person can be uploaded at most twice (the counter resets after 60 s). People moving toward the exit are skipped entirely.

</details>

<details>
<summary><b>Can I use my own model?</b></summary>

<br/>

Yes. Point `processing.yolo_model_path` at any YOLOv8 weights that detect people (class `0`). Production uses weights tuned for each store's camera angle and lighting; a stock COCO model works for trying it out.

</details>

<details>
<summary><b>Does it need a GPU?</b></summary>

<br/>

A GPU is recommended for real-time use. With `"use_cuda": true` the model moves to CUDA when it is available; otherwise it runs on the CPU.

</details>

<details>
<summary><b>What about stores with more than one camera?</b></summary>

<br/>

Each camera gets its own config file and its own process. Stores in this deployment run one or two.

</details>

<details>
<summary><b>What happens when the network or a camera drops?</b></summary>

<br/>

If a frame read fails the capture is released and re-opened automatically. If an upload fails the image is saved to `image_backup/` and retried every hour by a background thread.

</details>

---

## 🔐 Privacy & security

- Screenshots in this repository have faces blurred; no raw footage is committed.
- Only a **cropped image of each accepted visitor** leaves the store, sent to the client's own analytics platform.
- Camera credentials and AWS keys are supplied through config files and environment variables, and are never committed. Config files with real credentials are kept out of the repository.

> 📎 This repository is shared as a portfolio piece. Client integrations and production weights are intentionally not included.

---

## 👨‍💻 Developer

**Musharaf Shah** — Software Systems Engineer<br/>
📧 musharafshah476@gmail.com<br/>
🗓️ Sept 2025 – May 2026

<div align="center">
<sub>Built with Python · YOLOv8 · OpenCV · AWS Rekognition</sub>
</div>
