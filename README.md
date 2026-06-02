# Visilytics — Real-Time Retail Person Detection System

**Developer:** Musharaf Shah | **Client:** iConnect Group US  
**Status:** Live across 45+ store locations  
**Stack:** Python · YOLOv8 · AWS Rekognition · OpenCV · RTSP

---

## Overview

Visilytics is a distributed computer vision system deployed across **45+ Ashley HomeStore retail locations** in the United States. It connects to IP cameras at store entrances, detects customers entering in real time, filters out back-facing people, and uploads clean front-facing visitor images to the iConnect analytics API.

## Architecture

```
RTSP Camera Stream
      ↓
OpenCV Frame Capture
      ↓
YOLOv8 + BotSort Person Tracking
      ↓
Direction Filter → Position Filter → Quality Filter
      ↓
AWS Rekognition Face Detection (front-facing only)
      ↓
REST API Upload → iConnect Backend
      ↓
Backup Queue (auto-retry on failure)
```

## Key Features

- **YOLOv8 + BotSort tracking** on live RTSP camera streams
- **AWS Rekognition** filters back-of-head images before upload
- **Config-driven** — one JSON file per store/camera, no code changes to add new stores
- **Non-blocking uploads** — background worker thread keeps stream smooth
- **99.9% uptime** — auto-reconnection, exponential backoff, hourly backup retry
- **Duplicate prevention** — perceptual hash comparison + per-person upload limits

## Project Structure

```
live_detection_aws.py        Main detection script
deduplication_utils.py       Hash-based duplicate prevention
retry_backup_uploads.py      Auto-retry failed uploads
monitor.py                   System health monitoring
test_setup.py                Verify dependencies before deployment
config_schema.json           Config file documentation/schema
config_template_normal.json  Template for new store setup
requirements.txt             Python dependencies
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set AWS credentials (never hardcode these)
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Verify setup
python test_setup.py

# Run a store
python live_detection_aws.py config_storename_cam1.json
```

## Deployment Scale

| Metric | Value |
|--------|-------|
| Active stores | 45+ |
| States | TX, VA, NY, NM, AZ, OH and more |
| Cameras per store | 1-2 |
| Uptime | 99.9% |
| AWS cost per store | ~$4/month |

## Config Example

```json
{
  "store_name": "Broad Street",
  "rtsp_url": "rtsp://admin:password@192.168.1.100:554",
  "api_url": "https://api.iconnectgroup.com/upload",
  "door_detection": {
    "enable_door_check": true,
    "threshold_opened_pixels": 800
  },
  "detection_params": {
    "min_confidence": 0.35,
    "min_height_pixels": 50,
    "max_uploads_per_person": 2
  }
}
```

## Developer

**Musharaf Shah** — Software Systems Engineer  
shahzeb3303@gmail.com  
Sept 2024 – Present
