# AWS Person Detection System

AWS Rekognition-powered person detection with automatic back-image filtering.

## Quick Start

```bash
# Install dependencies
pip3 install -r requirements.txt

# Run detection
python3 live_detection_aws.py config_yourstore.json
```

## Features

- AWS Rekognition face filtering (Yaw ≤ 70°, Pitch ≤ 85°)
- Lenient mode: accepts no-face detections to avoid missing customers
- Auto-backup system with hourly retry for failed uploads
- Process every frame (frame_skip = 1) for zero misses
- YOLO v8 + BOTSORT tracking

## Configuration Templates

### Standard Camera:
Use `config_template_standard.json` for normal distance cameras.

### Close Camera:
Use `config_template_close_camera.json` for cameras very close to entrance.
**Note:** Close cameras need code adjustments:
- Line 71: `self.best_frame_wait_time = 0.3`
- Line 79: `self.max_frames_per_person = 15`
- Line 55: `self.min_body_height = 80`

## Key Settings

**Replace in template:**
- `STORE_NAME`: Store name
- `CLIENT_CODE`: Client code (e.g., "lacksvalley", "BerkS")
- `COMPANY_ID`: Company ID
- `STORE_ID`: Store ID
- `CAMERA_ID`: Camera ID
- `PASSWORD`: Camera password
- `IP_ADDRESS:PORT`: Camera RTSP address

**Door Threshold:**
- `door_sy_ratio`: 0.5 = door at 50% height
- `invert_door_threshold`: false = people must be BELOW line

**RTSP Examples:**
- Default port: `rtsp://admin:YOUR_PASSWORD@192.168.1.100`
- Custom port: `rtsp://admin:YOUR_PASSWORD@192.168.1.100:3554`

## AWS Filtering

**Lenient Mode (Current):**
- No face detected → ACCEPT (could be looking down)
- Face detected:
  - Yaw > 70° → REJECT (extreme back)
  - Pitch > 85° → REJECT (extreme down)
  - Otherwise → ACCEPT

## Backup System

Failed uploads saved to `image_backup/` folder.
- Auto-retry every hour
- Keep for 15 days
- Auto-delete after upload success

## Troubleshooting

**Missing People:**
1. Ensure frame_skip = 1 (line 739)
2. For close cameras: use 0.3s wait time
3. Check door threshold position

**Gray Screen:**
- Check camera connectivity
- Verify RTSP URL
- Test with VLC player

## Cost

~$4/month per store (AWS Rekognition API)
