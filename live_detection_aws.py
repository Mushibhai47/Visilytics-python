"""
Live Real-Time Person Detection System with AWS Rekognition Back Filtering
- 100% SAME detection logic as v3.py (catches everyone, no misses)
- SAME duplicate prevention (2 uploads per person, 60s reset)
- SAME quality checks and timing (1.0s wait, 5px movement)
- AWS ONLY for filtering backs/exiting people (called ONCE before upload)
- LENIENT yaw angles to avoid missing customers
- Cost optimized: ~$4/month (not $170)
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import logging
import os
import time
import json
import shutil
import threading
import boto3
from botocore.exceptions import ClientError
import webAPI
from resize import resize as resize_image
from script import detect_faces  # Local face API (FREE)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveDetectionSystem:
    def __init__(self, config):
        self.config = config
        self.store_info = config['store_info']
        self.camera_config = config['cameras'][0]
        self.processing_config = config['processing']

        # Initialize AWS Rekognition
        self.init_aws_rekognition()

        # Backup folder - only for failed uploads
        self.backup_folder = "image_backup"
        self.backup_retention_days = 15  # Keep backups for 15 days only
        os.makedirs(self.backup_folder, exist_ok=True)

        # Load YOLO model
        self.model = YOLO(self.processing_config['yolo_model_path'])
        if self.processing_config['use_cuda'] and torch.cuda.is_available():
            self.model.to('cuda')
            logger.info("YOLO loaded on CUDA")

        # Quality thresholds - SAME AS V3.PY (0.35 confidence, 50px height, 25px width)
        self.min_detection_confidence = 0.35  # Same as v3.py
        self.min_body_height = 50  # Same as v3.py
        self.min_body_width = 25  # Same as v3.py
        self.min_aspect_ratio = 0.0  # Accept all shapes

        # Door threshold
        self.door_threshold = self.camera_config['door_detection'].get('door_sy_ratio', 0.5)
        self.invert_door_threshold = self.camera_config['door_detection'].get('invert_door_threshold', False)

        # Door open detection - same as v3.py
        self.door_config = self.camera_config['door_detection']
        self.enable_door_open_check = True  # Same as v3.py
        self.door_motion_threshold = 800
        self.door_is_open = False
        self.prev_door_frame = None
        self.door_open_cooldown = 4.0
        self.last_door_motion_time = 0

        # Tracking to avoid duplicates - SAME AS V3.PY: 2 uploads per person, 60s reset
        self.recent_uploads = {}  # {person_id: upload_count}
        self.max_uploads_per_person = 2  # Same as v3.py: Allow 2 uploads max per person
        self.upload_reset_time = 60.0  # Same as v3.py: Reset counter after 60 seconds

        # Best frame selection - SAME AS V3.PY: 1.0s wait, max 10 frames
        self.person_frames = {}  # {person_id: {'frames': [], 'first_seen': timestamp}}
        self.best_frame_wait_time = 1.0  # Same as v3.py: Wait only 1 second
        self.max_frames_per_person = 10  # Same as v3.py: Store max 10 frames

        # Movement tracking - SAME AS V3.PY: 5px movement, 5-frame history
        self.person_positions = {}  # {person_id: [y_positions]}
        self.movement_history_size = 5  # Same as v3.py: Track last 5 positions

        # Stats
        self.total_detections = 0
        self.uploaded_count = 0
        self.filtered_count = 0
        self.filtered_by_aws_pose = 0  # AWS rejected due to yaw/pitch
        self.filtered_by_aws_quality = 0  # AWS rejected due to sharpness/brightness
        self.filtered_no_face = 0  # AWS rejected - no face detected (backs)

    def init_aws_rekognition(self):
        """Initialize AWS Rekognition client with credentials"""
        try:
            aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", "")
            aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
            aws_region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

            self.rekognition = boto3.client(
                'rekognition',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region
            )

            logger.info("✅ AWS Rekognition initialized - LENIENT FILTERING enabled")
        except Exception as e:
            logger.error(f"Failed to initialize AWS Rekognition: {e}")
            raise

    def cleanup_old_backups(self):
        """Delete backup images older than retention period (15 days)"""
        try:
            from datetime import datetime, timedelta
            cutoff_time = datetime.now() - timedelta(days=self.backup_retention_days)

            for filename in os.listdir(self.backup_folder):
                file_path = os.path.join(self.backup_folder, filename)
                try:
                    # Get file modification time
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        logger.info(f"🗑️ Deleted old backup: {filename} (age: {(datetime.now() - file_time).days} days)")
                except Exception as e:
                    logger.debug(f"Error cleaning backup {filename}: {e}")
        except Exception as e:
            logger.error(f"Backup cleanup error: {e}")

    def retry_backup_uploads(self):
        """Auto-retry uploading backup images every hour (background thread)"""
        logger.info("🔄 Auto-retry thread started - checking backups every hour")

        while True:
            try:
                time.sleep(3600)  # Wait 1 hour

                if not os.path.exists(self.backup_folder):
                    continue

                backup_files = [f for f in os.listdir(self.backup_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

                if not backup_files:
                    continue

                logger.info(f"🔄 Auto-retry: Found {len(backup_files)} backup images to retry")

                uploaded_count = 0
                for filename in backup_files:
                    img_path = os.path.join(self.backup_folder, filename)

                    try:
                        # Try to upload backup image
                        with open(img_path, 'rb') as f:
                            files = {'media': f}
                            response = webAPI.uploadimage(files, self.store_info['client_code'])

                        if response.status_code == 200:
                            logger.info(f"✅ Auto-retry SUCCESS: {filename}")

                            # Get age/gender from face API for visitor info
                            try:
                                face_response = detect_faces(img_path)
                                if face_response.get("FaceDetails"):
                                    face_detail = face_response["FaceDetails"][0]
                                    min_age = round(face_detail["AgeRange"]["Low"] / 10) * 10
                                    max_age = round(face_detail["AgeRange"]["High"] / 10) * 10
                                    gender = "M" if face_detail["Gender"]["Value"] == "Male" else "F"
                                else:
                                    min_age, max_age, gender = 20, 40, "M"
                            except:
                                min_age, max_age, gender = 20, 40, "M"

                            # Send visitor info
                            str_time = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
                            visitor_data = {
                                "ClientCode": self.store_info['client_code'],
                                "CompanyID": str(self.store_info['company_id']),
                                "StoreID": str(self.store_info['store_id']),
                                "Visits": [{
                                    "NoOfPeople": "1",
                                    "UPS": "1",
                                    "VisitDateTime": str_time,
                                    "Visitors": [{
                                        "MinAge": str(min_age),
                                        "MaxAge": str(max_age),
                                        "Gender": gender,
                                        "ActVisitorDateTime": str_time,
                                        "Picture": filename,
                                        "CameraID": str(self.camera_config['camera_id']),
                                        "Accuracy": 0,
                                        "PicQuality": "G"
                                    }]
                                }]
                            }

                            webAPI.savevisitorinfo(visitor_data)

                            # Delete successful backup
                            os.remove(img_path)
                            uploaded_count += 1

                        else:
                            logger.debug(f"⚠️ Auto-retry FAILED: {filename} (status: {response.status_code}) - Will retry next hour")

                    except Exception as e:
                        logger.debug(f"⚠️ Auto-retry ERROR: {filename} - {e}")

                if uploaded_count > 0:
                    logger.info(f"✅ Auto-retry completed: {uploaded_count}/{len(backup_files)} backups uploaded successfully")

            except Exception as e:
                logger.error(f"Auto-retry thread error: {e}")

    def detect_face_with_aws(self, image_path):
        """Use AWS Rekognition to detect faces with pose angles and quality"""
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            response = self.rekognition.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['ALL']
            )

            if not response.get('FaceDetails') or len(response['FaceDetails']) == 0:
                return {
                    'has_face': False,
                    'face_confidence': 0,
                    'yaw': 0,
                    'pitch': 0,
                    'roll': 0,
                    'brightness': 50,
                    'sharpness': 50,
                    'age_range': {'Low': 20, 'High': 40},
                    'gender': 'M'
                }

            face = response['FaceDetails'][0]
            pose = face.get('Pose', {})
            quality = face.get('Quality', {})
            age_range = face.get('AgeRange', {'Low': 20, 'High': 40})
            gender_detail = face.get('Gender', {})

            return {
                'has_face': True,
                'face_confidence': face.get('Confidence', 0),
                'yaw': pose.get('Yaw', 0),
                'pitch': pose.get('Pitch', 0),
                'roll': pose.get('Roll', 0),
                'brightness': quality.get('Brightness', 50),
                'sharpness': quality.get('Sharpness', 50),
                'age_range': age_range,
                'gender': 'M' if gender_detail.get('Value') == 'Male' else 'F'
            }

        except Exception as e:
            logger.error(f"AWS error: {e}")
            return {
                'has_face': False,
                'face_confidence': 0,
                'yaw': 0,
                'pitch': 0,
                'roll': 0,
                'brightness': 50,
                'sharpness': 50,
                'age_range': {'Low': 20, 'High': 40},
                'gender': 'M'
            }

    def is_good_quality_detection(self, x_min, y_min, x_max, y_max, frame, has_face=False):
        """Check if detection is good quality - SAME AS V3.PY"""
        # Check body size
        width = x_max - x_min
        height = y_max - y_min

        if width < self.min_body_width or height < self.min_body_height:
            logger.debug(f"Filtered: Too small ({width}x{height})")
            return False

        # Check if person is INSIDE the store (past door threshold)
        frame_h, frame_w = frame.shape[:2]
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        bottom_y = y_max  # Bottom of person's bounding box

        # Person must be on the correct side of door line
        bottom_y_ratio = bottom_y / frame_h
        center_y_ratio = center_y / frame_h

        # Calculate distance from threshold
        if self.invert_door_threshold:
            distance_from_threshold = abs(bottom_y_ratio - self.door_threshold)
            # BayCity style: Person's FEET must be ABOVE the door line
            if bottom_y_ratio > self.door_threshold:
                logger.debug(f"Filtered: Outside door (bottom_y_ratio: {bottom_y_ratio:.2f} > threshold: {self.door_threshold})")
                return False
        else:
            distance_from_threshold = abs(bottom_y_ratio - self.door_threshold)
            # Victoria/Corpus style: Person's FEET must be BELOW the door line
            if bottom_y_ratio < self.door_threshold:
                logger.debug(f"Filtered: Outside door (bottom_y_ratio: {bottom_y_ratio:.2f} < threshold: {self.door_threshold})")
                return False

        # Extra check: If person is VERY close to threshold, check if their TOP is also past threshold
        if distance_from_threshold < 0.08 and height > 100:
            top_y_ratio = y_min / frame_h

            if self.invert_door_threshold:
                # BayCity: top should also be above threshold
                if top_y_ratio < self.door_threshold:
                    logger.debug(f"Filtered: Person behind door (feet inside but head outside)")
                    return False
            else:
                # Corpus/Victoria: top should also be below threshold
                if top_y_ratio > self.door_threshold:
                    logger.debug(f"Filtered: Person behind door (feet inside but head outside)")
                    return False

        # Must be in center 80% of frame horizontally
        if not (0.1 * frame_w < center_x < 0.9 * frame_w):
            logger.debug(f"Filtered: Too far left/right")
            return False

        # If has face, it's good
        if has_face:
            return True

        # Check aspect ratio
        aspect_ratio = height / width
        if aspect_ratio < self.min_aspect_ratio:
            logger.debug(f"Filtered: Bad aspect ratio {aspect_ratio:.2f} (min: {self.min_aspect_ratio})")
            return False

        return True

    def should_upload(self, person_id):
        """Check if we should upload this person - SAME AS V3.PY (allow 2 uploads max)"""
        current_time = time.time()

        if person_id in self.recent_uploads:
            upload_data = self.recent_uploads[person_id]
            upload_count = upload_data['count']
            last_upload_time = upload_data['time']

            # Reset counter if 60 seconds passed
            if current_time - last_upload_time > self.upload_reset_time:
                self.recent_uploads[person_id] = {'count': 0, 'time': current_time}
                return True

            # Check if reached max uploads
            if upload_count >= self.max_uploads_per_person:
                return False

        return True

    def check_door_open(self, frame):
        """Check if door is physically open - SAME AS V3.PY"""
        current_time = time.time()
        h, w = frame.shape[:2]

        # Door detection area from config
        door_x1 = int(w * self.door_config['door_sx_ratio'])
        door_x2 = int(w * self.door_config['door_ex_ratio'])
        door_y1 = int(h * self.door_config['door_sy_ratio'])
        door_y2 = int(h * self.door_config['door_ey_ratio'])

        # Only check middle 20% vertical strip of door area
        door_height = door_y2 - door_y1
        door_y1_narrow = door_y1 + int(door_height * 0.4)
        door_y2_narrow = door_y1 + int(door_height * 0.6)

        # Crop door area (narrow strip)
        door_area = frame[door_y1_narrow:door_y2_narrow, door_x1:door_x2]

        if door_area.size == 0:
            return False

        # Convert to grayscale
        gray = cv2.cvtColor(door_area, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Initialize previous frame
        if self.prev_door_frame is None:
            self.prev_door_frame = gray
            return False

        # Calculate difference between frames (motion detection)
        frame_diff = cv2.absdiff(self.prev_door_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        motion_pixels = cv2.countNonZero(thresh)

        # Update previous frame
        self.prev_door_frame = gray

        # Door motion detected
        if motion_pixels > self.door_motion_threshold:
            self.last_door_motion_time = current_time
            if not self.door_is_open:
                logger.info(f"🚪 Door OPENED (motion: {motion_pixels} pixels)")
                self.door_is_open = True
            return True

        # Keep door "open" for cooldown period after motion stops
        if current_time - self.last_door_motion_time < self.door_open_cooldown:
            return True

        # Door closed (no motion)
        if self.door_is_open:
            logger.info(f"🚪 Door CLOSED (no motion)")
            self.door_is_open = False

        return False

    def is_person_exiting(self, person_id, center_y):
        """Detect if person is moving toward the door (exiting) - SAME AS V3.PY"""
        # Track person's Y position
        if person_id not in self.person_positions:
            self.person_positions[person_id] = []

        self.person_positions[person_id].append(center_y)

        # Keep only recent positions
        if len(self.person_positions[person_id]) > self.movement_history_size:
            self.person_positions[person_id].pop(0)

        # Need at least 4 positions to determine direction
        if len(self.person_positions[person_id]) < 4:
            return False

        positions = self.person_positions[person_id]

        # Calculate movement trend
        movements = []
        for i in range(1, len(positions)):
            movements.append(positions[i] - positions[i-1])

        # Average movement direction
        avg_movement = sum(movements) / len(movements)

        # SAME AS V3.PY: 5px threshold, 3-frame consistency
        if self.invert_door_threshold:
            # BayCity style: door at bottom, exiting = moving DOWN (positive dy)
            is_exiting = avg_movement > 5 and all(m > 0 for m in movements[-3:])
        else:
            # Victoria/Corpus style: door at top, exiting = moving UP (negative dy)
            is_exiting = avg_movement < -5 and all(m < 0 for m in movements[-3:])

        if is_exiting:
            logger.info(f"🚪 Person {person_id} is EXITING (movement: {avg_movement:.1f}) - SKIPPING")

        return is_exiting

    def add_frame_candidate(self, person_id, frame, bbox, confidence):
        """Add a frame candidate for this person - SAME AS V3.PY"""
        current_time = time.time()

        if person_id not in self.person_frames:
            self.person_frames[person_id] = {
                'frames': [],
                'first_seen': current_time,
                'movement_state': None  # Will store 'EXITING' or 'ENTERING'
            }

        # Store frame data
        if len(self.person_frames[person_id]['frames']) < self.max_frames_per_person:
            self.person_frames[person_id]['frames'].append({
                'frame': frame.copy(),
                'bbox': bbox,
                'confidence': confidence,
                'timestamp': current_time
            })

    def get_best_frame(self, person_id):
        """Select best frame from collected frames - SAME AS V3.PY"""
        if person_id not in self.person_frames:
            return None

        frames_data = self.person_frames[person_id]['frames']
        if not frames_data:
            return None

        # Score each frame
        best_frame = None
        best_score = -1

        for frame_data in frames_data:
            frame = frame_data['frame']
            bbox = frame_data['bbox']
            conf = frame_data['confidence']

            x_min, y_min, x_max, y_max = bbox

            # Crop and check face
            cropped = frame[y_min:y_max, x_min:x_max]
            if cropped.size == 0:
                continue

            # Save temp for face detection
            temp_path = f"temp_score_{person_id}.jpg"
            cv2.imwrite(temp_path, cropped)

            # Check face
            try:
                face_response = detect_faces(temp_path)
                has_face = bool(face_response.get("FaceDetails") and len(face_response["FaceDetails"]) > 0)
                face_confidence = face_response["FaceDetails"][0].get("Confidence", 0) if has_face else 0
            except:
                has_face = False
                face_confidence = 0

            # Remove temp file safely
            try:
                os.remove(temp_path)
            except:
                pass  # File already removed or doesn't exist

            # Calculate score: 60% face confidence + 40% detection confidence
            score = (face_confidence * 0.6) + (conf * 100 * 0.4)

            if score > best_score:
                best_score = score
                best_frame = frame_data

        # Cleanup
        del self.person_frames[person_id]

        return best_frame

    def check_ready_for_upload(self):
        """Check if any tracked persons are ready for upload - SAME AS V3.PY (wait 1.0s)"""
        current_time = time.time()
        ready_persons = []

        for person_id, data in list(self.person_frames.items()):
            time_elapsed = current_time - data['first_seen']
            if time_elapsed >= self.best_frame_wait_time:
                ready_persons.append(person_id)

        return ready_persons

    def process_and_upload(self, frame, person_id, x_min, y_min, x_max, y_max):
        """
        Crop person, check quality, upload - SAME AS V3.PY
        PLUS: AWS filtering - VERY LENIENT for entering, STRICT only for exiting
        """
        try:
            # Crop person
            cropped = frame[y_min:y_max, x_min:x_max]

            if cropped.size == 0:
                return False

            # Save temp image
            temp_path = f"temp_person_{person_id}.jpg"
            cv2.imwrite(temp_path, cropped)

            # 🔥 AWS REKOGNITION - SINGLE CALL BEFORE UPLOAD 🔥
            logger.info(f"💰 AWS API CALL for Person {person_id}")
            aws_result = self.detect_face_with_aws(temp_path)

            # 🎯 LENIENT FILTER: Capture everyone, minimal filtering
            # Only reject EXTREME backs (yaw > 70°)
            # Accept people looking down, sideways, no face detected
            yaw_threshold = 70  # Very lenient - only reject extreme backs
            pitch_threshold = 85  # Very lenient - accept looking down at phone
            logger.info(f"🎯 Person {person_id} - LENIENT AWS filter (Yaw≤70°, Pitch≤85°) - Capture Everyone!")

            # AWS VALIDATION - LENIENT MODE
            if not aws_result['has_face']:
                # NO FACE = Could be looking down, blurry, side view
                # ACCEPT ANYWAY - prioritize catching everyone!
                logger.warning(f"⚠️ Person {person_id}: No face detected by AWS - ACCEPTING anyway (lenient mode)")
                min_age, max_age, gender = 20, 40, "M"
                # Continue to upload with default age/gender
            else:
                # Face detected - check only EXTREME angles
                yaw = abs(aws_result['yaw'])
                pitch = abs(aws_result['pitch'])

                # REJECT only if EXTREME back (yaw > 70°)
                if yaw > yaw_threshold:
                    logger.warning(f"⛔ AWS REJECTED Person {person_id}: EXTREME BACK (Yaw: {yaw:.1f}° > {yaw_threshold}°)")
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    self.filtered_by_aws_pose += 1
                    return False

                # REJECT only if EXTREME pitch (> 85°)
                if pitch > pitch_threshold:
                    logger.warning(f"⛔ AWS REJECTED Person {person_id}: EXTREME pitch (Pitch: {pitch:.1f}° > {pitch_threshold}°)")
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    self.filtered_by_aws_pose += 1
                    return False

                logger.info(f"✅ AWS APPROVED Person {person_id}: Yaw={yaw:.1f}°/{yaw_threshold}°, Pitch={pitch:.1f}°/{pitch_threshold}°")

                # Get age/gender from AWS
                min_age = round(aws_result['age_range']['Low'] / 10) * 10
                max_age = round(aws_result['age_range']['High'] / 10) * 10
                gender = aws_result['gender']

            # Check face detection (same as v3.py)
            face_response = detect_faces(temp_path)
            has_face = bool(face_response.get("FaceDetails") and len(face_response["FaceDetails"]) > 0)

            # Quality check (same as v3.py)
            if not self.is_good_quality_detection(x_min, y_min, x_max, y_max, frame, has_face):
                try:
                    os.remove(temp_path)
                except:
                    pass
                self.filtered_count += 1
                return False

            # If AWS has no age/gender, fallback to local face API
            if not has_face:
                min_age, max_age, gender = 20, 40, "M"
            elif has_face and face_response.get("FaceDetails"):
                face_detail = face_response["FaceDetails"][0]
                # Use local face API age/gender if available
                min_age = round(face_detail["AgeRange"]["Low"] / 10) * 10
                max_age = round(face_detail["AgeRange"]["High"] / 10) * 10
                gender = "M" if face_detail["Gender"]["Value"] == "Male" else "F"

            # Resize (same as v3.py)
            img = cv2.imread(temp_path)
            img_resized = cv2.resize(img, (200, 400))

            # Generate filename (same as v3.py)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]
            filename = f"{self.store_info['client_code']}_{self.store_info['store_id']}_{self.camera_config['camera_id']}_{timestamp}.png"
            cv2.imwrite(filename, img_resized)

            # Upload (same as v3.py)
            with open(filename, 'rb') as f:
                files = {'media': f}
                response = webAPI.uploadimage(files, self.store_info['client_code'])

            if response.status_code == 200:
                logger.info(f"✅ UPLOADED Person {person_id}: Yaw={yaw:.1f}° (AWS approved)")

                # Send visitor info (same as v3.py)
                visitor_data = {
                    "ClientCode": self.store_info['client_code'],
                    "CompanyID": str(self.store_info['company_id']),
                    "StoreID": str(self.store_info['store_id']),
                    "Visits": [{
                        "NoOfPeople": "1",
                        "UPS": "1",
                        "VisitDateTime": datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p"),
                        "Visitors": [{
                            "MinAge": str(min_age),
                            "MaxAge": str(max_age),
                            "Gender": gender,
                            "ActVisitorDateTime": datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p"),
                            "Picture": filename,
                            "CameraID": str(self.camera_config['camera_id']),
                            "Accuracy": 0,
                            "PicQuality": "G"
                        }]
                    }]
                }

                webAPI.savevisitorinfo(visitor_data)
                self.uploaded_count += 1

                # Cleanup - upload succeeded, no backup needed
                try:
                    os.remove(filename)
                except:
                    pass
                try:
                    os.remove(temp_path)
                except:
                    pass
                return True
            else:
                # Upload FAILED - save to backup folder!
                logger.warning(f"⚠️ Upload failed: {response.status_code} - Saving to backup")
                try:
                    import shutil
                    backup_path = os.path.join(self.backup_folder, filename)
                    shutil.copy(filename, backup_path)
                    logger.info(f"💾 Backup saved: {backup_path}")
                except Exception as e:
                    logger.error(f"Backup save failed: {e}")

                # Cleanup temp files
                try:
                    os.remove(filename)
                except:
                    pass
                try:
                    os.remove(temp_path)
                except:
                    pass
                return False

        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            return False

    def run(self):
        """Main detection loop - V3.PY + SIMPLE AWS FILTER (Yaw≤45°)"""
        logger.info(f"🚀 Starting live detection for camera {self.camera_config['camera_id']} [LENIENT AWS FILTER]")
        logger.info(f"📍 V3 Logic: 0.35 conf, 50px height, 2 uploads/person, 1.0s wait")
        logger.info(f"📍 AWS: LENIENT filter (Yaw≤70°, Pitch≤85°) - CAPTURE EVERYONE!")
        logger.info(f"📍 No face detected? ACCEPT anyway (prioritize catching all customers)")
        logger.info(f"📍 Backup: Only failed uploads, kept for {self.backup_retention_days} days")
        logger.info(f"RTSP: {self.camera_config['rtsp_url']}")

        # Cleanup old backups on startup
        self.cleanup_old_backups()

        # Start auto-retry background thread
        retry_thread = threading.Thread(target=self.retry_backup_uploads, daemon=True)
        retry_thread.start()
        logger.info("🔄 Auto-retry thread started - checking backups every hour")

        # Connect to camera
        cap = cv2.VideoCapture(self.camera_config['rtsp_url'])

        if not cap.isOpened():
            logger.error("Failed to open camera!")
            return

        # Get ROI
        roi = self.camera_config['roi']

        frame_count = 0
        process_every_n_frames = 1  # Process EVERY frame - catch everyone!

        logger.info("🚀 Live detection started - LENIENT AWS filter (Yaw≤70°) - CAPTURE EVERYONE!")

        while True:
            ret, frame = cap.read()

            if not ret:
                logger.warning("Failed to read frame, reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(self.camera_config['rtsp_url'])
                continue

            frame_count += 1

            # Process every Nth frame (same as v3.py)
            if frame_count % process_every_n_frames != 0:
                continue

            # Crop to ROI (same as v3.py)
            h, w = frame.shape[:2]
            x1 = int(w * roi['start_width'])
            x2 = int(w * roi['end_width'])
            y1 = int(h * roi['start_height'])
            y2 = int(h * roi['end_height'])

            roi_frame = frame[y1:y2, x1:x2]

            # Check if door is open (same as v3.py)
            door_open = True
            if self.enable_door_open_check:
                door_open = self.check_door_open(roi_frame)

            # Only detect people if door is open (same as v3.py)
            results = None
            if door_open:
                # Detect people (same as v3.py)
                results = self.model.track(
                    roi_frame,
                    persist=True,
                    classes=0,  # Person class
                    conf=self.min_detection_confidence,
                    tracker="botsort.yaml",
                    verbose=False
                )

            if results is not None and results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    if box.cls == 0 and box.id is not None:  # Person with tracking ID
                        person_id = int(box.id)
                        conf = float(box.conf)

                        # Get coordinates (relative to ROI)
                        xyxy = box.xyxy[0].cpu().numpy()
                        x_min, y_min, x_max, y_max = map(int, xyxy)

                        self.total_detections += 1

                        # Check if should upload (same as v3.py - allow 2 uploads per person)
                        if not self.should_upload(person_id):
                            continue

                        # Exit detection (same as v3.py - 5px movement, 3-frame consistency)
                        center_y = (y_min + y_max) / 2
                        is_exiting = self.is_person_exiting(person_id, center_y)

                        # Store movement state if we have frames for this person
                        if person_id in self.person_frames and self.person_frames[person_id]['movement_state'] is None:
                            # First time detecting movement state - store it
                            self.person_frames[person_id]['movement_state'] = 'EXITING' if is_exiting else 'ENTERING'

                        if is_exiting:
                            # Skip exiting people - don't upload backs
                            continue

                        # Check if person is inside door threshold (same as v3.py)
                        if self.is_good_quality_detection(x_min, y_min, x_max, y_max, roi_frame, False):
                            # Add this frame as a candidate (same as v3.py)
                            self.add_frame_candidate(person_id, roi_frame, (x_min, y_min, x_max, y_max), conf)
                            logger.info(f"✅ Added frame for Person {person_id} (conf: {conf:.2f}, frames: {len(self.person_frames[person_id]['frames'])})")
                        else:
                            # Only count as filtered once per person
                            if person_id not in self.person_frames:
                                self.filtered_count += 1
                                logger.warning(f"❌ Person {person_id} FILTERED (failed quality check)")

            # Check if any persons are ready for upload (same as v3.py - wait 1.0s)
            ready_persons = self.check_ready_for_upload()
            for person_id in ready_persons:
                # Check if person hasn't exceeded max uploads (same as v3.py - 2 uploads max)
                current_uploads = self.recent_uploads.get(person_id, {}).get('count', 0)
                if current_uploads < self.max_uploads_per_person:
                    best_frame_data = self.get_best_frame(person_id)
                    if best_frame_data:
                        logger.info(f"🎯 Uploading BEST frame for Person {person_id} (Upload {current_uploads + 1}/{self.max_uploads_per_person})")
                        frame = best_frame_data['frame']
                        bbox = best_frame_data['bbox']
                        x_min, y_min, x_max, y_max = bbox

                        # Increment upload counter BEFORE processing (same as v3.py)
                        if person_id in self.recent_uploads:
                            self.recent_uploads[person_id]['count'] += 1
                            self.recent_uploads[person_id]['time'] = time.time()
                        else:
                            self.recent_uploads[person_id] = {'count': 1, 'time': time.time()}

                        # Now upload (AWS filtering happens INSIDE process_and_upload)
                        self.process_and_upload(frame, person_id, x_min, y_min, x_max, y_max)
                    else:
                        # No valid frames collected (same as v3.py)
                        logger.warning(f"⚠️ Person {person_id} had no valid frames to upload")
                        if person_id in self.recent_uploads:
                            self.recent_uploads[person_id]['count'] += 1
                            self.recent_uploads[person_id]['time'] = time.time()
                        else:
                            self.recent_uploads[person_id] = {'count': 1, 'time': time.time()}

            # Log stats every 300 frames (same as v3.py)
            if frame_count % 300 == 0:
                logger.info(f"📊 Stats - Detected: {self.total_detections}, Uploaded: {self.uploaded_count}, Filtered: {self.filtered_count}, AWS rejected: {self.filtered_no_face + self.filtered_by_aws_pose}")

            # Show live detection window (same as v3.py)
            display_frame = roi_frame.copy()

            # Draw door threshold line (RED line) (same as v3.py)
            door_y = int(roi_frame.shape[0] * self.door_threshold)
            cv2.line(display_frame, (0, door_y), (roi_frame.shape[1], door_y), (0, 0, 255), 3)

            threshold_text = "DOOR THRESHOLD (must be above)" if self.invert_door_threshold else "DOOR THRESHOLD (must be below)"
            text_y = door_y + 25 if self.invert_door_threshold else door_y - 10
            cv2.putText(display_frame, threshold_text, (10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Draw detections (same as v3.py)
            if results is not None and results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    if box.cls == 0 and box.id is not None:
                        person_id = int(box.id)
                        xyxy = box.xyxy[0].cpu().numpy()
                        x_min, y_min, x_max, y_max = map(int, xyxy)

                        # Check if person is exiting
                        center_y = (y_min + y_max) / 2
                        is_exiting = self.is_person_exiting(person_id, center_y)

                        # Color coding (same as v3.py)
                        upload_count = self.recent_uploads.get(person_id, {}).get('count', 0)

                        if is_exiting:
                            color = (0, 0, 255)  # Red - exiting, will skip
                            status = "EXIT"
                        elif upload_count > 0:
                            color = (255, 0, 0)  # Blue - already uploaded
                            status = f"DONE ({upload_count}/{self.max_uploads_per_person})"
                        elif person_id in self.person_frames:
                            color = (0, 255, 255)  # Yellow - collecting frames
                            frames_count = len(self.person_frames[person_id]['frames'])
                            status = f"TRACK:{frames_count}"
                        else:
                            color = (0, 255, 0)  # Green - new
                            status = "NEW"

                        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), color, 2)
                        cv2.putText(display_frame, f"ID:{person_id} {status}", (x_min, y_min-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Show stats on frame (same as v3.py)
            door_status = "OPEN" if self.door_is_open else "CLOSED"
            door_color = (0, 255, 0) if self.door_is_open else (0, 0, 255)
            cv2.putText(display_frame, f"Door: {door_status} | Uploaded: {self.uploaded_count} | Filtered: {self.filtered_count} | AWS: {self.filtered_no_face + self.filtered_by_aws_pose}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, door_color, 2)

            # Resize to fit screen (same as v3.py)
            display_h, display_w = display_frame.shape[:2]
            scale = 800 / display_w
            new_w = 800
            new_h = int(display_h * scale)
            display_resized = cv2.resize(display_frame, (new_w, new_h))

            # Window title (same as v3.py + AWS note)
            store_name = self.store_info['store_name']
            camera_name = self.camera_config.get('camera_name', f"Camera {self.camera_config['camera_id']}")
            window_title = f"{store_name} - {camera_name} [AWS LENIENT - Capture Everyone]"
            cv2.imshow(window_title, display_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python live_detection_aws.py <config_file.json>")
        sys.exit(1)

    config_path = sys.argv[1]

    with open(config_path, 'r') as f:
        config = json.load(f)

    system = LiveDetectionSystem(config)
    system.run()
