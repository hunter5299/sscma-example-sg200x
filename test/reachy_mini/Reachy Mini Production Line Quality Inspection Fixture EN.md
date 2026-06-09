---
description: Deploy a reCamera-based fixture for servo orientation and marking inspection on the Reachy Mini production line.
title: Reachy Mini Production Line Quality Inspection Fixture
keywords:
  - reCamera
  - Reachy Mini
  - Production Line QA
  - Groove Detection
  - OCR
slug: /reachy-mini-production-line-quality-inspection-fixture-en
sidebar_position: 9
last_update:
  date: 04/08/2026
  author: Steven
createdAt: '2026-04-08'
updatedAt: '2026-04-08'
---

# Reachy Mini Production Line Quality Inspection Fixture

## Introduction

reCamera is an AI camera with edge-computing capabilities, supporting ROI, OpenCV image processing, OSD overlays, and YOLO inference. This project targets the Reachy Mini servo assembly line and replaces manual visual inspection with an automated online inspection fixture.

This fixture focuses on two key quality points:

- Groove orientation detection: determine whether the servo arm is mounted in reverse by groove count and shape.
- Text and code verification: read and validate key markings with OCR to prevent wrong-part and missing-marking issues.

After deployment, the line delivers the following capabilities:

- Real-time decisions: per-unit inspection results are returned within station takt time.
- Traceability: each unit keeps a structured inspection record.
- Unified standard: the same rule set is applied across shifts and operators.
- Scalability: additional vision rules can be added later (dimensions, defects, assembly status).

## Hardware Preparation

- One reCamera (this project uses reCamera HQ PoE)
- One host PC (for deployment, debugging, and result receiving)
- Reachy Mini servos and stable lighting
- LAN connectivity (device and host in the same subnet)

On-site setup:

- Lighting: fixed-angle bar light or ring light to avoid strong surface reflections.
- Mounting: rigidly fix camera, part locator, and light source to avoid drift.
- Network: wired connection between device and host to reduce packet loss and latency jitter.
- Power: PoE or stable power input to avoid reboot caused by power fluctuation.

## 1. Project Scope and Line Targets

### 1.1 Applicable Station

This solution is used at the online inspection station after Reachy Mini servo assembly, covering:

- Servo arm orientation check
- Product marking readability and rule consistency check

### 1.2 Target Metrics

- Miss rate (NG misclassified as OK): < 0.005%
- False reject rate (OK misclassified as NG): < 0.005%
- Per-unit detection latency: < 200 ms (excluding manual pick/place)
- Continuous runtime stability: 7x24 operation without crashes

### 1.3 Output Definitions

- OK: unit is allowed to proceed to next process
- NG: unit is blocked and returned for rework
- RECHECK: image quality is insufficient or rules conflict, requiring manual review

## 2. Objectives and Decision Criteria

### 2.1 Business Background

At the Reachy Mini servo assembly station, the historical process relied on manual observation of groove orientation to determine whether mounting was reversed. This caused:

- Misses and misjudgments due to operator fatigue
- Inconsistent criteria across shifts
- Difficulty in structuring and retaining QA records

After introducing the vision fixture, detection runs under a unified rule set and outputs standardized data per unit for subsequent quality analysis and process optimization.

### 2.2 Decision Rules

The project uses the following unified rules:

- Two valid grooves detected: forward orientation (OK), unit proceeds.
- One valid groove detected: reverse orientation (NG), unit is reworked.
- Abnormal groove count or insufficient image quality: RECHECK.

Additional rules:

- If OCR confidence is below threshold (for example 0.80), force RECHECK.
- If groove result conflicts with OCR model/part rule in the same frame, classify as NG.
- Output final decision only after N consecutive consistent frames (for example 3) to reduce jitter.

### 2.3 System Architecture and Data Flow

Project data flow:

1. Camera captures part image.
2. Groove module outputs groove count and bounding boxes.
3. OCR module outputs text, confidence, and rule match result.
4. Decision logic outputs three independent red/green statuses (servo orientation, model, black dot).
5. Results are sent to host and line systems via UDP/MQTT.
6. Host records logs, generates reports, and displays live view.

Unified data fields:

- product_id
- station_id
- timestamp
- groove_count
- groove_score
- ocr_text
- ocr_score
- final_result
- reason_code
- image_path (optional)

## 3. Groove Detection (OpenCV + ROI)

### 3.1 Overview

Groove detection uses a traditional image-processing pipeline within a fixed ROI. Reference project:

- solutions/sesg-project/groove_features
- GitHub: https://github.com/RobotXTeam/sscma-example-sg200x/tree/main/solutions/sesg-project/groove_features

Core flow:

1. Capture camera frame and crop fixed ROI.
2. Convert to grayscale and apply threshold binarization to highlight groove structures.
3. Apply morphological operations to suppress noise and connect targets.
4. Filter contours by area and aspect ratio, then count valid grooves.
5. Output groove count and bounding boxes, and stream in real time via UDP.

### 3.2 Deployment Steps

1. Deploy the groove_features executable on the device.
2. Calibrate ROI according to fixture position (avoid background interference regions).
3. Tune threshold and kernel size for stable detection under reflective surface variation across batches.
4. Start UDP streaming and send results to host for visualization and recording.

Post-deployment on-site self-check:

- Capture 50 consecutive OK samples and verify no false blocks.
- Capture 50 consecutive NG samples and verify stable interception.
- Randomly vary placement angle and verify robustness to minor pose changes.

## 4. OCR Detection (Text and Code Validation)

### 4.1 Overview

OCR detection reads and validates text/code in specified regions. Reference project:

- solutions/cosg-project/ppocr-reader
- GitHub: https://github.com/RobotXTeam/sscma-example-sg200x/tree/main/solutions/cosg-project/ppocr-reader

Typical uses:

- Verify that batch ID, serial number, and model code exist and are readable.
- Compare OCR result with work-order rules and output pass/fail of rule matching.

OCR key points:

- Keep text region as front-facing as possible to reduce perspective distortion.
- Prioritize edge sharpness of characters; avoid blindly increasing exposure.
- For fixed-format text, add regex constraints such as letter+digit length checks.

### 4.2 Deployment Steps

1. Fix camera angle and working distance for the text region in the fixture.
2. Set OCR ROI to ensure sufficient character height and contrast.
3. Tune preprocessing for font type, print depth, and material reflections.
4. Output structured results (text, confidence, rule-match status) for line systems.

Output schema:

- text_raw: raw recognized text
- text_norm: normalized text (spaces removed, case unified)
- confidence: OCR confidence
- rule_match: whether work-order rule is matched
- fail_reason: failure reason (low confidence/format error/not in whitelist)

### 4.3 Linkage with Groove Detection

Online logic is aligned with solutions/vision/factory/feature/factory.py. The UI is not a PASS/NG/RECHECK three-state output, but three independent red/green status bars:

- Row 1 "Servo Orientation": green when groove_det_count == 2, otherwise red.
- Row 2 "Model": red if left or right side has no text for 30 consecutive frames.
- Row 2 "Model": if no-text condition is not triggered, run stabilized linkage validation (StabilizedLogicEvaluator); green when rules pass, otherwise red.
- Row 3 "Black Dot": green when left_dot_exists is True, otherwise red.

Model linkage rules are implemented in evaluate_linked_logic():

- If left side recognizes F, right side must contain XC330.
- If left side recognizes any digit from 1 to 7, right side must contain both XL330 and M288.
- If left side recognizes both R and L, right side must contain M077.
- If none of the above is triggered on the left side, linkage check returns no-check-required.

The host side also provides decision explanation codes for fast troubleshooting, for example:

- E101: groove count anomaly
- E201: low OCR confidence
- E202: OCR text mismatches work-order rules
- E301: image overexposed or underexposed

## 5. Reference Projects

- Groove detection: solutions/sesg-project/groove_features
- OCR detection: solutions/cosg-project/ppocr-reader

## 6. Conclusion

By deploying this reCamera inspection fixture on the Reachy Mini production line, the critical QA step has been upgraded from manual judgment to a standardized, quantifiable, and traceable automated process. The solution balances cost, deployment efficiency, and scalability, and supports expansion from single-station validation to full-line rollout.

Project implementation shows that success depends on the combination of optical conditions, fixture stability, parameter versioning, and closed-loop on-site workflow, rather than model accuracy alone. The team first stabilized one station and then replicated the template to similar stations for low-risk and reusable scale deployment.
