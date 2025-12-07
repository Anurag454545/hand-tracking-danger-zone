"""
Hand Tracking POC for Arvyax Internship Assignment

Features:
- Real-time hand/fingertip tracking without MediaPipe/OpenPose
- Uses classical CV: skin color segmentation + contour analysis
- Virtual object (rectangle) drawn on screen
- Distance-based state logic: SAFE / WARNING / DANGER
- Displays "DANGER DANGER" when in DANGER state
- Aims for >=8 FPS on CPU (use --fast to trade accuracy for speed)

Usage:
- Install dependencies: pip install opencv-python numpy
- Run: python hand_tracking_poc.py --camera 0
- Optional flags: --width, --fast, --mirror

Notes and tips:
- The algorithm assumes a reasonably plain background and that the hand is the largest skin-colored contour.
- You can tune thresholds in the PARAMETERS block below.
- For better robustness, try different lighting or calibrate skin thresholds.

Author: ChatGPT (prototype)
"""

import cv2
import numpy as np
import argparse
import time

# -----------------------------
# PARAMETERS (tweak if needed)
# -----------------------------
DEFAULT_WIDTH = 640
# virtual object parameters (relative to frame size)
OBJ_W_RATIO = 0.2  # object width as fraction of frame width
OBJ_H_RATIO = 0.4  # object height as fraction of frame height
OBJ_MARGIN_RATIO = 0.05  # distance from right edge

# distance thresholds (fractions of frame diagonal) -- dynamic
DANGER_FRAC = 0.03
WARNING_FRAC = 0.12

# morphological kernel
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# -----------------------------
# Helper functions
# -----------------------------

def get_skin_mask(frame_bgr):
    """Return a binary mask where skin-like pixels are white.
    Uses combined HSV and YCrCb heuristics for improved robustness.
    """
    frame_blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)

    # HSV thresholding
    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
    upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # YCrCb thresholding
    ycrcb = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # combine
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    # morphological cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    return mask


def largest_contour(mask, min_area=1000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # choose largest by area
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    return c


def fingertip_from_contour(cnt):
    """Simple fingertip estimation: choose the point with minimum y (top-most) in image coordinates
    and also return convex hull points for visualization.
    This is fast and often aligns with an extended index finger pointing upward.
    """
    # contour points as array
    pts = cnt.reshape(-1, 2)
    # top-most point (smallest y)
    top_idx = np.argmin(pts[:, 1])
    fingertip = tuple(pts[top_idx])
    hull = cv2.convexHull(cnt)
    return fingertip, hull


def distance_point_to_rect(pt, rect):
    """Compute Euclidean distance from point to rectangle (rect as (x,y,w,h)).
    If pt is inside rect, distance is zero.
    """
    x, y, w, h = rect
    px, py = pt
    # compute dx
    dx = max(x - px, 0, px - (x + w))
    dy = max(y - py, 0, py - (y + h))
    return np.hypot(dx, dy)


# -----------------------------
# Main
# -----------------------------

def main(args):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    # For lightweight processing, set a small resolution
    target_w = args.width
    fps_counter = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        # resize keeping aspect ratio
        h, w = frame.shape[:2]
        if w != target_w:
            scale = target_w / float(w)
            frame = cv2.resize(frame, (target_w, int(h * scale)))
            h, w = frame.shape[:2]

        # define virtual object rectangle at right side
        obj_w = int(w * OBJ_W_RATIO)
        obj_h = int(h * OBJ_H_RATIO)
        obj_x = int(w - int(w * OBJ_MARGIN_RATIO) - obj_w)
        obj_y = int((h - obj_h) / 2)
        virtual_rect = (obj_x, obj_y, obj_w, obj_h)

        mask = get_skin_mask(frame)
        cnt = largest_contour(mask, min_area=500)

        fingertip = None
        hull = None
        if cnt is not None:
            fingertip, hull = fingertip_from_contour(cnt)
            # draw contour and hull
            cv2.drawContours(frame, [cnt], -1, (200, 160, 0), 2)
            if hull is not None:
                cv2.drawContours(frame, [hull], -1, (0, 200, 0), 2)
            cv2.circle(frame, fingertip, 8, (255, 0, 255), -1)

        # compute distance metric
        # use frame diagonal to make thresholds scale-invariant
        diag = np.hypot(w, h)
        danger_thresh = DANGER_FRAC * diag
        warning_thresh = WARNING_FRAC * diag

        state = "SAFE"
        if fingertip is None:
            state = "SAFE"
        else:
            dist = distance_point_to_rect(fingertip, virtual_rect)
            # determine state
            if dist <= danger_thresh:
                state = "DANGER"
            elif dist <= warning_thresh:
                state = "WARNING"
            else:
                state = "SAFE"

        # draw virtual rect with color based on state
        color = (0, 255, 0)  # green
        if state == "WARNING":
            color = (0, 200, 255)  # orange
        elif state == "DANGER":
            color = (0, 0, 255)  # red

        x, y, ow, oh = virtual_rect
        thickness = 3
        cv2.rectangle(frame, (x, y), (x + ow, y + oh), color, thickness)

        # labels
        label = f"State: {state}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if state == "DANGER":
            # big red flashing DANGER message
            cv2.putText(frame, "DANGER DANGER", (int(w * 0.1), int(h * 0.6)),
                        cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 4)

        # show mask in small window for debugging
        mask_small = cv2.resize(mask, (int(w * 0.25), int(h * 0.25)))
        mh, mw = mask_small.shape[:2]
        frame[0:mh, w - mw:w] = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)

        # show FPS
        fps_counter += 1
        if time.time() - t0 >= 1.0:
            fps = fps_counter / (time.time() - t0)
            fps_counter = 0
            t0 = time.time()
        try:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except NameError:
            pass

        cv2.imshow("Hand Tracking POC", frame)

        # fast mode: skip some processing cycles to save CPU
        if args.fast:
            key = cv2.waitKey(1)
        else:
            key = cv2.waitKey(1)

        if key & 0xFF == 27:  # Esc to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hand tracking POC")
    parser.add_argument('--camera', type=int, default=0, help='camera device index')
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH, help='target frame width')
    parser.add_argument('--fast', action='store_true', help='enable fast mode (less accurate)')
    parser.add_argument('--mirror', action='store_true', help='mirror camera horizontally')
    args = parser.parse_args()
    main(args)
