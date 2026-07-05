"""
ASLRecognizerApp.py
===================
Live-camera demo for the ASL Sign Language Recognizer.

Detects hand landmarks in real-time using HandTrackingModule, passes them
through ASLRecognizer, and renders an annotated camera feed with:

  - Large letter display with colour-coded confidence
  - Animated confidence bar
  - Per-letter score chart (top-5 matches)
  - Temporal smoothing to eliminate single-frame flickers
  - Active-sign reference panel
  - Debug panel toggle (D key) showing raw feature values
  - Phase progression (N key adds the next batch of signs)

Controls
--------
  N  → Add the next phase of signs
  D  → Toggle debug / feature-values panel
  R  → Reset (clear history and detection)
  Q  → Quit

Author  : ASL Recognizer Suite
Version : 1.0 — Phase 1 (I, L, Y, B, 5)
"""

import cv2
import time
import sys
from collections import Counter
from typing import Dict, List, Optional

# ── Project imports ───────────────────────────────────────────────────────────
try:
    from HandTrackingModule import handDetector
    from ASLRecognizer import ASLRecognizer
    from ASLFeatureExtractor import ASLFeatures
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("Make sure HandTrackingModule.py, ASLRecognizer.py, and")
    print("ASLFeatureExtractor.py are in the same directory.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Colour Palette  (BGR)
# ─────────────────────────────────────────────────────────────────────────────
C = {
    'green'      : (72,  209, 100),
    'gold'       : (40,  190, 235),
    'blue_muted' : (200, 130,  70),
    'white'      : (255, 255, 255),
    'gray'       : (170, 170, 185),
    'dim'        : ( 90,  90, 105),
    'panel'      : ( 22,  22,  30),
    'panel_light': ( 36,  36,  48),
    'red_muted'  : ( 70,  70, 200),
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX

# How many frames the recognised letter must be stable before it is displayed
SMOOTH_FRAMES = 5

# Phase descriptions for the HUD
PHASE_LABELS = {
    1: "Phase 1  [I  L  Y  B  5]",
    2: "Phase 2  [1  W  U  V  A  S  O]",
    3: "Phase 3  [D  F  E  C  T  M  N]",
    4: "Phase 4  [K  R  X  G  H  P  Q]",
    5: "Phase 5  [0  2  3  4  6  7  8  9]",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def overlay_rect(img, x: int, y: int, w: int, h: int,
                 color=(22, 22, 30), alpha: float = 0.72) -> None:
    """Draw a semi-transparent filled rectangle."""
    x2, y2 = x + w, y + h
    # Clamp to image bounds
    ih, iw = img.shape[:2]
    x  = max(0, x);  y  = max(0, y)
    x2 = min(iw, x2); y2 = min(ih, y2)
    if x2 <= x or y2 <= y:
        return
    roi = img[y:y2, x:x2]
    overlay = roi.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def conf_color(conf: float) -> tuple:
    """Return BGR colour based on confidence level."""
    if conf >= 0.85:
        return C['green']
    elif conf >= 0.65:
        return C['gold']
    else:
        return C['red_muted']


def draw_bar(img, x: int, y: int, w: int, h: int,
             value: float, color: tuple) -> None:
    """Horizontal filled progress bar."""
    # Background track
    cv2.rectangle(img, (x, y), (x + w, y + h), C['dim'], -1)
    # Fill
    filled = int(w * max(0.0, min(1.0, value)))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x + filled, y + h), color, -1)
    # Border
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 95), 1)


def put_text(img, text: str, x: int, y: int, scale: float = 0.55,
             color: tuple = C['gray'], thickness: int = 1,
             font=FONT) -> None:
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# HUD drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_top_banner(img, fps: float, active_phases: List[int]) -> None:
    """Title bar at the top of the frame."""
    h, w = img.shape[:2]
    overlay_rect(img, 0, 0, w, 52, color=C['panel'], alpha=0.82)

    # Title
    put_text(img, "ASL Sign Language Recognizer", 14, 34,
             scale=0.80, color=C['white'], thickness=1, font=FONT_BOLD)

    # Phase + FPS info (right-aligned)
    phase_str = "  |  ".join(PHASE_LABELS[p] for p in sorted(active_phases))
    info = f"{phase_str}   |   FPS {fps:>4.0f}"
    (tw, _), _ = cv2.getTextSize(info, FONT, 0.45, 1)
    put_text(img, info, w - tw - 12, 34, scale=0.45, color=C['dim'])


def draw_letter_panel(img, letter: Optional[str], confidence: float,
                      px: int, py: int, pw: int, ph: int) -> None:
    """Large letter + confidence bar panel."""
    overlay_rect(img, px, py, pw, ph, color=C['panel'], alpha=0.85)

    # Panel border
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (55, 55, 70), 1)

    if letter:
        col = conf_color(confidence)

        # Header label
        put_text(img, "DETECTED", px + 14, py + 22,
                 scale=0.45, color=C['dim'])

        # Shadow layer for mega-letter
        (lw, lh), _ = cv2.getTextSize(letter, FONT_BOLD, 5.0, 9)
        lx = px + (pw - lw) // 2
        ly = py + ph - 50
        cv2.putText(img, letter, (lx + 3, ly + 3),
                    FONT_BOLD, 5.0, (10, 10, 18), 12, cv2.LINE_AA)
        # Main letter
        cv2.putText(img, letter, (lx, ly),
                    FONT_BOLD, 5.0, col, 8, cv2.LINE_AA)

        # Confidence bar
        bar_x = px + 12
        bar_y = py + ph - 28
        bar_w = pw - 70
        draw_bar(img, bar_x, bar_y, bar_w, 14, confidence, col)
        put_text(img, f"{confidence * 100:.0f}%",
                 bar_x + bar_w + 6, bar_y + 11,
                 scale=0.50, color=col)
    else:
        # No detection placeholder
        put_text(img, "Show a sign...", px + 18, py + ph // 2 + 8,
                 scale=0.55, color=C['dim'])
        put_text(img, "Hold still for detection", px + 10, py + ph // 2 + 30,
                 scale=0.42, color=(60, 60, 75))


def draw_active_panel(img, active_letters: List[str],
                      px: int, py: int, pw: int) -> None:
    """Shows which signs are currently active."""
    ph = 72
    overlay_rect(img, px, py, pw, ph, color=C['panel_light'], alpha=0.80)
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (55, 55, 70), 1)

    put_text(img, "Active Signs", px + 10, py + 20,
             scale=0.45, color=C['dim'])

    # Render letters in rows of 10
    letters_row = "  ".join(active_letters)
    put_text(img, letters_row, px + 10, py + 44,
             scale=0.56, color=C['white'], thickness=1)


def draw_score_chart(img, scores: Dict[str, float],
                     sx: int, sy: int) -> None:
    """
    Shows the top-5 confidence scores as a small bar chart
    in the bottom-left corner.
    """
    if not scores:
        return

    panel_w, panel_h = 240, 180
    overlay_rect(img, sx, sy, panel_w, panel_h,
                 color=C['panel'], alpha=0.82)
    cv2.rectangle(img, (sx, sy), (sx + panel_w, sy + panel_h),
                  (55, 55, 70), 1)

    put_text(img, "Top Matches", sx + 10, sy + 20,
             scale=0.48, color=C['dim'])

    sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
    bar_max_w = 140

    for idx, (ltr, sc) in enumerate(sorted_items):
        row_y = sy + 40 + idx * 28
        col = conf_color(sc) if sc >= 0.65 else C['dim']

        # Letter label
        put_text(img, ltr, sx + 10, row_y + 2,
                 scale=0.72, color=col, thickness=2, font=FONT_BOLD)

        # Bar
        bar_x = sx + 38
        filled = int(bar_max_w * sc)
        cv2.rectangle(img, (bar_x, row_y - 10), (bar_x + bar_max_w, row_y + 4),
                      (40, 40, 52), -1)
        if filled > 0:
            cv2.rectangle(img, (bar_x, row_y - 10), (bar_x + filled, row_y + 4),
                          col, -1)
        cv2.rectangle(img, (bar_x, row_y - 10), (bar_x + bar_max_w, row_y + 4),
                      (70, 70, 85), 1)

        # Numeric score
        put_text(img, f"{sc:.2f}", bar_x + bar_max_w + 6, row_y + 2,
                 scale=0.44, color=C['gray'])


def draw_debug_panel(img, features: Optional[ASLFeatures],
                     dx: int, dy: int) -> None:
    """Raw feature values panel (toggled by D key)."""
    if features is None:
        return

    pw, ph = 280, 310
    overlay_rect(img, dx, dy, pw, ph, color=(16, 16, 22), alpha=0.88)
    cv2.rectangle(img, (dx, dy), (dx + pw, dy + ph), (60, 60, 80), 1)

    put_text(img, "DEBUG — Feature Values", dx + 8, dy + 20,
             scale=0.45, color=C['gold'])

    f = features
    lines = [
        f"fingers_up  : {['T' if v else 'F' for v in f.fingers_up]}",
        f"finger_curl : [{' '.join(f'{c:.2f}' for c in f.finger_curl)}]",
        f"thumb_across: {f.thumb_across_palm}",
        f"thumb_above : {f.thumb_above_knuckles}",
        f"o_score     : {f.o_shape_score:.3f}",
        f"idx_mid_dist: {f.index_middle_dist:.3f}",
        f"idx_spread  : {f.index_middle_spread:.3f}",
        f"avg_spread  : {(f.index_middle_spread + f.middle_ring_spread + f.ring_pinky_spread) / 3:.3f}",
        f"idx_dir_ang : {f.index_dir_angle:.1f} deg",
        f"idx_pip_ang : {f.index_pip_angle:.1f} deg",
        f"hand_size   : {f.hand_size:.1f} px",
        f"right_hand  : {f.is_right_hand}",
    ]

    for i, line in enumerate(lines):
        put_text(img, line, dx + 8, dy + 40 + i * 22,
                 scale=0.42, color=C['gray'])


def draw_controls(img) -> None:
    """Controls hint bar at the bottom."""
    h, w = img.shape[:2]
    overlay_rect(img, 0, h - 30, w, 30, color=C['panel'], alpha=0.80)
    put_text(img, "D: Debug    R: Reset    Q: Quit",
             14, h - 10, scale=0.48, color=C['dim'])


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Camera ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check that it is connected.")
        sys.exit(1)

    # ── Modules ──────────────────────────────────────────────────────────────
    detector   = handDetector(maxHands=1, detectionCon=0.85, trackCon=0.80)
    recognizer = ASLRecognizer(active_phases=[1, 2, 3])

    # ── State ────────────────────────────────────────────────────────────────
    history       : List[Optional[str]] = []
    debug_mode    : bool = False
    pTime         : float = time.time()

    # Cached display values (updated each frame)
    display_letter : Optional[str]        = None
    display_conf   : float                = 0.0
    display_scores : Dict[str, float]     = {}
    last_features  : Optional[ASLFeatures] = None

    # Window
    cv2.namedWindow("ASL Sign Language Recognizer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ASL Sign Language Recognizer", 1280, 720)

    print("=" * 60)
    print("  ASL Sign Language Recognizer  —  Phase 1-3")
    print(f"  Signs active: {' '.join(recognizer.active_letters)}")
    print("  D → debug  |  R → reset  |  Q → quit")
    print("=" * 60)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Dropped frame — retrying…")
            continue

        # Mirror and process
        frame = cv2.flip(frame, 1)
        frame = detector.findHands(frame, draw=True)
        lmList, bbox = detector.findPosition(frame, draw=False)

        # ── Recognition ───────────────────────────────────────────────────
        raw_letter, raw_conf, raw_scores = None, 0.0, {}

        if lmList:
            raw_letter, raw_conf, raw_scores = recognizer.recognize(lmList)
            # Cache features for debug panel
            last_features = recognizer.extractor.extract(lmList)

            # ── Temporal smoothing ─────────────────────────────────────────
            history.append(raw_letter)
            if len(history) > SMOOTH_FRAMES:
                history.pop(0)

            counts = Counter(history)
            best_mc = counts.most_common(1)[0]
            # Letter shown only when it holds the majority of the window
            if best_mc[1] >= (SMOOTH_FRAMES // 2 + 1) and best_mc[0] is not None:
                display_letter = best_mc[0]
                display_conf   = raw_conf
                display_scores = raw_scores
            else:
                display_letter = None
                display_scores = raw_scores
        else:
            history.clear()
            display_letter = None
            display_scores = {}
            last_features  = None

        # ── FPS ───────────────────────────────────────────────────────────
        now   = time.time()
        fps   = 1.0 / max(now - pTime, 1e-6)
        pTime = now

        # ── Draw HUD ──────────────────────────────────────────────────────
        h, w = frame.shape[:2]

        draw_top_banner(frame, fps, recognizer._active_phases)

        # Right-side panels
        panel_x = w - 250
        draw_letter_panel(frame, display_letter, display_conf,
                          panel_x, 60, 245, 210)
        draw_active_panel(frame, recognizer.active_letters,
                          panel_x, 278, 245)

        # Score chart (bottom-left)
        draw_score_chart(frame, display_scores, 10, h - 200)

        # Debug panel (left side, above score chart)
        if debug_mode:
            draw_debug_panel(frame, last_features, 10, 60)

        draw_controls(frame)

        cv2.imshow("ASL Sign Language Recognizer", frame)

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quitting…")
            break

        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

        elif key == ord('r'):
            history.clear()
            display_letter = None
            display_conf   = 0.0
            display_scores = {}
            print("History cleared.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
