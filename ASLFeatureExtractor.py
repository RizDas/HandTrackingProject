"""
ASLFeatureExtractor.py
======================
Rich geometric feature layer for ASL sign language recognition.

Takes a 21-point MediaPipe hand landmark list (lmList) and computes a
comprehensive set of spatial features used by ASLRecognizer to classify
individual ASL letter / digit hand-shapes.

Landmark Reference (lmList[i] = [id, cx_pixel, cy_pixel]):
    Wrist  : 0
    Thumb  : CMC=1, MCP=2,  IP=3,   TIP=4
    Index  : MCP=5, PIP=6,  DIP=7,  TIP=8
    Middle : MCP=9, PIP=10, DIP=11, TIP=12
    Ring   : MCP=13,PIP=14, DIP=15, TIP=16
    Pinky  : MCP=17,PIP=18, DIP=19, TIP=20

Author  : ASL Recognizer Suite
Version : 1.0  — Phase 1 (I, L, Y, B, 5)
"""

import math
from dataclasses import dataclass
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Feature Data Class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ASLFeatures:
    """
    All geometric features derived from one frame's hand landmarks.

    All *_dist values are normalised by `hand_size` so they are invariant
    to how close or far the hand is from the camera.
    Curl values: 0.0 = fully straight finger, 1.0 = fully curled.
    """

    # ── Raw / Scale ──────────────────────────────────────────────────────────
    lmList    : list        # original 21-landmark list
    hand_size : float       # scale reference = dist(wrist→middle-MCP) × 2

    # ── Finger Extension States ──────────────────────────────────────────────
    # True  = finger is extended / "up"
    # False = finger is curled  / "down"
    # Index: [thumb(0), index(1), middle(2), ring(3), pinky(4)]
    fingers_up  : List[bool]   # boolean extended state for each finger
    finger_curl : List[float]  # continuous curl ratio 0=straight → 1=curled

    # ── Pairwise Distances (all normalised) ──────────────────────────────────
    thumb_index_dist  : float  # tip 4 ↔ tip 8
    thumb_middle_dist : float  # tip 4 ↔ tip 12
    thumb_ring_dist   : float  # tip 4 ↔ tip 16
    thumb_pinky_dist  : float  # tip 4 ↔ tip 20
    index_middle_dist : float  # tip 8 ↔ tip 12
    middle_ring_dist  : float  # tip 12 ↔ tip 16
    ring_pinky_dist   : float  # tip 16 ↔ tip 20

    # ── Thumb Position Flags ─────────────────────────────────────────────────
    thumb_across_palm      : bool  # Thumb tip x falls within index-pinky MCP x-range
    thumb_above_knuckles   : bool  # Thumb tip y is above (lower y) avg PIP line → S-wrap
    thumb_between_idx_mid  : bool  # Thumb tip y between index-PIP and middle-PIP → T

    # ── Lateral Spread (normalised x-distance between adjacent fingertips) ──
    index_middle_spread : float  # |tip8.x  − tip12.x| / hand_size
    middle_ring_spread  : float  # |tip12.x − tip16.x| / hand_size
    ring_pinky_spread   : float  # |tip16.x − tip20.x| / hand_size

    # ── Orientation ─────────────────────────────────────────────────────────
    index_dir_angle    : float  # degrees from vertical: 0=pointing up, 90=sideways
    index_middle_cross : bool   # index & middle fingertips appear crossed → R

    # ── Shape Scores ─────────────────────────────────────────────────────────
    o_shape_score : float  # 0–1: how well tips form a circular O-shape

    # ── DIP Joint Angles (for detecting bent/hooked fingers) ────────────────
    index_pip_angle  : float  # angle at index  PIP joint (MCP-PIP-DIP)  180=straight
    middle_pip_angle : float  # angle at middle PIP joint
    ring_pip_angle   : float  # angle at ring   PIP joint
    pinky_pip_angle  : float  # angle at pinky  PIP joint
    index_dip_angle  : float  # angle at index  DIP joint (PIP-DIP-TIP)  180=straight

    # ── Handedness ───────────────────────────────────────────────────────────
    is_right_hand : bool   # True = right hand detected (after mirror flip)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class ASLFeatureExtractor:
    """
    Computes ASLFeatures from a MediaPipe lmList in a single call to extract().

    Usage:
        extractor = ASLFeatureExtractor()
        features  = extractor.extract(lmList)
        if features:
            print(features.fingers_up)
    """

    def extract(self, lmList: list) -> Optional[ASLFeatures]:
        """
        Build and return an ASLFeatures object from the given landmark list.
        Returns None if lmList is empty or has fewer than 21 points.
        """
        if not lmList or len(lmList) < 21:
            return None

        lm = lmList
        hs = self._hand_size(lm)

        return ASLFeatures(
            lmList    = lm,
            hand_size = hs,

            # Finger states
            fingers_up  = self._fingers_up(lm),
            finger_curl = self._all_curls(lm),

            # Pairwise distances
            thumb_index_dist  = self._ndist(lm, 4,  8,  hs),
            thumb_middle_dist = self._ndist(lm, 4,  12, hs),
            thumb_ring_dist   = self._ndist(lm, 4,  16, hs),
            thumb_pinky_dist  = self._ndist(lm, 4,  20, hs),
            index_middle_dist = self._ndist(lm, 8,  12, hs),
            middle_ring_dist  = self._ndist(lm, 12, 16, hs),
            ring_pinky_dist   = self._ndist(lm, 16, 20, hs),

            # Thumb flags
            thumb_across_palm      = self._thumb_across_palm(lm),
            thumb_above_knuckles   = self._thumb_above_knuckles(lm),
            thumb_between_idx_mid  = self._thumb_between_idx_mid(lm),

            # Spread
            index_middle_spread = self._spread(lm, 8,  12, hs),
            middle_ring_spread  = self._spread(lm, 12, 16, hs),
            ring_pinky_spread   = self._spread(lm, 16, 20, hs),

            # Orientation
            index_dir_angle    = self._finger_dir_angle(lm, 5, 8),
            index_middle_cross = self._cross_check(lm),

            # Shape
            o_shape_score = self._o_shape_score(lm, hs),

            # Joint angles
            index_pip_angle  = self._angle(lm, 5,  6,  7),
            middle_pip_angle = self._angle(lm, 9,  10, 11),
            ring_pip_angle   = self._angle(lm, 13, 14, 15),
            pinky_pip_angle  = self._angle(lm, 17, 18, 19),
            index_dip_angle  = self._angle(lm, 6,  7,  8),

            # Handedness
            is_right_hand = lm[4][1] < lm[17][1],  # thumb tip left of pinky MCP after flip
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _dist(lm, a: int, b: int) -> float:
        """Pixel-space Euclidean distance between landmarks a and b."""
        return math.hypot(lm[a][1] - lm[b][1], lm[a][2] - lm[b][2])

    def _ndist(self, lm, a: int, b: int, hs: float) -> float:
        """Normalised distance (divided by hand_size)."""
        return self._dist(lm, a, b) / (hs + 1e-6)

    def _spread(self, lm, a: int, b: int, hs: float) -> float:
        """Lateral (x-axis only) normalised separation between two landmarks."""
        return abs(lm[a][1] - lm[b][1]) / (hs + 1e-6)

    def _hand_size(self, lm) -> float:
        """
        Scale reference: wrist(0) → middle-finger MCP(9) distance, doubled.
        Robust against hand rotation and distance from camera.
        """
        return self._dist(lm, 0, 9) * 2.0

    @staticmethod
    def _angle(lm, a: int, b: int, c: int) -> float:
        """
        Angle in degrees at landmark b, formed by the path a→b→c.
        Returns 180.0 (straight) when vectors are degenerate.
        """
        ax = lm[a][1] - lm[b][1]; ay = lm[a][2] - lm[b][2]
        cx = lm[c][1] - lm[b][1]; cy = lm[c][2] - lm[b][2]
        mag = math.hypot(ax, ay) * math.hypot(cx, cy)
        if mag < 1e-6:
            return 180.0
        cos_val = max(-1.0, min(1.0, (ax * cx + ay * cy) / mag))
        return math.degrees(math.acos(cos_val))

    def _curl_from_angle(self, angle_deg: float) -> float:
        """
        Convert a joint angle to a curl ratio.
            180° (straight) → 0.0
             90° (right-angle bend) → 1.0
        """
        return 1.0 - max(0.0, min(1.0, (angle_deg - 90.0) / 90.0))

    def _all_curls(self, lm) -> List[float]:
        """
        Compute curl [0=straight, 1=fully curled] for all 5 fingers.

        Thumb  : angle at IP joint  (MCP=2, IP=3, TIP=4)
        Others : angle at PIP joint (MCP, PIP, DIP)
        """
        curls = []

        # Thumb — IP joint
        curls.append(self._curl_from_angle(self._angle(lm, 2, 3, 4)))

        # Index, Middle, Ring, Pinky — PIP joint
        for mcp, pip, dip in [(5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)]:
            curls.append(self._curl_from_angle(self._angle(lm, mcp, pip, dip)))

        return curls

    def _fingers_up(self, lm) -> List[bool]:
        """
        Determine which fingers are extended (True) or curled (False).

        Thumb  : compare tip x to IP x, adjusted for left/right hand
        Others : tip y < PIP y  (tip is physically higher on screen)
        """
        fu = []

        # ── Thumb ────────────────────────────────────────────────────────────
        # After mirror-flip: right hand → thumb is on the left side of the frame
        # (thumb-tip x < pinky-MCP x)
        if lm[4][1] < lm[17][1]:           # right hand in mirror mode
            fu.append(lm[4][1] < lm[3][1]) # tip x < IP x → extended left
        else:                               # left hand
            fu.append(lm[4][1] > lm[3][1]) # tip x > IP x → extended right

        # ── Four fingers ─────────────────────────────────────────────────────
        for tip_id, pip_id in zip([8, 12, 16, 20], [6, 10, 14, 18]):
            fu.append(lm[tip_id][2] < lm[pip_id][2])  # tip higher than PIP

        return fu

    def _finger_dir_angle(self, lm, mcp: int, tip: int) -> float:
        """
        Angle of a finger from vertical.
        0° = pointing straight up, 90° = pointing horizontally sideways.
        """
        dx = lm[tip][1] - lm[mcp][1]
        dy = lm[mcp][2] - lm[tip][2]   # note: y-axis is flipped (down = larger)
        if abs(dy) < 1:
            return 90.0
        return math.degrees(math.atan2(abs(dx), abs(dy)))

    def _thumb_across_palm(self, lm) -> bool:
        """
        True when thumb tip x falls within the horizontal span of the four
        finger MCPs (index=5 … pinky=17).  Indicates thumb is tucked/folded
        across the palm (key for distinguishing B, S, M, N, T from A, E).
        """
        mcp_xs = [lm[5][1], lm[9][1], lm[13][1], lm[17][1]]
        return min(mcp_xs) <= lm[4][1] <= max(mcp_xs)

    def _thumb_above_knuckles(self, lm) -> bool:
        """
        True when the thumb tip y is above (smaller y = higher on screen)
        the average PIP-joint y of index, middle, and ring fingers.
        This indicates the thumb wraps over the curled fingers → S-shape.
        """
        avg_pip_y = (lm[6][2] + lm[10][2] + lm[14][2]) / 3.0
        return lm[4][2] < avg_pip_y

    def _thumb_between_idx_mid(self, lm) -> bool:
        """
        True when thumb tip y sits between the index PIP y and middle PIP y.
        Indicates thumb poking up between those two fingers → T-shape.
        """
        idx_pip_y = lm[6][2]
        mid_pip_y = lm[10][2]
        lo = min(idx_pip_y, mid_pip_y)
        hi = max(idx_pip_y, mid_pip_y)
        return lo < lm[4][2] < hi

    def _cross_check(self, lm) -> bool:
        """
        Rough cross-detection for index and middle fingers (R).
        Returns True if index-tip and middle-tip are very close together
        (indicating they are crossing / interleaved).
        """
        return abs(lm[8][1] - lm[12][1]) < 20   # < 20 px lateral gap at tips

    def _o_shape_score(self, lm, hs: float) -> float:
        """
        Score in [0, 1] for how well the five fingertips form an O-ring shape.

        Method: compute the average Euclidean distance of each fingertip from
        their collective centroid, then normalise.  A tight cluster (all tips
        meet) scores close to 1.  Fully spread fingers score near 0.

        Calibration: avg_radius ≈ 0.20 × hand_size ↔ score ≈ 0.
        """
        tip_ids = [4, 8, 12, 16, 20]
        cx = sum(lm[i][1] for i in tip_ids) / 5.0
        cy = sum(lm[i][2] for i in tip_ids) / 5.0
        avg_r = sum(math.hypot(lm[i][1] - cx, lm[i][2] - cy) for i in tip_ids) / 5.0
        norm_r = avg_r / (hs + 1e-6)
        return max(0.0, 1.0 - norm_r / 0.20)
