"""
ASLRecognizer.py
================
Rule-based ASL hand-sign classifier built on top of ASLFeatureExtractor.

Each letter or digit has a dedicated _check_X(f: ASLFeatures) → float method
that returns a confidence score between 0.0 and 1.0.

Letters are organised into phases from most-distinct (Phase 1) to most-nuanced
(Phase 4), with digits in Phase 5.  Add a phase at runtime via `add_phase(n)`
or by pressing N in the demo app.

Phase 1 — Highly Distinct:  I  L  Y  B  5
Phase 2 — One Extra Feature: 1  W  U  V  A  S  O
Phase 3 — Two+ Features:     D  F  E  C  T  M  N
Phase 4 — Complex / Nuanced: K  R  X  G  H  P  Q
Phase 5 — Digits:            0  2  3  4  6  7  8  9

Author  : ASL Recognizer Suite
Version : 1.0  — Phase 1 implemented
"""

from ASLFeatureExtractor import ASLFeatureExtractor, ASLFeatures
from typing import Dict, List, Optional, Tuple


class ASLRecognizer:
    """
    Classify a single ASL hand-sign from a MediaPipe lmList.

    Usage::

        recognizer = ASLRecognizer(active_phases=[1])
        letter, confidence, all_scores = recognizer.recognize(lmList)

    ``recognize`` returns (None, 0.0, {}) when no hand is visible or when
    the best-match confidence is below ``MIN_CONFIDENCE``.
    """

    # ── Phase map ─────────────────────────────────────────────────────────────
    PHASES: Dict[int, List[str]] = {
        1: ['I', 'L', 'Y', 'B', '5'],
        2: ['1', 'W', 'U', 'V', 'A', 'S', 'O'],
        3: ['D', 'F', 'E', 'C', 'T', 'M', 'N'],
        4: ['K', 'R', 'X', 'G', 'H', 'P', 'Q'],
        5: ['0', '2', '3', '4', '6', '7', '8', '9'],
    }

    # Minimum confidence required to return a letter (avoids noise detections)
    MIN_CONFIDENCE: float = 0.65

    def __init__(self, active_phases: Optional[List[int]] = None):
        self.extractor      = ASLFeatureExtractor()
        self._active_phases : List[int] = []
        self.active_letters : List[str] = []

        for phase in (active_phases or [1, 2]):
            self.add_phase(phase)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def add_phase(self, phase: int) -> None:
        """Enable all letters in the given phase number."""
        if phase in self._active_phases:
            return
        if phase not in self.PHASES:
            raise ValueError(f"Unknown phase {phase}. Valid phases: {list(self.PHASES)}")
        self._active_phases.append(phase)
        for letter in self.PHASES[phase]:
            if letter not in self.active_letters:
                self.active_letters.append(letter)

    def recognize(
        self, lmList: list
    ) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Classify the current hand pose.

        Parameters
        ----------
        lmList : list of [id, cx, cy] from HandTrackingModule.findPosition()

        Returns
        -------
        letter     : str or None — best matching sign, or None if uncertain
        confidence : float       — score of best match (0–1)
        all_scores : dict        — {letter: score} for every active letter
        """
        features = self.extractor.extract(lmList)
        if features is None:
            return None, 0.0, {}

        scores: Dict[str, float] = {}
        for letter in self.active_letters:
            fn = getattr(self, f'_check_{letter}', None)
            if fn is not None:
                scores[letter] = round(fn(features), 4)

        if not scores:
            return None, 0.0, {}

        best_letter = max(scores, key=scores.get)
        best_score  = scores[best_letter]

        if best_score < self.MIN_CONFIDENCE:
            return None, best_score, scores

        return best_letter, best_score, scores

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Highly Distinct Handshapes
    # ═════════════════════════════════════════════════════════════════════════
    #
    # All five letters are mutually exclusive by their finger-up pattern alone,
    # so there is no cross-letter ambiguity within this phase.
    #
    # Scoring convention
    # ------------------
    # • Hard requirements:  return 0.0 immediately if violated.
    # • Base score (0.50–0.60): awarded if all hard requirements pass.
    # • Bonus terms:  add up to the remaining slack toward 1.0.
    # • Total is capped at 1.0.
    # ═════════════════════════════════════════════════════════════════════════

    def _check_I(self, f: ASLFeatures) -> float:
        """
        I — Only the pinky finger is extended. Thumb tucked or resting.

        Hard requirements:
            • pinky  UP
            • index  DOWN
            • middle DOWN
            • ring   DOWN

        Bonus:
            • index / middle / ring are well curled (+0.25)
            • pinky is well extended (low curl)    (+0.20)

        Penalty:
            • If thumb is clearly extended (fingers_up[0]=True AND low curl),
              subtract 0.15.  This breaks the Y vs I tie: in Y the thumb IS
              fully up, so _check_Y will score higher because it requires thumb
              up as a hard condition.  Penalising here prevents I from tying Y.
        """
        fu, fc = f.fingers_up, f.finger_curl

        # ── Hard checks ────────────────────────────────────────────────────
        if not fu[4]:  return 0.0   # pinky MUST be up
        if     fu[1]:  return 0.0   # index  must be DOWN
        if     fu[2]:  return 0.0   # middle must be DOWN
        if     fu[3]:  return 0.0   # ring   must be DOWN

        # ── Base ───────────────────────────────────────────────────────────
        score = 0.55

        # ── Bonus: other 3 fingers well curled ────────────────────────────
        avg_curl_imr = (fc[1] + fc[2] + fc[3]) / 3.0
        score += avg_curl_imr * 0.25

        # ── Bonus: pinky genuinely extended ───────────────────────────────
        score += (1.0 - fc[4]) * 0.20

        # ── Penalty: thumb clearly extended → likely Y, not I ──────────────
        # A well-extended thumb (up=True, low curl) strongly suggests Y.
        if fu[0] and fc[0] < 0.35:
            score -= 0.18

        return min(1.0, max(0.0, score))

    def _check_L(self, f: ASLFeatures) -> float:
        """
        L — Thumb and index extended in an L-shape; other three fingers curled.

        Hard requirements:
            • thumb  UP
            • index  UP
            • middle DOWN
            • ring   DOWN
            • pinky  DOWN

        Bonus:
            • middle / ring / pinky well curled (+0.25)
            • index  genuinely extended         (+0.15)
            • thumb  genuinely extended         (+0.10)
        """
        fu, fc = f.fingers_up, f.finger_curl

        # ── Hard checks ────────────────────────────────────────────────────
        if not fu[0]:  return 0.0   # thumb  MUST be up
        if not fu[1]:  return 0.0   # index  MUST be up
        if     fu[2]:  return 0.0   # middle must be DOWN
        if     fu[3]:  return 0.0   # ring   must be DOWN
        if     fu[4]:  return 0.0   # pinky  must be DOWN

        # ── Base ───────────────────────────────────────────────────────────
        score = 0.50

        # ── Bonus: curled fingers are actually curled ──────────────────────
        avg_curl_mrp = (fc[2] + fc[3] + fc[4]) / 3.0
        score += avg_curl_mrp * 0.25

        # ── Bonus: extended fingers are actually extended ──────────────────
        score += (1.0 - fc[1]) * 0.15   # index
        score += (1.0 - fc[0]) * 0.10   # thumb

        return min(1.0, score)

    def _check_Y(self, f: ASLFeatures) -> float:
        """
        Y — Thumb and pinky extended ("hang-loose"); middle three fingers curled.

        Hard requirements:
            • thumb  UP  ← this is the key differentiator from I
            • index  DOWN
            • middle DOWN
            • ring   DOWN
            • pinky  UP

        Bonus:
            • index / middle / ring well curled (+0.25)
            • pinky genuinely extended          (+0.15)
            • thumb genuinely extended          (+0.15)
              Higher thumb bonus vs I to ensure Y wins whenever thumb is up.
        """
        fu, fc = f.fingers_up, f.finger_curl

        # ── Hard checks ────────────────────────────────────────────────────
        if not fu[0]:  return 0.0   # thumb  MUST be up — key Y discriminator
        if     fu[1]:  return 0.0   # index  must be DOWN
        if     fu[2]:  return 0.0   # middle must be DOWN
        if     fu[3]:  return 0.0   # ring   must be DOWN
        if not fu[4]:  return 0.0   # pinky  MUST be up

        # ── Base ───────────────────────────────────────────────────────────
        score = 0.55

        # ── Bonus: middle three genuinely curled ──────────────────────────
        avg_curl_imr = (fc[1] + fc[2] + fc[3]) / 3.0
        score += avg_curl_imr * 0.25

        # ── Bonus: extended fingers genuinely extended ─────────────────────
        score += (1.0 - fc[4]) * 0.15   # pinky extension quality
        score += (1.0 - fc[0]) * 0.15   # thumb extension quality (larger than I bonus)

        # ── Bonus: thumb and pinky spread far apart (Y-shape) ──────────────
        if f.thumb_pinky_dist > 0.45:
            score += 0.05

        return min(1.0, score)

    def _check_B(self, f: ASLFeatures) -> float:
        """
        B — Four fingers extended vertically together; thumb tucked across palm.

        Hard requirements:
            • thumb  DOWN (tucked)
            • index  UP
            • middle UP
            • ring   UP
            • pinky  UP

        Bonus:
            • all four fingers well extended           (+0.20)
            • thumb_across_palm is True                (+0.20)
            • fingers are close together (low spread)  (+0.10)
              This prevents future confusion with digit-4 (thumb to side, fingers spread)

        Note: When digit-4 is added in Phase 5, B vs 4 disambiguates via
        thumb_across_palm (B=True, 4=False) and finger spread.
        """
        fu, fc = f.fingers_up, f.finger_curl

        # ── Hard checks ────────────────────────────────────────────────────
        if     fu[0]:  return 0.0   # thumb  MUST be DOWN
        if not fu[1]:  return 0.0   # index  MUST be up
        if not fu[2]:  return 0.0   # middle MUST be up
        if not fu[3]:  return 0.0   # ring   MUST be up
        if not fu[4]:  return 0.0   # pinky  MUST be up

        # ── Base ───────────────────────────────────────────────────────────
        score = 0.45

        # ── Penalty: if fingers are significantly curled, it's likely C, not B
        avg_curl_4 = sum(fc[1:5]) / 4.0
        if avg_curl_4 > 0.25:
            score -= 0.25

        # ── Bonus: four fingers well extended ─────────────────────────────
        avg_ext = sum(1.0 - fc[i] for i in [1, 2, 3, 4]) / 4.0
        score += avg_ext * 0.20

        # ── Bonus: thumb genuinely tucked ─────────────────────────────────
        if f.thumb_across_palm:
            score += 0.20

        # ── Bonus: fingers held close together ────────────────────────────
        avg_spread = (f.index_middle_spread + f.middle_ring_spread + f.ring_pinky_spread) / 3.0
        # threshold: spread < 0.07 (normalised) is "close"
        score += max(0.0, (0.07 - avg_spread)) / 0.07 * 0.15

        return min(1.0, score)

    def _check_5(self, f: ASLFeatures) -> float:
        """
        5 — All five fingers extended and spread apart ("open hand").

        Hard requirements:
            • ALL five fingers UP

        Bonus:
            • all fingers well extended     (+0.20)
            • fingers visibly spread apart  (+0.40)
              Uses the average lateral spread between adjacent fingertips,
              normalised so spread ≥ 0.10 earns the full bonus.

        Note: B has all four fingers UP but thumb DOWN, so fingers_up[0]
        alone cleanly separates 5 from B.  The spread bonus further
        separates 5 from future letters with all fingers up but held tightly.
        """
        fu, fc = f.fingers_up, f.finger_curl

        # ── Hard check: every finger must be extended ──────────────────────
        if not all(fu):
            return 0.0

        # ── Base ───────────────────────────────────────────────────────────
        score = 0.35

        # ── Bonus: genuinely extended fingers ─────────────────────────────
        avg_ext = sum(1.0 - c for c in fc) / 5.0
        score += avg_ext * 0.20

        # ── Bonus: fingers spread apart ───────────────────────────────────
        avg_spread = (f.index_middle_spread + f.middle_ring_spread + f.ring_pinky_spread) / 3.0
        # Normalise: full bonus at avg_spread >= 0.10
        spread_score = min(avg_spread / 0.10, 1.0)
        score += spread_score * 0.40

        # ── Soft penalty: if thumb is barely up, penalise  ─────────────────
        # (prevents 5 from firing when only 4 fingers are truly extended)
        if fc[0] > 0.5:            # thumb is more curled than extended
            score -= 0.10

        return min(1.0, max(0.0, score))

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 2 — One Extra Feature Required
    # ═════════════════════════════════════════════════════════════════════════

    def _check_1(self, f: ASLFeatures) -> float:
        """1 — Only index finger up, pointing up."""
        fu, fc = f.fingers_up, f.finger_curl
        if not fu[1]:  return 0.0
        if     fu[2]:  return 0.0
        if     fu[3]:  return 0.0
        if     fu[4]:  return 0.0
        if     fu[0]:  return 0.0  # thumb tucked

        score = 0.50
        score += (fc[2] + fc[3] + fc[4]) / 3.0 * 0.20
        score += (1.0 - fc[1]) * 0.15
        
        # Pointing vertically
        vert_score = max(0.0, (45.0 - f.index_dir_angle) / 45.0)
        score += vert_score * 0.15
        
        # Penalty: if thumb touches middle/ring, it's a D, not a 1
        min_thumb_dist = min(f.thumb_middle_dist, f.thumb_ring_dist)
        if min_thumb_dist < 0.4:
            score -= 0.25
        
        return min(1.0, max(0.0, score))

    def _check_W(self, f: ASLFeatures) -> float:
        """W — Index, middle, ring up. Pinky and thumb down."""
        fu, fc = f.fingers_up, f.finger_curl
        if not fu[1]:  return 0.0
        if not fu[2]:  return 0.0
        if not fu[3]:  return 0.0
        if     fu[4]:  return 0.0
        if     fu[0]:  return 0.0

        score = 0.50
        score += sum((1.0 - fc[i]) for i in [1, 2, 3]) / 3.0 * 0.25
        score += (fc[0] + fc[4]) / 2.0 * 0.10
        
        # Spread apart
        avg_spread = (f.index_middle_spread + f.middle_ring_spread) / 2.0
        spread_score = min(avg_spread / 0.15, 1.0)
        score += spread_score * 0.15
        
        return min(1.0, score)

    def _check_U(self, f: ASLFeatures) -> float:
        """U — Index and middle up, HELD CLOSE."""
        fu, fc = f.fingers_up, f.finger_curl
        if not fu[1]:  return 0.0
        if not fu[2]:  return 0.0
        if     fu[3]:  return 0.0
        if     fu[4]:  return 0.0
        if     fu[0]:  return 0.0

        if f.index_middle_dist > 0.25:  # Threshold tuned for real hand spread
            return 0.0

        score = 0.55
        score += sum((1.0 - fc[i]) for i in [1, 2]) / 2.0 * 0.15
        score += sum(fc[i] for i in [0, 3, 4]) / 3.0 * 0.15
        
        close_score = max(0.0, (0.25 - f.index_middle_dist) / 0.25)
        score += close_score * 0.15
        return min(1.0, score)

    def _check_V(self, f: ASLFeatures) -> float:
        """V — Index and middle up, SPREAD APART."""
        fu, fc = f.fingers_up, f.finger_curl
        if not fu[1]:  return 0.0
        if not fu[2]:  return 0.0
        if     fu[3]:  return 0.0
        if     fu[4]:  return 0.0
        if     fu[0]:  return 0.0

        if f.index_middle_dist <= 0.25:
            return 0.0

        score = 0.55
        score += sum((1.0 - fc[i]) for i in [1, 2]) / 2.0 * 0.15
        score += sum(fc[i] for i in [0, 3, 4]) / 3.0 * 0.15
        
        spread_score = min((f.index_middle_dist - 0.25) / 0.35, 1.0)
        score += spread_score * 0.20
        return min(1.0, score)

    def _check_A(self, f: ASLFeatures) -> float:
        """A — Fist, thumb beside."""
        fu, fc = f.fingers_up, f.finger_curl
        if any(fu[1:5]): return 0.0

        if f.thumb_across_palm or f.thumb_above_knuckles:
            return 0.0

        if f.o_shape_score > 0.4:
            return 0.0

        score = 0.50
        score += sum(fc[1:5]) / 4.0 * 0.25
        score += (1.0 - fc[0]) * 0.15
        score += 0.10
        return min(1.0, score)

    def _check_S(self, f: ASLFeatures) -> float:
        """S — Fist, thumb over knuckles."""
        fu, fc = f.fingers_up, f.finger_curl
        if any(fu[1:5]): return 0.0

        if not (f.thumb_across_palm or f.thumb_above_knuckles):
            return 0.0

        if f.o_shape_score > 0.4:
            return 0.0

        score = 0.50
        score += sum(fc[1:5]) / 4.0 * 0.25
        
        if f.thumb_above_knuckles:
            score += 0.15
        elif f.thumb_across_palm:
            score += 0.10
            
        if fc[0] < 0.2:
            score -= 0.10
        else:
            score += fc[0] * 0.10
            
        # S features a horizontal thumb wrap
        lm = f.lmList
        dx = abs(lm[4][1] - lm[3][1])
        dy = abs(lm[4][2] - lm[3][2])
        if dx > dy:
            score += 0.15  # strong horizontal wrap
            
        # Penalty: if fingertips are resting ON the thumb, it's E, not S
        avg_tip_dist = sum([f.thumb_index_dist, f.thumb_middle_dist, f.thumb_ring_dist, f.thumb_pinky_dist]) / 4.0
        if avg_tip_dist < 0.6:
            score -= 0.25
            
        return min(1.0, max(0.0, score))

    def _check_O(self, f: ASLFeatures) -> float:
        """O — All fingers curve to meet thumb."""
        fu, fc = f.fingers_up, f.finger_curl
        
        if f.o_shape_score < 0.45:
            return 0.0
            
        # O shouldn't have too many fingers straight up
        if sum(fu) >= 2 and sum(fc) < 1.0:
            return 0.0
            
        score = 0.40
        o_bonus = min((f.o_shape_score - 0.45) / 0.4, 1.0)
        score += o_bonus * 0.40
        
        avg_curl = sum(fc) / 5.0
        if 0.4 < avg_curl < 0.9:
            score += 0.20
        else:
            score += 0.10
            
        return min(1.0, score)

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 3 — Two+ Features Required
    # ═════════════════════════════════════════════════════════════════════════

    def _get_t_n_m_position(self, f: ASLFeatures) -> str:
        """Returns 'A', 'T', 'N', 'M' based on lateral thumb tip position."""
        lm = f.lmList
        tx = lm[4][1]
        
        # Sort index, middle, ring, pinky MCPs by X coordinate
        coords = [
            (lm[5][1], 'I'),
            (lm[9][1], 'M'),
            (lm[13][1], 'R'),
            (lm[17][1], 'P')
        ]
        coords.sort(key=lambda x: x[0])
        
        if tx < coords[0][0]: pos = 0
        elif tx < coords[1][0]: pos = 1
        elif tx < coords[2][0]: pos = 2
        elif tx < coords[3][0]: pos = 3
        else: pos = 4
        
        if f.is_right_hand:
            return {0:'A', 1:'T', 2:'N', 3:'M', 4:'none'}.get(pos, 'none')
        else:
            return {4:'A', 3:'T', 2:'N', 1:'M', 0:'none'}.get(pos, 'none')

    def _check_D(self, f: ASLFeatures) -> float:
        """D — Index UP, others DOWN. Thumb touches middle/ring."""
        fu, fc = f.fingers_up, f.finger_curl
        if not fu[1]: return 0.0
        if any(fu[2:5]): return 0.0
        
        # Thumb should touch/be near middle or ring tip
        min_dist = min(f.thumb_middle_dist, f.thumb_ring_dist)
        if min_dist > 0.45:
            return 0.0
            
        score = 0.60
        score += (fc[2] + fc[3] + fc[4]) / 3.0 * 0.20
        score += (1.0 - fc[1]) * 0.10
        
        # Bonus if thumb and middle are very close
        close_bonus = max(0.0, (0.45 - min_dist) / 0.45)
        score += close_bonus * 0.10
        return min(1.0, score)

    def _check_F(self, f: ASLFeatures) -> float:
        """F — Index DOWN, Middle/Ring/Pinky UP. Thumb touches index."""
        fu, fc = f.fingers_up, f.finger_curl
        if fu[1]: return 0.0
        if not all(fu[2:5]): return 0.0
        
        if f.thumb_index_dist > 0.4:
            return 0.0
            
        score = 0.50
        score += sum((1.0 - fc[i]) for i in [2, 3, 4]) / 3.0 * 0.20
        score += fc[1] * 0.15
        
        close_bonus = max(0.0, (0.4 - f.thumb_index_dist) / 0.4)
        score += close_bonus * 0.15
        return min(1.0, score)

    def _check_E(self, f: ASLFeatures) -> float:
        """E — All fingers curled tightly, thumb folded across BELOW knuckles."""
        fu, fc = f.fingers_up, f.finger_curl
        if any(fu[1:5]): return 0.0
        
        # S has thumb_above_knuckles = True
        if f.thumb_above_knuckles:
            return 0.0
            
        # Tips should be somewhat close to thumb
        avg_tip_dist = sum([f.thumb_index_dist, f.thumb_middle_dist, f.thumb_ring_dist, f.thumb_pinky_dist]) / 4.0
        if avg_tip_dist > 0.6:
            return 0.0
            
        score = 0.60
        score += sum(fc[1:5]) / 4.0 * 0.20
        
        close_bonus = max(0.0, (0.6 - avg_tip_dist) / 0.6)
        score += close_bonus * 0.20
        return min(1.0, score)

    def _check_C(self, f: ASLFeatures) -> float:
        """C — Hand open, all fingers curved like a C."""
        fu, fc = f.fingers_up, f.finger_curl
        
        # If it's a tight O, it's not C
        if f.o_shape_score > 0.4:
            return 0.0
            
        # Fingers should be moderately curled
        avg_curl = sum(fc) / 5.0
        if avg_curl < 0.20 or avg_curl > 0.80:
            return 0.0
            
        # Thumb and index should be moderately separated (forming the C opening)
        if f.thumb_index_dist < 0.25:
            return 0.0
            
        # If thumb is touching middle/ring, it's a D, not C
        if min(f.thumb_middle_dist, f.thumb_ring_dist) < 0.35:
            return 0.0
            
        score = 0.55
        
        # Reward intermediate curl (closest to 0.5)
        curl_quality = 1.0 - (abs(avg_curl - 0.5) / 0.30)
        score += curl_quality * 0.20
        
        # Reward vertical alignment of the C (index above thumb)
        if f.lmList[8][2] < f.lmList[4][2]:
            score += 0.15
            
        # Base bonus
        score += 0.15
        return min(1.0, score)

    def _check_T(self, f: ASLFeatures) -> float:
        """T — Fist, thumb under index."""
        if any(f.fingers_up[1:5]): return 0.0
        
        pos = self._get_t_n_m_position(f)
        if pos != 'T':
            return 0.0
            
        # Penalty if thumb is wrapped horizontally across the hand (that's an S)
        lm = f.lmList
        dx = abs(lm[4][1] - lm[3][1])
        dy = abs(lm[4][2] - lm[3][2])
        if dx > dy * 1.5:
            return 0.0
            
        score = 0.70
        score += sum(f.finger_curl[1:5]) / 4.0 * 0.30
        
        # Bonus if thumb is pointing distinctly UP
        if dy > dx * 1.5:
            score += 0.15
            
        return min(1.0, score)

    def _check_M(self, f: ASLFeatures) -> float:
        """M — Fist, thumb under index, middle, ring."""
        if any(f.fingers_up[1:5]): return 0.0
        
        pos = self._get_t_n_m_position(f)
        if pos != 'M':
            return 0.0
            
        lm = f.lmList
        dx = abs(lm[4][1] - lm[3][1])
        dy = abs(lm[4][2] - lm[3][2])
        if dx > dy * 1.5:
            return 0.0
            
        score = 0.70
        score += sum(f.finger_curl[1:5]) / 4.0 * 0.30
        if dy > dx * 1.5:
            score += 0.15
        return min(1.0, score)

    def _check_N(self, f: ASLFeatures) -> float:
        """N — Fist, thumb under index, middle."""
        if any(f.fingers_up[1:5]): return 0.0
        
        pos = self._get_t_n_m_position(f)
        if pos != 'N':
            return 0.0
            
        lm = f.lmList
        dx = abs(lm[4][1] - lm[3][1])
        dy = abs(lm[4][2] - lm[3][2])
        if dx > dy * 1.5:
            return 0.0
            
        score = 0.70
        score += sum(f.finger_curl[1:5]) / 4.0 * 0.30
        if dy > dx * 1.5:
            score += 0.15
        return min(1.0, score)

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 4 — Complex / Nuanced (stubs)
    # ═════════════════════════════════════════════════════════════════════════
    def _check_K(self, f: ASLFeatures) -> float: return 0.0
    def _check_R(self, f: ASLFeatures) -> float: return 0.0
    def _check_X(self, f: ASLFeatures) -> float: return 0.0
    def _check_G(self, f: ASLFeatures) -> float: return 0.0
    def _check_H(self, f: ASLFeatures) -> float: return 0.0
    def _check_P(self, f: ASLFeatures) -> float: return 0.0
    def _check_Q(self, f: ASLFeatures) -> float: return 0.0

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 5 — Digits (stubs)
    # ═════════════════════════════════════════════════════════════════════════
    def _check_0(self, f: ASLFeatures) -> float: return 0.0
    def _check_2(self, f: ASLFeatures) -> float: return 0.0
    def _check_3(self, f: ASLFeatures) -> float: return 0.0
    def _check_4(self, f: ASLFeatures) -> float: return 0.0
    def _check_6(self, f: ASLFeatures) -> float: return 0.0
    def _check_7(self, f: ASLFeatures) -> float: return 0.0
    def _check_8(self, f: ASLFeatures) -> float: return 0.0
    def _check_9(self, f: ASLFeatures) -> float: return 0.0
