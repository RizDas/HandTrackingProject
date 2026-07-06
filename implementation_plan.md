# ASL Alphabet & Digit Sign Language Recognizer

Build a high-accuracy, letter-by-letter ASL static hand-sign recognizer on top of the existing `HandTrackingModule.py`. Each letter (A–Z) and digit (0–9) is added one at a time, verified for accuracy before the next is introduced.

---

## Architecture Overview

```
HandTrackingModule.py        ← existing landmark engine (unchanged)
        │
        ▼
ASLFeatureExtractor.py       ← [NEW] rich geometric feature layer
        │
        ▼
ASLRecognizer.py             ← [NEW] per-letter rule classifier
        │
        ▼
ASLRecognizerApp.py          ← [NEW] live camera demo app
```

---

## Why Rule-Based (Not ML)?

| | Rule-Based | ML Classifier |
|---|---|---|
| Accuracy | High (deterministic) | Variable (needs 1000s of samples) |
| No training data needed | ✅ | ❌ |
| Explainable/debuggable | ✅ | ❌ |
| Works immediately | ✅ | ❌ |
| Handles all 26 letters | ✅ with careful rules | ❌ without data |

We will use **rich geometric features** (angles, distances, curl ratios, spread) beyond just `fingersUp()` — this is the key to extreme accuracy.

---

## Feature Extraction Layer — `ASLFeatureExtractor.py`

### Landmark IDs (MediaPipe 21-point model)
```
Wrist: 0
Thumb:  CMC=1, MCP=2,  IP=3,   TIP=4
Index:  MCP=5, PIP=6,  DIP=7,  TIP=8
Middle: MCP=9, PIP=10, DIP=11, TIP=12
Ring:   MCP=13,PIP=14, DIP=15, TIP=16
Pinky:  MCP=17,PIP=18, DIP=19, TIP=20
```

### Features Computed
| Feature | Description | Used For |
|---|---|---|
| `fingers_up[5]` | Boolean: is each finger extended? | Base classification |
| `finger_curl[5]` | 0.0–1.0 curl ratio per finger | A vs S vs E vs M/N/T |
| `fingertip_dist(a,b)` | Pixel distance between landmarks | F/O/D shapes |
| `finger_spread[4]` | Lateral distance between adjacent tips | U vs V, B vs W |
| `thumb_across_palm` | Is thumb tip over middle of palm? | A vs S, M/N/T |
| `thumb_between_fingers` | Thumb tip y between index & middle | T detection |
| `angle_at_joint(a,b,c)` | Angle in degrees at joint b | X (hooked index) |
| `hand_size` | Diagonal of bounding box | Normalization |
| `index_pointing_dir` | Vertical vs horizontal | G vs 1, H |
| `o_shape_score` | Closeness of all tips to center | O vs C vs E |

---

## Letter Implementation Order (Easy → Hard)

Letters are grouped by how many unique features they require. Each group is built, tested, and verified before the next.

### 🟢 Phase 1 — Highly Distinct (5 letters)
These are unambiguous from basic finger-up state + 1-2 features:

| Letter | Finger State | Key Distinguisher |
|---|---|---|
| **I** | `[0,0,0,0,1]` | Only pinky up |
| **L** | `[1,1,0,0,0]` | Thumb out + only index up |
| **Y** | `[1,0,0,0,1]` | Thumb + pinky only |
| **B** | `[0,1,1,1,1]` | 4 fingers fully up, thumb tucked across palm |
| **5** | `[1,1,1,1,1]` | All 5 up AND spread apart |

### 🟡 Phase 2 — Distinct with 1 extra feature (7 letters)
| Letter | Finger State | Key Distinguisher |
|---|---|---|
| **1** | `[0,1,0,0,0]` | Only index up, pointing straight up |
| **W** | `[0,1,1,1,0]` | Index+Middle+Ring up, spread |
| **U** | `[0,1,1,0,0]` | Index+Middle up, close together |
| **V** | `[0,1,1,0,0]` | Index+Middle up, spread apart |
| **S** | `[0,0,0,0,0]` | Fist, thumb OVER fingers |
| **A** | `[0,0,0,0,0]` | Fist, thumb beside (not over) |
| **O** | `[0,0,0,0,0]` | All curve → tips all close to center |

### 🟠 Phase 3 — Requires 2+ extra features (7 letters)
| Letter | Finger State | Key Distinguisher |
|---|---|---|
| **D** | `[0,1,0,0,0]` | Index up, others curl round thumb (circle) |
| **F** | `[0,0,1,1,1]` | Index+thumb tip touch, middle/ring/pinky up |
| **E** | `[0,0,0,0,0]` | All fingers curl DOWN, touching or near thumb |
| **C** | `[0,0,0,0,0]` | All fingers curve open (C not closed O) |
| **T** | `[0,0,0,0,0]` | Fist, thumb tip visible between index and middle |
| **M** | `[0,0,0,0,0]` | 3 fingers folded OVER tucked thumb |
| **N** | `[0,0,0,0,0]` | 2 fingers folded OVER tucked thumb |

### 🔴 Phase 4 — Complex / Nuanced (7 letters)
| Letter | Key Distinguisher |
|---|---|
| **K** | Index + middle up, thumb touching middle-side, angled |
| **R** | Index + middle up but crossed |
| **X** | Only index up, but hooked/bent at DIP |
| **G** | Index + thumb horizontal pointing same direction |
| **H** | Index + middle horizontal pointing together |
| **P** | Like K but hand rotated downward |
| **Q** | Like G but pointing down |

### ⚪ Phase 5 — Digits (0–9)
| Digit | Mapping |
|---|---|
| **0** | Same as O |
| **1** | Same as 1 (only index up) |
| **2** | Same as V |
| **3** | Thumb + index + middle up |
| **4** | 4 fingers up (no thumb) |
| **5** | All 5 up |
| **6** | Pinky + thumb touch, 3 others up |
| **7** | Ring + thumb touch, others up |
| **8** | Middle + thumb touch, others up |
| **9** | Index + thumb circle (like F but different curl) |

---

## Files to Create

### [NEW] `ASLFeatureExtractor.py`
Pure geometry layer. Input: `lmList` (21 landmarks). Output: a rich `ASLFeatures` dataclass.

**Key methods:**
- `extract(lmList)` → `ASLFeatures`
- `finger_curl(lmList, finger_id)` — ratio of how curled a finger is
- `angle_at_joint(lmList, a, b, c)` — angle in degrees
- `landmark_dist(lmList, id1, id2)` — normalized by hand size
- `thumb_position(lmList)` — `'across_palm'`, `'beside_fist'`, `'extended'`

### [NEW] `ASLRecognizer.py`
Classifier module. Input: `lmList`. Output: `(letter, confidence, all_scores_dict)`.

**Design:**
- Each letter has its own `_check_X(features)` function returning 0.0–1.0 confidence
- `recognize(lmList)` runs all enabled letter checks, returns highest-confidence match
- Letters are added incrementally — new letters are simply new `_check_X` functions
- Minimum confidence threshold of `0.70` to prevent false positives

### [NEW] `ASLRecognizerApp.py`
Live demo application. Features:
- Real-time camera feed with landmark overlay
- Large, centered letter display with confidence bar
- Color-coded: 🟢 high confidence (>0.85), 🟡 medium (0.70–0.85)
- Shows which phase of letters are currently active
- Displays all feature values for debugging

---

## Accuracy Strategy

1. **Normalized distances** — all distances divided by hand bounding-box diagonal to be scale-invariant
2. **Curl ratios** — uses 3-point angle at each joint (not just tip vs PIP) for precise curl detection
3. **Soft confidence scores** — each letter returns 0–1 score, best match wins; tie → `None`
4. **Disambiguation rules** — where letters share finger states (A/E/S/M/N/T/O/C), extra features provide tiebreakers
5. **Minimum threshold** — no letter output if max confidence < 0.70 (avoids junk detections)
6. **Temporal smoothing** — letter shown only after it persists for 3 consecutive frames (avoids flicker)

---

## Verification Plan

### Per-Phase Testing
- After each phase, run `ASLRecognizerApp.py` and hold each sign for visual verification
- Check confidence values are > 0.80 for correct sign, < 0.40 for all others

### Confusion Matrix (Phase 4+)
- Most-confused pairs: U vs V, A vs S vs E vs M/N/T, D vs 1, O vs 0 vs C
- If confusion > 10%, tighten the distinguishing feature threshold

### Manual Verification
- User holds each letter steadily while observing the live confidence display
- Adjust thresholds until accuracy is acceptable before moving to next phase
