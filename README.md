# Real-Time A.I. Hand Tracking & Gesture Analysis Suite

A premium, production-grade Computer Vision and Human-Computer Interaction (HCI) framework developed from scratch. This suite leverages **OpenCV** and **MediaPipe** to perform real-time hand detection, landmark extraction, and spatial analysis to build intuitive gesture-controlled interfaces.

---

## Core Computer Vision Concepts Developed From Scratch

Rather than relying purely on pre-built API black boxes, this project implements the mathematical, spatial, and logical foundations of gesture tracking and application control. Below are the key engineering challenges solved in this codebase:

### 1. Coordinate Space Transformation & Landmark Extraction
MediaPipe outputs hand landmark coordinates (`x`, `y`, `z`) normalized in the range `[0.0, 1.0]` relative to the image dimensions. To make these usable for pixel-based drawing or clicking, we translate them to screen coordinates:
$$cx = \text{int}(lm.x \times w)$$
$$cy = \text{int}(lm.y \times h)$$
*Where $w$ and $h$ are the camera frame width and height.*

### 2. Spatial Distance Analysis (Euclidean Metrics)
To detect clicks or pinch gestures, we compute the Euclidean distance between arbitrary fingertips (e.g., Thumb Tip `ID 4` and Index Tip `ID 8`):
$$\text{Distance} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$
This distance is calculated in real-time using `math.hypot(x2 - x1, y2 - y1)` and used to trigger UI events when it falls below a specific threshold (e.g., $< 20$ pixels for mouse click, or dynamically mapped for volume adjustment).

### 3. Linear Range Interpolation
Different systems operate on different scales. To translate hand movements into OS controls, we implement linear interpolation (`np.interp`) to map values between disparate mathematical domains:
*   **Volume Control**: Maps finger pinch distance `[40, 200]` pixels to Windows Audio Endpoint levels `[-65.0, 0.0]` dB.
*   **Virtual Mouse**: Maps index fingertip coordinates within a reduced bounding box to the user's primary monitor resolution dimensions `(wScr, hScr)`.

### 4. Exponential Smoothing (Jitter Reduction)
Webcam sensors and lighting variances cause raw hand landmark positions to fluctuate, leading to a shaky cursor. To solve this, an **Exponential Moving Average (EMA)** low-pass filter is implemented:
$$X_{\text{smooth}} = X_{\text{prev}} + \frac{X_{\text{target}} - X_{\text{prev}}}{\text{smoothening}}$$
$$Y_{\text{smooth}} = Y_{\text{prev}} + \frac{Y_{\text{target}} - Y_{\text{prev}}}{\text{smoothening}}$$
*Where `smoothening` is a configurable factor (e.g., `6`).* This smooths out high-frequency noise and results in premium, butter-smooth cursor motion.

### 5. Advanced Image Masking & Bitwise Operations
To create a real-time painting canvas without lag or translucency issues:
1.  Drawings are captured on an independent black canvas (`imgCanvas`).
2.  The canvas is converted to grayscale, and an **Inverse Binary Threshold** mask is generated. This creates a black silhouette of the drawings on a white background.
3.  A bitwise `AND` merges the real-time webcam frame with this inverse mask, carving out precise transparent "channels" in the live video.
4.  A bitwise `OR` overlays the colored canvas drawings directly into those channels.

---

## MediaPipe Hand Landmarks Mapping Reference

The project parses 21 coordinate points on the hand to calculate gesture configurations:

```text
            8 (Index Tip)    12 (Middle Tip)    16 (Ring Tip)    20 (Pinky Tip)
                |                |                  |                |
            7 (Index PIP)    11 (Middle PIP)    15 (Ring PIP)    19 (Pinky PIP)
                |                |                  |                |
            6 (Index MCP)    10 (Middle MCP)    14 (Ring MCP)    18 (Pinky MCP)
     4 (Thumb)  |                |                  |                |
        \      /                 \_________________/________________/
      3 (Thumb)                                   |
         \                                   17 (Pinky MCP base)
        2 (Thumb)                                 |
           \                                     /
          1 (Thumb MCP)                        /
             \                                /
              \______________________________/
                             |
                          0 (Wrist)
```

---

## Component Breakdown

### 1. 🛠️ The Core Engine: `HandTrackingModule.py`
The architectural backbone of the entire suite. It encapsulates the MediaPipe initialization and exposes high-level utility APIs for other sub-applications.
*   **`handDetector` Class**: Configures detection confidence, tracking confidence, complex models, and maximum hands.
*   **`findHands(img, draw=True)`**: Preprocesses the image (converts from OpenCV's BGR format to MediaPipe's RGB format) and extracts hand nodes.
*   **`findPosition(img, handNo=0, draw=True)`**: Converts raw landmark values to pixel locations, calculates a tight bounding box around the active hand, and returns a detailed landmark list (`lmList`).
*   **`fingersUp()`**: Evaluates which fingers are raised. For standard fingers, it checks if the tip landmark is higher than the joint below it. For the thumb, it compares the horizontal location relative to the palm center to account for side-to-side hand orientation.
*   **`findDistance(p1, p2, img)`**: Computes the Euclidean distance between two specific landmarks and draws visual aids (connecting lines and centers).

### 2. 🧪 Proof of Concept: `HandTrackingMin.py`
A lightweight, single-script pipeline created as the starting blueprint. Used to test basic FPS tracking, cv2 camera captures, and verify the correct initialization of the MediaPipe pipeline.

### 3. 🔊 Gestural OS HUD: `VolumeHandControl.py`
A hardware-integration utility that translates hand postures into physical OS system volume adjustments.
*   Uses `pycaw` (Python Common Audio Windows) to hook into core system endpoints.
*   Tracks the distance between the **Thumb (ID 4)** and **Index (ID 8)** fingertips.
*   Includes custom UI elements, drawing a sleek vertical volume slider bar and live percentages on screen.
*   Features threshold indicators: turns grey when audio is close to muted ($<40$px) and turns green when set to maximum volume ($>200$px).

### 4. 🎨 Interactive Air-Drawing Studio: `VirtualPainter.py`
An interactive tool that lets you paint in three-dimensional air space, choosing colors and erasing elements dynamically.
*   **Dual-Gesture Interaction**:
    *   *Selection Mode* (Index & Middle fingers raised): Used to navigate the toolbar at the top of the frame. Hovering coordinates over different areas changes the drawing color (Yellow, Purple, Green, Blue) or selects the Eraser.
    *   *Drawing Mode* (Only Index finger raised): Draws continuous paths on the screen by storing the previous index coordinate and drawing lines on a virtual canvas.
*   **State Persistence**: Uses a double-buffer architecture (frame buffer + canvas buffer) to ensure paint strokes persist across frames even when hands are temporarily out of view.

### 5. 🖱️ High-Precision Virtual Cursor: `VirtualMouseProject.py`
Replaces a physical hardware mouse with hand gestures, supporting smooth pointer movement and left-clicking.
*   **Screen Space Mapping**: Translates finger tracking coordinates to screen-resolution pixels using `autopy`.
*   **Active Boundary Frame**: Restricts the tracking region (`framR = 100`) so users do not have to make exaggerated hand sweeps near the edge of the webcam's field of view.
*   **Smoothened Motion**: Employs low-pass filtering to neutralize micro-jitters from hand tremors.
*   **Click Action**: Triggered by bringing the Index and Middle fingertips close together ($<20$px), simulating a physical click.

### 6. 🔢 Gesture Recognizer: `FingerCountingProject.py`
A computer vision model mapping system that identifies the count of raised fingers (0 to 5) and displays matching hand posture diagrams.
*   Loads overlay assets from the [FingerImages](file:///c:/Users/User/Documents/GitHub/HandTrackingProject/FingerImages) folder.
*   Calculates the active finger count in real-time, displaying both a digital HUD text box and a visual graphical card on the camera feed.

---

## 🛠️ Required Python Modules and Setup

Make sure you are using **Python 3.8+** (recommended **Python 3.10** or **3.11** for library compatibility).

### Dependencies
```bash
pip install opencv-python numpy mediapipe autopy pycaw
```

*   `opencv-python` (`4.12.0.88`): Frame capture, image pre-processing, rendering HUD, and drawing.
*   `numpy` (`1.26.4`): Numerical transformations and grid matrix operations for canvas masking.
*   `mediapipe` (`0.10.21`): Machine learning model execution for hand landmark regression.
*   `autopy` (`4.0.1`): Cross-platform OS mouse control interface.
*   `pycaw`: Windows Core Audio API binding.

---

## 🚀 Running the Projects

You can run each project individually. Position your hand clearly in front of your system webcam.

1.  **Run the core test script**:
    ```bash
    python HandTrackingModule.py
    ```
2.  **Launch the Gesture Mouse**:
    ```bash
    python VirtualMouseProject.py
    ```
3.  **Launch the Air-Drawing Studio**:
    ```bash
    python VirtualPainter.py
    ```
4.  **Control system volume via gestures**:
    ```bash
    python VolumeHandControl.py
    ```
5.  **Run the Finger Counter**:
    ```bash
    python FingerCountingProject.py
    ```

---

*Developed with ❤️ as a custom hand-tracking human-computer interaction workspace.*
