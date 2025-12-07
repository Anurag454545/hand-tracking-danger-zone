🖐️ Real-Time Hand Tracking Danger Zone Detection

This project is a real-time computer vision prototype developed as part of the Arvyax Internship Assignment.
It uses a live camera feed to track the user’s hand and detect when the hand approaches a virtual object on the screen. Based on the distance, the system dynamically classifies the interaction as SAFE, WARNING, or DANGER, and displays a clear on-screen alert: “DANGER DANGER”.

The solution is built using classical computer vision techniques only, without relying on MediaPipe, OpenPose, or any pose-detection APIs.

🚀 Features

✅ Real-time hand and fingertip tracking using OpenCV

✅ No pose-detection APIs used (MediaPipe/OpenPose not used)

✅ Virtual boundary displayed on screen

✅ Dynamic distance-based state detection

✅ Visual state overlay: SAFE, WARNING, DANGER

✅ Large “DANGER DANGER” alert when hand is too close

✅ CPU-only execution

✅ Achieves real-time performance (8+ FPS)

🧠 Approach Used

The system uses classical computer vision techniques:

Skin color segmentation using HSV + YCrCb color spaces

Morphological filtering for noise removal

Largest contour detection for hand segmentation

Fingertip estimation using contour geometry

Euclidean distance calculation between fingertip and virtual object

Threshold-based classification into SAFE, WARNING, and DANGER states

This approach ensures fast performance and avoids dependency on heavy AI models or cloud APIs.

📊 State Logic
Distance from Virtual Object	System State
Far from object	SAFE
Approaching object	WARNING
Very close / touching	DANGER

When in DANGER state, the system displays a bold red warning:

DANGER DANGER

🛠️ Tech Stack

Python 3

OpenCV

NumPy

📦 Installation

Install required dependencies using:

pip install opencv-python numpy

▶️ How to Run

Clone the repository and run:

python hand_tracking_poc.py --mirror


If your camera does not open, try:

python hand_tracking_poc.py --camera 0


or

python hand_tracking_poc.py --camera 1


Press ESC to exit the application.

📷 Output Display

The live camera feed displays:

Detected hand contour

Fingertip marker

Virtual object boundary

Current system state (SAFE / WARNING / DANGER)

“DANGER DANGER” alert in real time when triggered

⚡ Performance

Runs fully on CPU

Achieves real-time performance (8+ FPS) under normal lighting conditions

🎯 Objective of the Project

This prototype demonstrates:

Real-time vision processing

Classical computer vision techniques

Distance-based interaction logic

Safety-zone simulation using hand tracking

It is designed as a Proof of Concept (POC) for interactive spatial safety detection.

👨‍💻 Author

Anurag Kumar
Engineering Student | AI & ML Enthusiast

📌 Note

Lighting conditions and background simplicity can affect skin segmentation performance. For best results, use proper lighting and a non-cluttered background.
