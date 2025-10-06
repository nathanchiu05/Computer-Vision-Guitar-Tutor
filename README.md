# Computer Vision Guitar Tutor

Computer vision can bridge the gap between expensive professional lessons and the often unstructured path of free self-teaching.

Chordially is a computer vision–based guitar tutor designed to make learning more accessible and interactive. Leveraging real-time hand tracking and fretboard mapping, it helps beginners visualize and practice the fundamental chords that serve as the foundation for most songs.

Instead of passively watching tutorials, learners receive instant feedback on finger placement, string accuracy, and chord correctness. The system overlays chord diagrams directly on the screen, guiding users as they adjust their hand positions until the shape is correct.

---
# How it Works
Chordially uses MediaPipe Hands to detect the position of your fingers in real time through a webcam feed. With the help of ArUco markers placed around the guitar fretboard, the system calibrates string and fret positions to create a virtual map of the instrument. Each detected fingertip is then matched to its corresponding string–fret coordinate, which is compared against predefined chord shapes. If the placement matches a chord (e.g., C major, A minor), the program overlays a visual chord diagram on the screen and provides instant feedback, helping the player adjust until the correct shape is formed.

---
# ArUcos
To accurately understand where a player’s fingers land, the application needs to know the orientation and boundaries of the fretboard. We use ArUco markers, simple black-and-white square patterns that can be easily generated and printed, to anchor the fretboard in the camera’s view. These markers act as reference points, allowing the program to calibrate strings and frets regardless of camera angle or lighting. 

For users without immediate access to a guitar, the application offers the ability to practice on a virtual fretboard. To set this up, print four ArUco markers and affix them securely to the corners of a rectangular object such as a piece of cardboard or a notebook. When positioned within the camera’s view, the software interprets this rectangle as a surrogate fretboard. This enables users to simulate finger placement, follow visual chord diagrams, and receive feedback on chord accuracy, thereby facilitating effective practice even in the absence of a physical instrument.

___

# Features
- Real-time hand tracking with MediaPipe Hands  
- Fret board mapping using ArUco markers for orientation  
- Finger position detection (index, string, fret)  
- Chord validation**: checks if the chord matches standard shapes  
- Chord diagram overlay for visual guidance  
- Keyboard controls to switch between chords (e.g., `C`, `Am`, etc.)

---

# Tech Stack
- **Python 3.10+**
- [OpenCV](https://opencv.org/) – computer vision & marker detection  
- [MediaPipe](https://developers.google.com/mediapipe) – hand tracking  
- NumPy – array operations  
- Custom Graphics Module – draws chord diagrams on screen  
