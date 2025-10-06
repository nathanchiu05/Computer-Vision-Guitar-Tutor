# Computer Vision Guitar Tutor

Computer vision can bridge the gap between expensive professional lessons and the often unstructured path of free self-teaching.

Chordially is a computer vision–based guitar tutor designed to make learning more accessible and interactive. Leveraging real-time hand tracking and fretboard mapping, it helps beginners visualize and practice the fundamental chords that serve as the foundation for most songs.

Instead of passively watching tutorials, learners receive instant feedback on finger placement, string accuracy, and chord correctness. The system overlays chord diagrams directly on the screen, guiding users as they adjust their hand positions until the shape is correct.

---

# Features
- Real-time hand tracking with MediaPipe Hands  
- Fret board mapping using ArUco markers for orientation  
- Finger position detection (index, string, fret)  
- Chord validation**: checks if the chord matches standard shapes  
- Chord diagram overlay for visual guidance  
- Keyboard controls to switch between chords (e.g., `C`, `Am`, etc.)

---

#Tech Stack
- **Python 3.10+**
- [OpenCV](https://opencv.org/) – computer vision & marker detection  
- [MediaPipe](https://developers.google.com/mediapipe) – hand tracking  
- NumPy – array operations  
- Custom Graphics Module – draws chord diagrams on screen  
