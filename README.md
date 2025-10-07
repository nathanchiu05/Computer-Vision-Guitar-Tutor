@ -1,176 +0,0 @@
# Computer Vision Guitar Tutor

A real-time guitar learning application that uses computer vision to track your hand positions and provide visual feedback on chord accuracy. The system uses ArUco markers to map the guitar fretboard and MediaPipe for hand tracking to help you learn guitar chords with immediate visual feedback.

## Features

- **Real-time Hand Tracking**: Uses MediaPipe to track fingertip positions in real-time
- **Guitar Fretboard Mapping**: ArUco markers detect and map the guitar fretboard for accurate positioning
- **Chord Recognition**: Visual feedback showing correct finger positions for various guitar chords
- **Accuracy Scoring**: Real-time accuracy percentage based on finger placement
- **Interactive Chord Selection**: Switch between different chords using keyboard shortcuts
- **Visual Chord Diagrams**: On-screen chord diagrams with finger placement instructions


## Requirements

### Hardware
- Webcam or camera (the application uses camera index 1 by default)
- Guitar with ArUco markers attached to the fretboard
- ArUco markers (4x4_1000 dictionary, IDs: 0, 1, 2, 3)

### Software Dependencies
- Python 3.7+
- OpenCV 4.8.1.78
- MediaPipe
- NumPy 1.24.3
- Pandas 2.0.3

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Computer-Vision-Guitar-Tutor.git
   cd Computer-Vision-Guitar-Tutor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up ArUco markers**:
   - Print the ArUco markers from the `arucos/` folder
   - Attach them to your guitar fretboard in the following positions:
     - Marker ID 0: Top-left corner
     - Marker ID 1: Top-right corner  
     - Marker ID 2: Bottom-right corner
     - Marker ID 3: Bottom-left corner

## Usage

1. **Start the application**:
   ```bash
   python main.py
   ```

2. **Position your guitar**:
   - Ensure the ArUco markers are visible to the camera
   - The system will automatically detect and map the fretboard

3. **Select a chord**:
   - Press number keys 1-8 to switch between different chords
   - The current chord will be displayed on screen

4. **Play the chord**:
   - Place your fingers on the guitar according to the on-screen instructions
   - The system will track your fingertip positions and show accuracy
   - Yellow circles indicate correct finger positions
   - Green circles show your actual fingertip positions

5. **Exit**:
   - Press `ESC` to exit the application

## How It Works

### 1. Fretboard Detection (`map_fret_board.py`)
- Uses ArUco markers to detect the guitar fretboard
- Calculates fret and string positions based on marker locations
- Provides real-time mapping of the physical guitar to screen coordinates

### 2. Hand Tracking (`map_hands.py`)
- Uses MediaPipe to detect and track hand landmarks
- Tracks fingertip positions for index, middle, ring, and pinky fingers
- Applies smoothing using a 5-frame history buffer

### 3. Chord Matching (`match_chord.py`)
- Contains a database of guitar chords from `GuitarChords.csv`
- Matches detected finger positions to known chord patterns
- Provides chord recognition functionality

### 4. Visual Feedback (`graphics_code.py`)
- Renders chord diagrams on screen
- Shows finger placement instructions
- Displays accuracy scoring and visual indicators

### 5. Main Application (`main.py`)
- Coordinates all components
- Handles real-time video processing
- Manages user input and chord selection
- Calculates and displays accuracy metrics

## File Structure

```
Computer-Vision-Guitar-Tutor/
├── main.py                 # Main application entry point
├── map_hands.py           # Hand tracking using MediaPipe
├── map_fret_board.py      # ArUco marker detection and fretboard mapping
├── match_chord.py         # Chord recognition and matching
├── graphics_code.py       # Visual rendering and chord diagrams
├── GuitarChords.csv       # Database of guitar chord fingerings
├── requirements.txt       # Python dependencies
├── arucos/               # ArUco marker images
│   ├── 4x4_1000-0.svg
│   ├── 4x4_1000-1.svg
│   ├── 4x4_1000-2.svg
│   └── 4x4_1000-3.svg
└── README.md             # This file
```

## Configuration

### Camera Settings
- Default camera index: 1 (change in `main.py` line 19)
- To use a different camera, modify: `cap = cv2.VideoCapture(0)` (for camera index 0)

### Accuracy Threshold
- Default accuracy threshold: 60 pixels (modify in `main.py` line 159)
- Adjust `max_distance` parameter to change sensitivity

### Chord Library
- Add new chords by modifying `CHORD_LIBRARY` in `graphics_code.py`
- Format: `"ChordName": {"frets": [fret_positions], "fingers": [finger_numbers], "name": "Display Name"}`

## Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Check camera index in `main.py`
   - Ensure camera is not being used by another application

2. **ArUco markers not detected**:
   - Ensure good lighting conditions
   - Check marker print quality and positioning
   - Verify markers are from the 4x4_1000 dictionary

3. **Hand tracking issues**:
   - Ensure good lighting on your hands
   - Keep hands within camera view
   - Avoid cluttered backgrounds

4. **Low accuracy scores**:
   - Adjust the `max_distance` threshold in `main.py`
   - Ensure proper camera positioning
   - Check ArUco marker calibration

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Additional chord support
- Improved accuracy algorithms
- UI/UX enhancements
- Performance optimizations

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Built using OpenCV for computer vision
- Hand tracking powered by MediaPipe
- ArUco marker detection for precise fretboard mapping
- Chord data sourced from guitar chord databases

