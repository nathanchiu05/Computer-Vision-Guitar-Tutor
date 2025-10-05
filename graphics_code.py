import cv2

# ============================================================================
# CHORD LIBRARY - Data structure for all chords
# ============================================================================
CHORD_LIBRARY = {
    "C": {"frets": [None, 3, 2, 0, 1, 0], "fingers": [None, 3, 2, None, 1, None], "name": "C Major"},
    "D": {"frets": [None, None, 0, 2, 3, 2], "fingers": [None, None, None, 1, 3, 2], "name": "D Major"},
    "E": {"frets": [0, 2, 2, 1, 0, 0], "fingers": [None, 2, 3, 1, None, None], "name": "E Major"},
    "G": {"frets": [3, 2, 0, 0, 0, 3], "fingers": [3, 2, None, None, None, 4], "name": "G Major"},
    "A": {"frets": [None, 0, 2, 2, 2, 0], "fingers": [None, None, 1, 2, 3, None], "name": "A Major"},
    "Em": {"frets": [0, 2, 2, 0, 0, 0], "fingers": [None, 2, 3, None, None, None], "name": "E Minor"},
    "Am": {"frets": [None, 0, 2, 2, 1, 0], "fingers": [None, None, 2, 3, 1, None], "name": "A Minor"},
    "Dm": {"frets": [None, None, 0, 2, 3, 1], "fingers": [None, None, None, 2, 4, 1], "name": "D Minor"},
}

# ============================================================================
# DRAW CHORD DIAGRAM - Draws the chord picture in a panel
# ============================================================================
def draw_chord_diagram(frame, x, y, width, height, show_instructions=True, current_chord=None):
    if not current_chord:
        return
    
    chord_info = CHORD_LIBRARY[current_chord]
    
    # Draw background panel
    cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    
    # Draw chord name
    cv2.putText(frame, chord_info['name'], (x + 10, y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Calculate diagram dimensions
    diagram_x = x + 20
    diagram_y = y + 45
    diagram_w = width - 40
    diagram_h = height - 65
    
    # Draw strings (vertical lines)
    string_spacing = diagram_w // (6 - 1)  # 6 strings
    for i in range(6):
        sx = diagram_x + i * string_spacing
        cv2.line(frame, (sx, diagram_y), (sx, diagram_y + diagram_h), (200, 200, 200), 1)
    
    # Draw frets (horizontal lines)
    fret_spacing = diagram_h // 5
    for i in range(6):
        fy = diagram_y + i * fret_spacing
        thickness = 3 if i == 0 else 1  # Nut is thicker
        cv2.line(frame, (diagram_x, fy), (diagram_x + diagram_w, fy), (200, 200, 200), thickness)
    
    # Draw finger positions on diagram
    for string_idx, fret in enumerate(chord_info['frets']):
        sx = diagram_x + string_idx * string_spacing
        
        if fret is None:
            cv2.putText(frame, "X", (sx - 5, diagram_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
        elif fret == 0:
            cv2.circle(frame, (sx, diagram_y - 10), 6, (100, 255, 100), 2)
        else:
            fy = diagram_y + (fret - 0.5) * fret_spacing
            cv2.circle(frame, (int(sx), int(fy)), 10, (0, 255, 255), -1)
            cv2.circle(frame, (int(sx), int(fy)), 10, (255, 255, 255), 2)
            finger_num = chord_info['fingers'][string_idx]
            if finger_num:
                cv2.putText(frame, str(finger_num), (int(sx) - 5, int(fy) + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
    
    # Draw instructions
    if show_instructions:
        instructions = []
        for string_idx, fret in enumerate(chord_info['frets']):
            finger = chord_info['fingers'][string_idx]
            if fret is not None and fret > 0 and finger:
                instructions.append(f"Finger {finger} on string {string_idx+1} fret {fret}")
        if not instructions:
            instructions = ["Open strings or muted"]
        for i, line in enumerate(instructions):
            cv2.putText(frame, line, (x + 10, y + height - 35 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
