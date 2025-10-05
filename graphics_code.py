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
def draw_chord_diagram(self, frame, x, y, width, height, show_instructions=True):
    if not self.current_chord:
        return
    
    chord_info = CHORD_LIBRARY[self.current_chord]
    
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
    string_spacing = diagram_w // (NUM_STRINGS - 1)
    for i in range(NUM_STRINGS):
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
            # Draw X for muted string
            cv2.putText(frame, "X", (sx - 5, diagram_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
        elif fret == 0:
            # Draw O for open string
            cv2.circle(frame, (sx, diagram_y - 10), 6, (100, 255, 100), 2)
        else:
            # Draw finger position
            fy = diagram_y + (fret - 0.5) * fret_spacing
            cv2.circle(frame, (int(sx), int(fy)), 10, (0, 255, 255), -1)  # Yellow filled
            cv2.circle(frame, (int(sx), int(fy)), 10, (255, 255, 255), 2)  # White outline
            
            # Draw finger number
            finger_num = chord_info['fingers'][string_idx]
            if finger_num:
                cv2.putText(frame, str(finger_num), (int(sx) - 5, int(fy) + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
    
    # Draw instructions (optional)
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

# ============================================================================
# DRAW FRETBOARD - Draws the green grid overlay on the actual fretboard
# ============================================================================
def draw_fretboard(image, region: FretboardRegion, debug=False, chord=None, show_grid_fingers=True):
    # Using quadrilateral perspective (tracked fretboard)
    if hasattr(region, 'quad_corners') and region.quad_corners is not None:
        corners = region.quad_corners.astype(np.int32)
        
        # Draw fretboard outline
        cv2.polylines(image, [corners], True, (0, 255, 0), 2)
        
        # Draw strings (horizontal lines across perspective)
        for i in range(NUM_STRINGS):
            t = (i + 0.5) / NUM_STRINGS
            left_point = corners[0, 0] + t * (corners[3, 0] - corners[0, 0])
            right_point = corners[1, 0] + t * (corners[2, 0] - corners[1, 0])
            cv2.line(image, 
                    (int(left_point[0]), int(left_point[1])),
                    (int(right_point[0]), int(right_point[1])),
                    (0, 200, 0), 1)
            
            # Draw string name label
            label_x = int(left_point[0] - 25)
            label_y = int(left_point[1] + 5)
            cv2.putText(image, STRING_NAMES[i], (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw frets (vertical lines)
        for fret in range(1, NUM_FRETS):
            t = fret / NUM_FRETS
            top = corners[0, 0] + t * (corners[1, 0] - corners[0, 0])
            bottom = corners[3, 0] + t * (corners[2, 0] - corners[3, 0])
            cv2.line(image, 
                     (int(top[0]), int(top[1])),
                     (int(bottom[0]), int(bottom[1])),
                     (0, 200, 0), 2)
            
            # Draw fret number
            cv2.putText(image, str(fret), (int(top[0]) - 8, int(top[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw nut (thicker left edge)
        cv2.line(image, 
                 (int(corners[0, 0][0]), int(corners[0, 0][1])),
                 (int(corners[3, 0][0]), int(corners[3, 0][1])),
                 (0, 255, 0), 4)
        
        # ============================================================================
        # DRAW CHORD FINGER POSITIONS ON THE GRID (Yellow circles)
        # ============================================================================
        if chord is not None:
            chord_info = CHORD_LIBRARY[chord]
            
            for string_idx, (fret, finger_num) in enumerate(zip(chord_info['frets'], chord_info['fingers'])):
                if fret is not None and fret > 0 and finger_num:
                    # Calculate position on perspective grid
                    t = (string_idx + 0.5) / NUM_STRINGS
                    string_start = corners[0, 0] + t * (corners[3, 0] - corners[0, 0])
                    string_end = corners[1, 0] + t * (corners[2, 0] - corners[1, 0])
                    
                    fret_t = fret / NUM_FRETS
                    sx = int(string_start[0] + (string_end[0] - string_start[0]) * fret_t)
                    sy = int(string_start[1] + (string_end[1] - string_start[1]) * fret_t)
                    
                    # Draw yellow circle for finger position
                    cv2.circle(image, (sx, sy), 14, (0, 255, 255), -1)  # Yellow filled
                    cv2.circle(image, (sx, sy), 14, (255, 255, 255), 2)  # White outline
                    
                    # Draw finger number on the circle
                    if show_grid_fingers:
                        cv2.putText(image, str(finger_num), (sx - 7, sy + 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                elif fret == 0:
                    # Draw open string indicator (O)
                    t = (string_idx + 0.5) / NUM_STRINGS
                    string_start = corners[0, 0] + t * (corners[3, 0] - corners[0, 0])
                    sx = int(string_start[0])
                    sy = int(string_start[1])
                    cv2.circle(image, (sx, sy - 18), 8, (100, 255, 100), 2)
                
                elif fret is None:
                    # Draw muted string indicator (X)
                    t = (string_idx + 0.5) / NUM_STRINGS
                    string_start = corners[0, 0] + t * (corners[3, 0] - corners[0, 0])
                    sx = int(string_start[0])
                    sy = int(string_start[1])
                    cv2.putText(image, "X", (sx - 8, sy - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
    
    # Fallback: Simple rectangle mode (non-perspective)
    else:
        tl_x, tl_y = region.top_left
        br_x, br_y = region.bottom_right
        
        # Draw rectangle outline
        cv2.rectangle(image, region.top_left, region.bottom_right, (0, 255, 0), 2)
        
        # Draw strings
        string_height = (br_y - tl_y) / NUM_STRINGS
        for i in range(NUM_STRINGS + 1):
            y = int(tl_y + i * string_height)
            cv2.line(image, (tl_x, y), (br_x, y), (0, 200, 0), 1)
            
            if i < NUM_STRINGS:
                mid_y = int(tl_y + (i + 0.5) * string_height)
                cv2.putText(image, STRING_NAMES[i], (tl_x - 25, mid_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw frets
        if region.fret_markers:
            for i, (x, y, w, h) in enumerate(region.fret_markers):
                cv2.line(image, (x, tl_y), (x, br_y), (0, 200, 0), 2)
                cv2.putText(image, str(i + 1), (x - 8, tl_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# ============================================================================
# TEXT INSTRUCTIONS (displayed below chord diagram in main loop)
# ============================================================================
# This code goes in the main() function where the tutor panel is drawn:

if tutor.current_chord:
    chord_info = CHORD_LIBRARY[tutor.current_chord]
    instructions = []
    
    # Generate instruction text
    for string_idx, fret in enumerate(chord_info['frets']):
        finger = chord_info['fingers'][string_idx]
        if fret is not None and fret > 0 and finger:
            instructions.append(f"Finger {finger} on string {string_idx+1} fret {fret}")
    
    if not instructions:
        instructions = ["Open strings or muted"]
    
    # Draw instructions below the chord diagram
    for i, line in enumerate(instructions):
        cv2.putText(frame, line, (panel_x, panel_y + panel_height + 80 + i*28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
