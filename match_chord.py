import pandas as pd

df = pd.read_csv("./GuitarChords.csv")
df.drop(columns=["Capo", "Key", "Note Order", "Note", "Roman Numeral"], axis=1, inplace=True)
df.drop_duplicates(subset=["Chord", "Finger Label", "Guitar String", "Fret"], inplace=True)
df.dropna(subset=["Finger Label"], inplace=True)
df["Fret"] = df["Fret"].replace("x", 0)
df["Fret"] = df["Fret"].astype(int)
# print(df)

CHORDS = {}

#iterate through dataframe and populate CHORDS dictionary
for chord_name, group in df.groupby("Chord"):
    positions = list(zip(group["Finger Label"], group["Guitar String"], group["Fret"]))
    CHORDS[chord_name] = positions

# print(CHORDS) #(finger num, string num, fret num)

def match_chord(finger_positions):
    """
    Match fingers to chord
    :param finger_positions: (finger num, string num, fret num)
    :return: Matched chord name or None
    """
    for chord_name, positions in CHORDS.items():
        if all(pos in finger_positions for pos in positions):
            return chord_name
    return None

