import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ------------------
# Calibration
# ------------------
# Note: Since values are now ratios, these constants 
# will require re-tuning (typically smaller values)
MAG_LOW  = 0.05
MAG_HIGH = 0.30

NOSE_VAR_LOW  = 0.01
NOSE_VAR_HIGH = 0.10

WEIGHTS = {
    "magnitude":  0.40,
    "asymmetry":  0.45,
    "nose_var":   0.15,
}

THRESHOLD = 0.50

# ------------------
# Webcam capture
# ------------------
cap = cv2.VideoCapture(0)

def capture(label):
    print(f"\n  [{label}]  — trigger your flash, then press ENTER to capture")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("  Camera error")
            break
        preview = frame.copy()
        cv2.putText(preview, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Capture", preview)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10):          # Enter
            return frame
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            quit()
    return None

print("\n=== Liveness capture ===")
print("  Position your face, trigger each flash, press ENTER.")
print("  Press Q at any time to quit.\n")

base  = capture("NORMAL — no flash")
left  = capture("LEFT FLASH")
right = capture("RIGHT FLASH")
top   = capture("TOP FLASH")

cap.release()
cv2.destroyWindow("Capture")

# ------------------
# InsightFace
# ------------------
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

def get_face(img):
    faces = app.get(img)
    if len(faces) == 0:
        return None
    return faces[0]

def align_to_base(img, ref_kps):
    face = get_face(img)
    if face is None:
        return None, None
    kps = face.kps.astype(np.float32)
    M = cv2.getAffineTransform(kps[[0, 1, 2]], ref_kps[[0, 1, 2]])
    aligned = cv2.warpAffine(img, M, (base.shape[1], base.shape[0]))
    return aligned, kps

base_face = get_face(base)
if base_face is None:
    print("Face not found")
    quit()

base_kps = base_face.kps.astype(np.float32)
left,  _ = align_to_base(left,  base_kps)
right, _ = align_to_base(right, base_kps)
top,   _ = align_to_base(top,   base_kps)

if left is None or right is None or top is None:
    print("Face not found")
    quit()

kps = base_kps.astype(int)
left_eye  = kps[0]
right_eye = kps[1]
nose      = kps[2]

# ------------------
# ROI helper
# ------------------
def roi_mean(data, x, y, size=20):
    x1 = max(0, x - size)
    y1 = max(0, y - size)
    x2 = min(data.shape[1], x + size)
    y2 = min(data.shape[0], y + size)
    return float(np.mean(data[y1:y2, x1:x2]))

# ------------------
# Facial regions
# ------------------
left_cheek  = (left_eye[0],  nose[1])
right_cheek = (right_eye[0], nose[1])
nose_pt     = (nose[0], nose[1])

# ------------------
# Per-flash analysis (UPDATED)
# ------------------
def analyze(img):
    g1 = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(float)
    g2 = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY).astype(float)
    
    # Normalized illumination: ratio - 1 (percentage change)
    norm_diff = (g2 / (g1 + 1.0)) - 1.0
    
    return (
        roi_mean(norm_diff, *left_cheek),
        roi_mean(norm_diff, *nose_pt),
        roi_mean(norm_diff, *right_cheek),
        (np.clip(norm_diff * 255, 0, 255)).astype(np.uint8)
    )

l_l, l_n, l_r, d1 = analyze(left)
r_l, r_n, r_r, d2 = analyze(right)
t_l, t_n, t_r, d3 = analyze(top)

# ------------------
# Raw score report
# ------------------
def print_scores(name, l, n, r):
    print(f"\n{name}")
    print("-" * 25)
    print(f"Left Cheek : {l:.4f}")
    print(f"Nose       : {n:.4f}")
    print(f"Right Cheek: {r:.4f}")

print_scores("LEFT FLASH",  l_l, l_n, l_r)
print_scores("RIGHT FLASH", r_l, r_n, r_r)
print_scores("TOP FLASH",   t_l, t_n, t_r)

# ------------------
# CONFIDENCE SCORE
# ------------------
all_scores      = [l_l, l_n, l_r, r_l, r_n, r_r, t_l, t_n, t_r]
magnitude       = float(np.mean(all_scores))
magnitude_score = float(np.clip((magnitude - MAG_LOW) / (MAG_HIGH - MAG_LOW), 0.0, 1.0))

eps = 1e-6
def asymmetry(brighter, darker):
    return (brighter - darker) / (brighter + darker + eps)

left_asym  = asymmetry(l_r, l_l)
right_asym = asymmetry(r_l, r_r)

asym_score = float(np.clip((left_asym + right_asym) / 2, 0.0, 1.0))

nose_variance = float(np.std([l_n, r_n, t_n]))
nose_score    = float(np.clip((nose_variance - NOSE_VAR_LOW) / (NOSE_VAR_HIGH - NOSE_VAR_LOW), 0.0, 1.0))

confidence = (
    WEIGHTS["magnitude"] * magnitude_score +
    WEIGHTS["asymmetry"] * asym_score +
    WEIGHTS["nose_var"]  * nose_score
)

# ------------------
# Decision + Report
# ------------------
label = "REAL" if confidence >= THRESHOLD else "SPOOF"

print("\n" + "=" * 42)
print("        LIVENESS CONFIDENCE REPORT")
print("=" * 42)
print(f"  Confidence       : {confidence * 100:5.1f}%")
print(f"  Decision         : {label}")
print("=" * 42)

# ------------------
# Visualize
# ------------------
vis = base.copy()
for p in [left_cheek, nose_pt, right_cheek]:
    cv2.circle(vis, p, 20, (0, 255, 0), 2)

cv2.imshow("Regions", vis)
cv2.imshow("Left Diff",  d1)
cv2.imshow("Right Diff", d2)
cv2.imshow("Top Diff",   d3)
cv2.waitKey(0)
cv2.destroyAllWindows()