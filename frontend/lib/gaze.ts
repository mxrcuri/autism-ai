type Landmark = { x: number; y: number; z: number };

type Gaze = {
  rx: number;
  ry: number;
  rz: number;
} | null;

/**
 * Calculates a true gaze vector using iris landmarks relative to eye corners,
 * matching precise tracking metrics similar to the DREAM dataset's eye_gaze.
 */
export function estimateGaze(faceLandmarks: Landmark[] | null): Gaze {
  if (!faceLandmarks) return null;

  // Key MediaPipe FaceMesh indices
  // Left eye (viewer's right)
  const LEFT_EYE_INNER = 133;
  const LEFT_EYE_OUTER = 33;
  const LEFT_EYE_TOP = 159;
  const LEFT_EYE_BOTTOM = 145;
  const LEFT_IRIS = 468; // Center of left iris

  // Right eye (viewer's left)
  const RIGHT_EYE_INNER = 362;
  const RIGHT_EYE_OUTER = 263;
  const RIGHT_EYE_TOP = 386;
  const RIGHT_EYE_BOTTOM = 374;
  const RIGHT_IRIS = 473; // Center of right iris

  // Ensure iris landmarks exist (they only generate if model is configured for it,
  // but face_landmarker.task generally includes them now)
  if (!faceLandmarks[LEFT_IRIS] || !faceLandmarks[RIGHT_IRIS]) {
     return null;
  }

  // --- Calculate relative Left Eye Gaze ---
  const lIris = faceLandmarks[LEFT_IRIS];
  const lInner = faceLandmarks[LEFT_EYE_INNER];
  const lOuter = faceLandmarks[LEFT_EYE_OUTER];
  const lTop = faceLandmarks[LEFT_EYE_TOP];
  const lBottom = faceLandmarks[LEFT_EYE_BOTTOM];

  // Horizontal: 0 = looking full left, 1 = looking full right
  const lHorizSpan = Math.abs(lOuter.x - lInner.x);
  const lGazeX = lHorizSpan > 0 ? (lIris.x - lInner.x) / lHorizSpan : 0.5;

  // Vertical: 0 = looking up, 1 = looking down
  const lVertSpan = Math.abs(lBottom.y - lTop.y);
  const lGazeY = lVertSpan > 0 ? (lIris.y - lTop.y) / lVertSpan : 0.5;

  // --- Calculate relative Right Eye Gaze ---
  const rIris = faceLandmarks[RIGHT_IRIS];
  const rInner = faceLandmarks[RIGHT_EYE_INNER];
  const rOuter = faceLandmarks[RIGHT_EYE_OUTER];
  const rTop = faceLandmarks[RIGHT_EYE_TOP];
  const rBottom = faceLandmarks[RIGHT_EYE_BOTTOM];

  const rHorizSpan = Math.abs(rOuter.x - rInner.x);
  const rGazeX = rHorizSpan > 0 ? (rIris.x - rOuter.x) / rHorizSpan : 0.5;

  const rVertSpan = Math.abs(rBottom.y - rTop.y);
  const rGazeY = rVertSpan > 0 ? (rIris.y - rTop.y) / rVertSpan : 0.5;

  // Average the two eyes
  const avgGazeX = (lGazeX + rGazeX) / 2;
  const avgGazeY = (lGazeY + rGazeY) / 2;

  // Convert the [0, 1] relative box mapped positions into pseudo-radians/vectors 
  // (-0.5 to 0.5 range) where 0 is looking straight ahead.
  const rx = (avgGazeX - 0.5) * 2; // Horizontal rotation
  const ry = (avgGazeY - 0.5) * 2; // Vertical rotation
  const rz = 1.0;                  // Depth vector assuming looking forward

  return {
    rx: Number(rx),
    ry: Number(ry),
    rz: Number(rz),
  };
}