type Landmark = {
    x: number;
    y: number;
    z: number;
  };
  
  type HeadPose = {
    yaw: number;
    pitch: number;
    roll: number;
  } | null;
  
  /**
   * Extract head pose (yaw, pitch, roll)
   * from MediaPipe face landmarks.
   *
   * Matches Python logic exactly.
   */
  export function extractHeadPose(
    faceLandmarks: Landmark[] | null,
    valid: boolean
  ): HeadPose {
    if (!valid || !faceLandmarks) {
      return null;
    }
  
    // Landmark indices (MediaPipe FaceMesh standard)
    const LEFT_EYE_INDEX = 33;
    const RIGHT_EYE_INDEX = 263;
    const NOSE_INDEX = 1;
  
    const leftEye = faceLandmarks[LEFT_EYE_INDEX];
    const rightEye = faceLandmarks[RIGHT_EYE_INDEX];
    const nose = faceLandmarks[NOSE_INDEX];
  
    if (!leftEye || !rightEye || !nose) {
      return null;
    }
  
    // === EXACT SAME MATH AS PYTHON ===
  
    const yaw = rightEye.x - leftEye.x;
  
    const pitch =
      nose.y - (leftEye.y + rightEye.y) / 2;
  
    const roll = rightEye.y - leftEye.y;
  
    return {
      yaw: Number(yaw),
      pitch: Number(pitch),
      roll: Number(roll),
    };
  }
  