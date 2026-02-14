type HeadPose = {
    yaw: number;
    pitch: number;
    roll: number;
  } | null;
  
  type Gaze = {
    gx: number;
    gy: number;
    gz: number;
  } | null;
  
  /**
   * Matches Python:
   *
   * gx = -yaw
   * gy = -pitch
   * gz = 1.0
   */
  export function estimateGaze(head: HeadPose): Gaze {
    if (!head) return null;
  
    const gx = -head.yaw;
    const gy = -head.pitch;
    const gz = 1.0;
  
    return {
      gx: Number(gx),
      gy: Number(gy),
      gz: Number(gz),
    };
  }
  