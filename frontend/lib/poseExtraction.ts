declare global {
    interface Window {
      Pose: any;
    }
  }
  
  let poseModel: any = null;
  let latestPose: any = null;
  let poseBusy = false;
  
  const LANDMARK_INDICES: Record<string, number> = {
    nose: 0,
    left_shoulder: 11,
    right_shoulder: 12,
    left_elbow: 13,
    right_elbow: 14,
    left_wrist: 15,
    right_wrist: 16,
    left_hip: 23,
    right_hip: 24,
  };
  
  /**
   * Load MediaPipe Pose script from CDN if not loaded
   */
  function loadPoseScript() {
    const src =
      "https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js";
  
    return new Promise<void>((resolve, reject) => {
      if (document.querySelector(`script[src="${src}"]`)) {
        resolve();
        return;
      }
      const script = document.createElement("script");
      script.src = src;
      script.async = true;
      script.onload = () => resolve();
      script.onerror = () => reject();
      document.body.appendChild(script);
    });
  }
  
  /**
   * Initialize Pose model (only once)
   */
  export async function initPose(): Promise<void> {
    if (poseModel) return;
  
    await loadPoseScript();
  
    const Pose = window.Pose; // constructor from CDN
  
    poseModel = new Pose({
      locateFile: (file: string) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });
  
    poseModel.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
    });
  
    poseModel.onResults((results: any) => {
      if (!results.poseLandmarks) {
        latestPose = null;
        poseBusy = false;
        return;
      }
  
      const lm = results.poseLandmarks;
      const joints: Record<string, number[]> = {};
  
      for (const key in LANDMARK_INDICES) {
        const idx = LANDMARK_INDICES[key];
        const landmark = lm[idx];
        joints[key] = [landmark.x, landmark.y, landmark.z];
      }
  
      // Normalize exactly like Python
      const leftHip = joints["left_hip"];
      const rightHip = joints["right_hip"];
  
      const torso = [
        (leftHip[0] + rightHip[0]) / 2,
        (leftHip[1] + rightHip[1]) / 2,
        (leftHip[2] + rightHip[2]) / 2,
      ];
  
      const shoulderWidth =
        Math.sqrt(
          (joints["left_shoulder"][0] -
            joints["right_shoulder"][0]) ** 2 +
            (joints["left_shoulder"][1] -
              joints["right_shoulder"][1]) ** 2 +
            (joints["left_shoulder"][2] -
              joints["right_shoulder"][2]) ** 2
        ) + 1e-6;
  
      for (const k in joints) {
        joints[k] = [
          (joints[k][0] - torso[0]) / shoulderWidth,
          (joints[k][1] - torso[1]) / shoulderWidth,
          (joints[k][2] - torso[2]) / shoulderWidth,
        ];
      }
  
      latestPose = joints;
      poseBusy = false;
    });
  }
  
  /**
   * Send video frame to Pose (non-blocking)
   */
  export function sendPoseFrame(video: HTMLVideoElement) {
    if (!poseModel || poseBusy) return;
    poseBusy = true;
    poseModel.send({ image: video });
  }
  
  /**
   * Get last computed pose
   */
  export function getLatestPose() {
    return latestPose;
  }
  