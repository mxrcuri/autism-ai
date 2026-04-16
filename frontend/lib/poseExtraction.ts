declare global {
  interface Window {
    Pose: any;
  }
}

let poseModel: any = null;
let latestSkeleton: any = null;
let poseBusy = false;

// MediaPipe basic indices
const LM = {
  nose: 0,
  l_shoulder: 11,
  r_shoulder: 12,
  l_elbow: 13,
  r_elbow: 14,
  l_wrist: 15,
  r_wrist: 16,
  l_pinky: 17,
  r_pinky: 18,
  l_index: 19,
  r_index: 20,
  l_thumb: 21,
  r_thumb: 22,
  l_hip: 23,
  r_hip: 24,
};

function loadPoseScript() {
  const src = "https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js";
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

export async function initPose(): Promise<void> {
  if (poseModel) return;

  await loadPoseScript();

  const Pose = window.Pose;
  poseModel = new Pose({
    locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
  });

  poseModel.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: false,
  });

  poseModel.onResults((results: any) => {
    if (!results.poseLandmarks) {
      latestSkeleton = null;
      poseBusy = false;
      return;
    }

    const lm = results.poseLandmarks;
    
    // Helper to extract a 3D point
    const pt = (idx: number) => [lm[idx].x, lm[idx].y, lm[idx].z];
    
    // Helper to average two 3D points
    const avg = (p1: number[], p2: number[]) => [
      (p1[0] + p2[0]) / 2,
      (p1[1] + p2[1]) / 2,
      (p1[2] + p2[2]) / 2,
    ];

    // Construct exactly as the DREAM dataset schema expects
    const skeleton: Record<string, number[]> = {
      head: pt(LM.nose),
      sholder_center: avg(pt(LM.l_shoulder), pt(LM.r_shoulder)),
      sholder_left: pt(LM.l_shoulder),
      sholder_right: pt(LM.r_shoulder),
      elbow_left: pt(LM.l_elbow),
      elbow_right: pt(LM.r_elbow),
      wrist_left: pt(LM.l_wrist),
      wrist_right: pt(LM.r_wrist),
      // Hand = center point between the index and pinky knuckle/end
      hand_left: avg(pt(LM.l_index), pt(LM.l_pinky)),
      hand_right: avg(pt(LM.r_index), pt(LM.r_pinky)),
    };

    // Normalize relative to torso center and shoulder width (just like Python pipeline)
    const torso = avg(pt(LM.l_hip), pt(LM.r_hip));
    const shoulderWidth = Math.max(
      1e-6,
      Math.sqrt(
        (skeleton.sholder_left[0] - skeleton.sholder_right[0]) ** 2 +
        (skeleton.sholder_left[1] - skeleton.sholder_right[1]) ** 2 +
        (skeleton.sholder_left[2] - skeleton.sholder_right[2]) ** 2
      )
    );

    for (const key in skeleton) {
      skeleton[key] = [
        (skeleton[key][0] - torso[0]) / shoulderWidth,
        (skeleton[key][1] - torso[1]) / shoulderWidth,
        (skeleton[key][2] - torso[2]) / shoulderWidth,
      ];
    }

    latestSkeleton = skeleton;
    poseBusy = false;
  });
}

export function sendPoseFrame(video: HTMLVideoElement) {
  if (!poseModel || poseBusy) return;
  poseBusy = true;
  poseModel.send({ image: video });
}

export function getLatestPose() {
  return latestSkeleton;
}