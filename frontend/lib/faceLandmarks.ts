import {
    FaceLandmarker,
    FilesetResolver,
  } from "@mediapipe/tasks-vision";
  
  let faceLandmarker: FaceLandmarker | null = null;
  let latestFaceLandmarks: any = null;
  
  export async function initFaceLandmarker(): Promise<void> {
    if (faceLandmarker) return;
  
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
    );
  
    faceLandmarker = await FaceLandmarker.createFromOptions(
      vision,
      {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        },
        runningMode: "VIDEO",
        numFaces: 1,
      }
    );
  }
  
  export function sendFaceLandmarkFrame(video: HTMLVideoElement) {
    if (!faceLandmarker) return;
  
    const result = faceLandmarker.detectForVideo(
      video,
      performance.now()
    );
  
    if (result.faceLandmarks.length > 0) {
      latestFaceLandmarks = result.faceLandmarks[0];
    } else {
      latestFaceLandmarks = null;
    }
  }
  
  export function getLatestFaceLandmarks() {
    return latestFaceLandmarks;
  }
  