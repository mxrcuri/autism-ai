import {
  FaceDetector,
  FilesetResolver,
} from "@mediapipe/tasks-vision";

let faceDetector: any = null;
let latestFaceCount = 0;
let faceInitialized = false;

export async function initFaceDetection(): Promise<void> {
  if (faceInitialized) return;

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
  );

  faceDetector = await FaceDetector.createFromOptions(
    vision,
    {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
      },
      runningMode: "VIDEO",
    }
  );

  faceInitialized = true;
  latestFaceCount = 0;
}

export function sendFaceFrame(video: HTMLVideoElement) {
  if (!faceDetector) return;

  console.log("Face detector object:", faceDetector);
  console.log("Available methods:", Object.keys(faceDetector));

  const result = faceDetector.detectForVideo(
    video,
    performance.now()
  );

  latestFaceCount = result.detections.length;
}


export function getLatestFaceCount(): number {
  return latestFaceCount;
}
