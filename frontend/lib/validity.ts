import { isFrameTooDark, isFrameBlurry } from "./qualityChecks";
import {
  sendFaceFrame,
  getLatestFaceCount,
} from "./faceDetection";

export type ValidityStats = {
  total_frames: number;
  valid_frames: number;
  no_face_frames: number;
  multi_face_frames: number;
  dark_frames: number;
  blurry_frames: number;
};

export function createInitialStats(): ValidityStats {
  return {
    total_frames: 0,
    valid_frames: 0,
    no_face_frames: 0,
    multi_face_frames: 0,
    dark_frames: 0,
    blurry_frames: 0,
  };
}

/**
 * Non-blocking frame validity processing
 */
export function processFrameValidity(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  stats: ValidityStats,
  validMask: boolean[]
) {
  if (
    video.videoWidth === 0 ||
    video.videoHeight === 0
  ) {
    return;
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  ctx.drawImage(video, 0, 0);

  const imageData = ctx.getImageData(
    0,
    0,
    canvas.width,
    canvas.height
  );

  stats.total_frames += 1;

  let valid = true;

  // 🔹 Face detection (event-driven)
  sendFaceFrame(video);
  const faceCount = getLatestFaceCount();

  if (faceCount === 0) {
    stats.no_face_frames += 1;
    valid = false;
  } else if (faceCount > 1) {
    stats.multi_face_frames += 1;
    valid = false;
  }

  // 🔹 Lighting
  if (isFrameTooDark(imageData)) {
    stats.dark_frames += 1;
    valid = false;
  }

  // 🔹 Blur (secondary rule, same as Python)
  if (
    isFrameBlurry(
      imageData,
      canvas.width,
      canvas.height
    ) &&
    faceCount !== 1
  ) {
    stats.blurry_frames += 1;
    valid = false;
  }

  if (valid) {
    stats.valid_frames += 1;
  }

  validMask.push(valid);
}

/**
 * Equivalent of evaluate_video_quality() from Python
 */
export function evaluateVideoQuality(
  validMask: boolean[],
  stats: ValidityStats,
  minValidFrames: number = 50,
  minValidRatio: number = 0.7,
  maxInvalidGapSec: number = 3.0,
  fps: number = 15
): { usable: boolean; reason: string } {
  if (stats.total_frames === 0) {
    return { usable: false, reason: "NO_FRAMES_DECODED" };
  }

  if (stats.valid_frames < minValidFrames) {
    return {
      usable: false,
      reason: "TOO_FEW_VALID_FRAMES",
    };
  }

  const validRatio =
    stats.valid_frames / stats.total_frames;

  if (validRatio < minValidRatio) {
    return {
      usable: false,
      reason: "LOW_VALID_RATIO",
    };
  }

  let maxGap = 0;
  let currentGap = 0;

  for (const v of validMask) {
    if (!v) {
      currentGap += 1;
      maxGap = Math.max(maxGap, currentGap);
    } else {
      currentGap = 0;
    }
  }

  if (maxGap / fps > maxInvalidGapSec) {
    return {
      usable: false,
      reason: "LONG_INVALID_GAP",
    };
  }

  return { usable: true, reason: "OK" };
}
