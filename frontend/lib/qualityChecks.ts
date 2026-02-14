/**
 * Equivalent to:
 * is_frame_too_dark(frame, threshold=40)
 */
export function isFrameTooDark(
    imageData: ImageData,
    threshold: number = 40
  ): boolean {
    const data = imageData.data;
    let sum = 0;
    const totalPixels = data.length / 4;
  
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
  
      const gray =
        0.299 * r +
        0.587 * g +
        0.114 * b;
  
      sum += gray;
    }
  
    const mean = sum / totalPixels;
  
    return mean < threshold;
  }
  
  
  /**
   * Equivalent to:
   * is_frame_blurry(frame, threshold=30)
   */
  export function isFrameBlurry(
    imageData: ImageData,
    width: number,
    height: number,
    threshold: number = 30
  ): boolean {
    const gray = toGrayscale(imageData);
    const laplacian = applyLaplacian(gray, width, height);
    const variance = computeVariance(laplacian);
  
    return variance < threshold;
  }
  
  
  function toGrayscale(imageData: ImageData): number[] {
    const data = imageData.data;
    const gray: number[] = [];
  
    for (let i = 0; i < data.length; i += 4) {
      const value =
        0.299 * data[i] +
        0.587 * data[i + 1] +
        0.114 * data[i + 2];
  
      gray.push(value);
    }
  
    return gray;
  }
  
  
  function applyLaplacian(
    gray: number[],
    width: number,
    height: number
  ): number[] {
    const output = new Array(gray.length).fill(0);
  
    const kernel = [
      0, 1, 0,
      1, -4, 1,
      0, 1, 0
    ];
  
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let sum = 0;
        let k = 0;
  
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const idx =
              (y + ky) * width + (x + kx);
  
            sum += gray[idx] * kernel[k++];
          }
        }
  
        output[y * width + x] = sum;
      }
    }
  
    return output;
  }
  
  
  function computeVariance(arr: number[]): number {
    const mean =
      arr.reduce((a, b) => a + b, 0) / arr.length;
  
    return (
      arr.reduce(
        (sum, val) => sum + (val - mean) ** 2,
        0
      ) / arr.length
    );
  }
  