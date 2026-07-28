"""
cv2_cuda_demo.py
=================
Demo program OpenCV dengan CUDA acceleration.

Program ini melakukan:
1. Cek ketersediaan GPU CUDA di OpenCV
2. Load gambar / generate gambar dummy
3. Bandingkan performa operasi CPU vs GPU (resize, grayscale, blur)
4. Contoh pipeline lengkap: upload -> proses di GPU -> download hasil

Requirement:
    - OpenCV yang di-build dengan WITH_CUDA=ON
    - numpy

Jalankan:
    python3 cv2_cuda_demo.py
    python3 cv2_cuda_demo.py --image path/ke/gambar.jpg
"""

import argparse
import time

import cv2
import numpy as np


def check_cuda_availability() -> bool:
    """Cek apakah OpenCV berhasil detect GPU CUDA."""
    print("=" * 60)
    print("CEK CUDA SUPPORT")
    print("=" * 60)
    print(f"OpenCV version : {cv2.__version__}")

    device_count = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"CUDA device count : {device_count}")

    if device_count == 0:
        print("\n[!] Tidak ada GPU CUDA terdeteksi.")
        print("    Pastikan OpenCV di-build dengan WITH_CUDA=ON")
        print("    dan driver NVIDIA terinstall dengan benar.")
        return False

    # Tampilkan detail device GPU pertama
    cv2.cuda.printCudaDeviceInfo(0)
    cv2.cuda.setDevice(0)
    return True


def load_or_generate_image(image_path: str | None) -> np.ndarray:
    """Load gambar dari path, atau generate gambar dummy kalau tidak ada."""
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            print(f"[!] Gagal load gambar dari '{image_path}', pakai gambar dummy.")
        else:
            print(f"[+] Gambar dimuat dari: {image_path} (shape={img.shape})")
            return img

    # Generate gambar dummy 1080p (simulasi frame kamera)
    print("[+] Menggunakan gambar dummy 1920x1080 (random noise)")
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


def benchmark_resize(img: np.ndarray, iterations: int = 100) -> None:
    """Bandingkan performa resize CPU vs GPU."""
    print("\n" + "=" * 60)
    print(f"BENCHMARK: RESIZE ({iterations}x, target 640x640)")
    print("=" * 60)

    # --- CPU ---
    start = time.perf_counter()
    for _ in range(iterations):
        _ = cv2.resize(img, (640, 640))
    cpu_time = time.perf_counter() - start
    print(f"CPU  : {cpu_time:.4f}s  ({cpu_time / iterations * 1000:.2f} ms/frame)")

    # --- GPU ---
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)

    start = time.perf_counter()
    for _ in range(iterations):
        _ = cv2.cuda.resize(gpu_img, (640, 640))
    gpu_time = time.perf_counter() - start
    print(f"GPU  : {gpu_time:.4f}s  ({gpu_time / iterations * 1000:.2f} ms/frame)")

    print(f"Speedup: {cpu_time / gpu_time:.2f}x")


def benchmark_grayscale_blur(img: np.ndarray, iterations: int = 100) -> None:
    """Bandingkan performa grayscale + gaussian blur CPU vs GPU."""
    print("\n" + "=" * 60)
    print(f"BENCHMARK: GRAYSCALE + GAUSSIAN BLUR ({iterations}x)")
    print("=" * 60)

    # --- CPU ---
    start = time.perf_counter()
    for _ in range(iterations):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _ = cv2.GaussianBlur(gray, (15, 15), 0)
    cpu_time = time.perf_counter() - start
    print(f"CPU  : {cpu_time:.4f}s  ({cpu_time / iterations * 1000:.2f} ms/frame)")

    # --- GPU ---
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)
    gaussian_filter = cv2.cuda.createGaussianFilter(
        cv2.CV_8UC1, cv2.CV_8UC1, (15, 15), 0
    )

    start = time.perf_counter()
    for _ in range(iterations):
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        _ = gaussian_filter.apply(gpu_gray)
    gpu_time = time.perf_counter() - start
    print(f"GPU  : {gpu_time:.4f}s  ({gpu_time / iterations * 1000:.2f} ms/frame)")

    print(f"Speedup: {cpu_time / gpu_time:.2f}x")


def demo_full_pipeline(img: np.ndarray, output_path: str = "output_gpu.jpg") -> None:
    """Contoh pipeline lengkap: upload -> proses di GPU -> download -> simpan."""
    print("\n" + "=" * 60)
    print("DEMO PIPELINE: Upload -> Resize -> Grayscale -> Blur -> Download")
    print("=" * 60)

    # 1. Upload gambar CPU -> GPU
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)
    print("[1] Upload gambar ke GPU memory: OK")

    # 2. Resize di GPU
    gpu_resized = cv2.cuda.resize(gpu_img, (640, 640))
    print("[2] Resize (640x640) di GPU: OK")

    # 3. Convert ke grayscale di GPU
    gpu_gray = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2GRAY)
    print("[3] Convert ke grayscale di GPU: OK")

    # 4. Gaussian blur di GPU
    gaussian_filter = cv2.cuda.createGaussianFilter(
        cv2.CV_8UC1, cv2.CV_8UC1, (9, 9), 0
    )
    gpu_blurred = gaussian_filter.apply(gpu_gray)
    print("[4] Gaussian blur di GPU: OK")

    # 5. Download hasil balik ke CPU (numpy array)
    result = gpu_blurred.download()
    print("[5] Download hasil ke CPU memory: OK")

    # 6. Simpan hasil
    cv2.imwrite(output_path, result)
    print(f"[6] Hasil disimpan ke: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo OpenCV CUDA")
    parser.add_argument(
        "--image", type=str, default=None, help="Path ke gambar input (opsional)"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Jumlah iterasi benchmark"
    )
    args = parser.parse_args()

    if not check_cuda_availability():
        return

    img = load_or_generate_image(args.image)

    benchmark_resize(img, args.iterations)
    benchmark_grayscale_blur(img, args.iterations)
    demo_full_pipeline(img)

    print("\n" + "=" * 60)
    print("SELESAI. Semua operasi GPU berhasil dijalankan.")
    print("=" * 60)


if __name__ == "__main__":
    main()