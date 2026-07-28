import cv2
import numpy as np
import json
import os
import glob
from lwcc import LWCC


def count_crowd_image(image_path, json_path, output_path="output/hasil_crowd.jpg",
                       model_name="CSRNet", model_weights="SHA", save_overlay=True):
    # load gambar asli
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    # load ground truth dari JSON
    with open(json_path) as f:
        gt_data = json.load(f)
    ground_truth = len(gt_data["annotations"])

    # hitung crowd pakai CSRNet + density map
    prediction, density_map = LWCC.get_count(
        image_path, model_name=model_name, model_weights=model_weights,
        resize_img=False, return_density=True
    )

    # hitung error
    error = abs(prediction - ground_truth)
    error_pct = (error / ground_truth * 100) if ground_truth > 0 else 0
    bias = prediction - ground_truth  # positif = overcount, negatif = undercount

    if save_overlay:
        h, w = img.shape[:2]
        heatmap = cv2.resize(density_map, (w, h))
        heatmap = (heatmap / (heatmap.max() + 1e-6) * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

        cv2.putText(overlay, f"CSRNet: {prediction:.1f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(overlay, f"Ground Truth: {ground_truth}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(overlay, f"Error: {error:.1f} ({error_pct:.1f}%)", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_path, overlay)

    print(f"[{os.path.basename(image_path)}] GT: {ground_truth:4d} | Pred: {prediction:7.1f} | "
          f"Error: {error:6.1f} ({error_pct:5.1f}%) | Bias: {bias:+.1f}")

    return {
        "image": os.path.basename(image_path),
        "prediction": prediction,
        "ground_truth": ground_truth,
        "error": error,
        "error_pct": error_pct,
        "bias": bias
    }


def evaluate_dataset(images_dir, labels_dir, model_name="CSRNet", model_weights="SHA",
                      output_dir="output/eval", save_overlay=False):
    """
    Opsi 1: Cek apakah error CSRNet konsisten (bias sistematis) atau acak,
    dengan menjalankan evaluasi di semua gambar dalam dataset.
    """
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    results = []

    print(f"Mengevaluasi {len(image_files)} gambar...\n")
    print("-" * 80)

    for img_path in image_files:
        filename = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(labels_dir, f"{filename}.json")

        if not os.path.exists(json_path):
            print(f"[SKIP] {filename}: JSON tidak ditemukan")
            continue

        out_path = os.path.join(output_dir, f"{filename}_hasil.jpg")
        try:
            r = count_crowd_image(
                img_path, json_path, output_path=out_path,
                model_name=model_name, model_weights=model_weights,
                save_overlay=save_overlay
            )
            results.append(r)
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

    print("-" * 80)

    if not results:
        print("Tidak ada hasil untuk dianalisis.")
        return results

    # === Ringkasan statistik ===
    avg_error_pct = sum(r["error_pct"] for r in results) / len(results)
    avg_bias = sum(r["bias"] for r in results) / len(results)
    avg_abs_error = sum(r["error"] for r in results) / len(results)  # MAE

    total_gt = sum(r["ground_truth"] for r in results)
    total_pred = sum(r["prediction"] for r in results)
    calibration_factor = total_gt / total_pred if total_pred > 0 else 1.0

    print(f"\n{'='*40}")
    print(f"RINGKASAN EVALUASI ({len(results)} gambar)")
    print(f"{'='*40}")
    print(f"MAE (rata-rata error absolut) : {avg_abs_error:.2f} orang")
    print(f"Rata-rata error persentase    : {avg_error_pct:.2f}%")
    print(f"Rata-rata bias                : {avg_bias:+.2f} orang "
          f"({'OVERCOUNT' if avg_bias > 0 else 'UNDERCOUNT'})")
    print(f"Faktor kalibrasi yang disarankan: {calibration_factor:.4f}")
    print(f"{'='*40}")

    # cek konsistensi bias (apakah semua gambar overcount/undercount searah)
    overcounts = sum(1 for r in results if r["bias"] > 0)
    undercounts = sum(1 for r in results if r["bias"] < 0)
    print(f"\nJumlah gambar overcount  : {overcounts}/{len(results)}")
    print(f"Jumlah gambar undercount : {undercounts}/{len(results)}")

    if overcounts / len(results) > 0.8 or undercounts / len(results) > 0.8:
        print(f"\n✔ Bias TERLIHAT KONSISTEN — kalibrasi dengan faktor "
              f"{calibration_factor:.4f} kemungkinan besar akan membantu.")
    else:
        print(f"\n✘ Bias TIDAK KONSISTEN — kalibrasi sederhana kurang efektif, "
              f"pertimbangkan fine-tuning atau ganti model/weights.")

    return results


if __name__ == "__main__":
    IMAGES_DIR = "dataset/crowd-counting/images/4000-5000"
    LABELS_DIR = "dataset/crowd-counting/labels/4000-5000"

    results = evaluate_dataset(
        IMAGES_DIR, LABELS_DIR,
        model_name="CSRNet", model_weights="SHA",
        save_overlay=False  # ganti True kalau mau simpan gambar overlay tiap sample
    )