import os
import gc
import json
import time
import shutil
import random
import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import sys
# Ensure imports work from main_detection directory or repository root
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
for p in [curr_dir, parent_dir, os.path.join(parent_dir, "Mammo-CLIP-repo", "src", "codebase")]:
    if p not in sys.path and os.path.exists(p):
        sys.path.append(p)

from Detectors.retinanet.detector_model import RetinaNet_efficientnet
from preprocessing_RSNA_MAMO import MammogramPreprocessor, read_and_window_dicom
from MammoDetectorXAI import MammoDetectorXAI
from visualize_detector_xai import visualize_detector_xai
from senAnals import run_threshold_sensitivity_analysis


# =========================================================================
# GLOBAL HYPERPARAMETERS & PIPELINE CONFIGURATION
# =========================================================================

# --- Random Seed & Reproducibility ---
RANDOM_SEED = 86

# --- Experiment & Model Architecture ---
FREEZE_BACKBONE = "y"  # Frozen base Mammo-CLIP model
MODEL_ARCH = "breast_clip_b5"
CLINICAL_BETA = 1.5

# --- Hardware & DataLoader Hyperparameters ---
BATCH_SIZE_INFER = 16
NUM_WORKERS = 8
PIN_MEMORY = True
PROCESS_POOL_WORKERS = 24

# --- Image Resolutions & Preprocessing ---
TARGET_IMG_SIZE = (1536, 1024)
SITE_STATS_SAMPLE_SIZE = 50  # Samples per site from (calibration + unused benign pool)

# --- XAI Hyperparameters ---
XAI_DRISE_MASKS = 25
XAI_MC_PASSES = 8
XAI_MC_NOISE_STD = 0.02
XAI_REPORT_DPI = 120
XAI_EVAL_MAX_SAMPLES = 100  # 50 malignant and 50 benign reports


def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def assign_age_group(age) -> str:
    if pd.isna(age):
        return "Unknown"
    age = float(age)
    if age < 20:
        return "before 20"
    elif age < 40:
        return "20-40"
    elif age < 55:
        return "40-55"
    else:
        return "55+"


def compute_metrics(y_true, y_probs, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos > 0 and n_neg > 0:
        auc = float(roc_auc_score(y_true, y_probs))
        bacc = float(balanced_accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sp = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        sn = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        acc = float((tp + tn) / len(y_true))
    elif n_pos > 0:
        auc = 0.5
        bacc = float(np.mean(y_pred == 1))
        tp = int(np.sum(y_pred == 1))
        fn = int(np.sum(y_pred == 0))
        tn, fp = 0, 0
        sp = 0.0
        sn = float(tp / n_pos)
        prec = 1.0 if tp > 0 else 0.0
        acc = sn
    else:
        auc = 0.5
        bacc = float(np.mean(y_pred == 0))
        tn = int(np.sum(y_pred == 0))
        fp = int(np.sum(y_pred == 1))
        tp, fn = 0, 0
        sp = float(tn / n_neg)
        sn = 0.0
        prec = 0.0
        acc = sp

    beta = CLINICAL_BETA
    f1 = 2 * (prec * sn) / (prec + sn + 1e-8)
    f_beta = (1 + beta**2) * (prec * sn) / ((beta**2 * prec) + sn + 1e-8)

    return {
        "balanced_ACC": round(bacc, 4),
        "AUC": round(auc, 4),
        "SP": round(sp, 4),
        "SN": round(sn, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "f_beta": round(f_beta, 4),
        "decision_threshold": round(threshold, 4),
        "total_samples": int(len(y_true)),
        "true_label_cancer_cases": n_pos,
        "true_label_benign_cases": n_neg,
        "cancer_cases": n_pos,
        "non_cancer_cases": n_neg,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    }


def compute_site_stats_from_pool(df_pool: pd.DataFrame, data_dir: str, sample_size: int = SITE_STATS_SAMPLE_SIZE) -> dict:
    """
    Computes empirical zero-mean unit-variance and contrast landmarks
    strictly from the provided pool (Calibration Set + Unused Benign Images).
    """
    print(f"[*] Calculating site_stats empirically on calibration + unused benign pool (sample_size={sample_size} per site)...")
    img_dir = os.path.join(data_dir, "train_images")
    pre = MammogramPreprocessor()
    site_stats_res = {}

    def analyze_single_exam(row):
        dcm_path = os.path.join(img_dir, str(row["patient_id"]), f"{row['image_id']}.dcm")
        if not os.path.exists(dcm_path):
            return None
        try:
            norm = read_and_window_dicom(dcm_path)
            seg, mask, _, _ = pre.segment_breast_tissue(
                norm,
                laterality=str(row.get("laterality", "R")),
                view=str(row.get("view", "MLO"))
            )
            valid_pixels = seg[mask > 0]
            if len(valid_pixels) < 100:
                return None

            q1 = float(np.quantile(valid_pixels, 0.05))
            q2 = float(np.quantile(valid_pixels, 0.50))
            q3 = float(np.quantile(valid_pixels, 0.95))

            # Contrast transform
            t = np.zeros_like(valid_pixels)
            idx1 = (valid_pixels <= q1)
            idx2 = (valid_pixels > q1) & (valid_pixels <= q2)
            idx3 = (valid_pixels > q2) & (valid_pixels <= q3)
            idx4 = (valid_pixels > q3)

            t[idx1] = 0.15 * ((valid_pixels[idx1] - valid_pixels.min()) / (q1 - valid_pixels.min() + 1e-6))
            t[idx2] = 0.15 + 0.35 * ((valid_pixels[idx2] - q1) / (q2 - q1 + 1e-6))
            t[idx3] = 0.50 + 0.35 * ((valid_pixels[idx3] - q2) / (q3 - q2 + 1e-6))
            t[idx4] = 0.85 + 0.15 * ((valid_pixels[idx4] - q3) / (valid_pixels.max() - q3 + 1e-6))

            return float(t.mean()), float(t.std())
        except Exception:
            return None

    for site in sorted(df_pool["site_id"].unique()):
        site_df = df_pool[df_pool["site_id"] == site]
        site_sample = site_df.sample(
            n=min(sample_size, len(site_df)),
            random_state=RANDOM_SEED
        )
        with ThreadPoolExecutor(max_workers=8) as ex:
            eval_results = list(ex.map(analyze_single_exam, [r for _, r in site_sample.iterrows()]))

        valid_results = [r for r in eval_results if r is not None]
        means = [m for m, s in valid_results]
        stds = [s for m, s in valid_results]

        emp_mean = float(np.mean(means)) if len(means) > 0 else 0.4836
        emp_std = float(np.mean(stds)) if len(stds) > 0 else 0.2012
        site_stats_res[int(site)] = {
            "p_low": 0.05,
            "p_mid": 0.50,
            "p_high": 0.95,
            "mean": round(emp_mean, 4),
            "std": round(emp_std, 4),
            "samples_analyzed": len(means),
            "source_pool": "calibration_set_and_unused_benign"
        }
        print(f"    -> Site {site}: mean={emp_mean:.4f}, std={emp_std:.4f} (from {len(means)} exams)")

    return site_stats_res


def _precache_worker(task):
    pid, imid, lat, v, s, data_dir, dest_path, target_size, site_stats = task
    if os.path.exists(dest_path):
        return True
    dcm_path = os.path.join(data_dir, "train_images", str(pid), f"{imid}.dcm")
    if not os.path.exists(dcm_path):
        return False
    try:
        prep = MammogramPreprocessor(target_size=target_size, site_stats=site_stats)
        pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=False, force=True)
        img_raw = read_and_window_dicom(dcm)
        img_clean = prep.preprocess_image(img_raw, laterality=lat, view=v, site_id=s)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        np.save(dest_path, img_clean.astype(np.float32))
        return True
    except Exception:
        return False


class MammoDetectionDiskDataset(Dataset):
    """
    High-throughput dataset that loads preprocessed mammograms and returns
    3-channel normalized tensors for Mammo-CLIP RetinaNet.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        search_cache_dirs: list,
        data_dir: str,
        preprocessor: MammogramPreprocessor
    ):
        self.df = df.reset_index(drop=True)
        self.search_cache_dirs = search_cache_dirs
        self.data_dir = data_dir
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row["patient_id"])
        imid = str(row["image_id"])
        key = f"{pid}_{imid}"
        target = float(row.get("cancer", 0))

        # Search existing cache folders
        img = None
        for cdir in self.search_cache_dirs:
            candidate = os.path.join(cdir, f"{key}.npy")
            if os.path.exists(candidate):
                try:
                    img = np.load(candidate)
                    break
                except Exception:
                    pass

        # Dynamic fallback if not yet in cache
        if img is None:
            dcm_path = os.path.join(self.data_dir, "train_images", pid, f"{imid}.dcm")
            if os.path.exists(dcm_path):
                try:
                    dcm = pydicom.dcmread(dcm_path, stop_before_pixels=False, force=True)
                    img_raw = read_and_window_dicom(dcm)
                    img = self.preprocessor.preprocess_image(
                        img_raw,
                        laterality=str(row.get("laterality", "R")),
                        view=str(row.get("view", "MLO")),
                        site_id=int(row.get("site_id", 1))
                    )
                except Exception:
                    img = np.zeros(self.preprocessor.target_size, dtype=np.float32)
            else:
                img = np.zeros(self.preprocessor.target_size, dtype=np.float32)

        # Standardize into 3-channel tensor for Mammo-CLIP EfficientNet-B5 stem [3, H, W]
        img_tensor = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).repeat(3, 1, 1).float()
        return img_tensor, torch.tensor(target, dtype=torch.float32), pid, imid


def main():
    seed_everything(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Executing Mammo-CLIP Detection Pipeline on device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = "/mnt/fourT/projects4TB/MAMO"
    data_dir = os.path.join(base_dir, "rsna-breast-cancer-detection")
    cache_root_dir = os.path.join(base_dir, "preprocessed_cache")
    clip_chk_pt_path = os.path.join(base_dir, "pretrained_checkpoints", "b5-model-best-epoch-7.tar")
    concept_vec_path = os.path.join(base_dir, "pretrained_checkpoints", "concept_mass_512.pt")

    # ---------------------------------------------------------
    # Setup Dedicated Output Directory in "main_detection/"
    # ---------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(script_dir, f"run_detection_{timestamp}")
    run_results_dir = os.path.join(run_dir, "results")
    run_checkpoints_dir = os.path.join(run_dir, "checkpoints")
    run_splits_dir = os.path.join(run_dir, "splits")
    run_xai_dir = os.path.join(run_dir, "rsna_xai_outputs")

    # Requirement: In rsna_xai_outputs put figures in two folders: "benign" and "malignant" based on true label
    xai_malignant_dir = os.path.join(run_xai_dir, "malignant")
    xai_benign_dir = os.path.join(run_xai_dir, "benign")

    for d in [run_dir, run_results_dir, run_checkpoints_dir, run_splits_dir, run_xai_dir, xai_malignant_dir, xai_benign_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 85)
    print(f"[*] Dedicated Run Directory:  {run_dir}")
    print(f"[*] Checkpoints:              {run_checkpoints_dir}")
    print(f"[*] Results:                  {run_results_dir}")
    print(f"[*] Dataset Splits:           {run_splits_dir}")
    print(f"[*] XAI Outputs (Malignant):  {xai_malignant_dir}")
    print(f"[*] XAI Outputs (Benign):     {xai_benign_dir}")
    print(f"[*] Pretrained Weights:       {clip_chk_pt_path}")
    print("=" * 85)

    # ---------------------------------------------------------
    # Step 1: Find All Malignant Images & Construct Cohort Splits
    # ---------------------------------------------------------
    print("\n--- [Step 1: Dataset Partitioning & Cohort Splitting] ---")
    df_raw = pd.read_csv(os.path.join(data_dir, "train.csv"))
    print(f"[*] Total dataset rows in train.csv: {len(df_raw)} images")

    # Demographic attributes
    df_raw["age_group"] = df_raw["age"].apply(assign_age_group)
    df_raw["density_clean"] = df_raw["density"].fillna("Unknown")

    # 1. Find all malignant images
    df_all_mal = df_raw[df_raw["cancer"] == 1].reset_index(drop=True)
    n_all_mal = len(df_all_mal)
    print(f"[✓] Found all malignant images: {n_all_mal} images (100.0% of cancers)")

    # 2. 20% of malignant to calibration set and 80% to inference testing set
    df_mal_calib, df_mal_test = train_test_split(df_all_mal, test_size=0.80, random_state=RANDOM_SEED)
    df_mal_calib = df_mal_calib.reset_index(drop=True)
    df_mal_test = df_mal_test.reset_index(drop=True)
    n_calib_mal = len(df_mal_calib)
    n_test_mal = len(df_mal_test)
    print(f"[✓] Malignant Split: Calibration = {n_calib_mal} (20%) | Inference Testing = {n_test_mal} (80%)")

    # 3. Double those sets with benign samples AND track the unused benign images
    df_all_ben = df_raw[df_raw["cancer"] == 0].reset_index(drop=True)
    n_all_ben = len(df_all_ben)
    print(f"[*] Total benign images in dataset: {n_all_ben}")

    df_ben_shuffled = df_all_ben.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    df_ben_calib = df_ben_shuffled.iloc[:n_calib_mal].reset_index(drop=True)
    df_ben_test = df_ben_shuffled.iloc[n_calib_mal:n_calib_mal + n_test_mal].reset_index(drop=True)
    df_ben_unused = df_ben_shuffled.iloc[n_calib_mal + n_test_mal:].reset_index(drop=True)

    n_calib_ben = len(df_ben_calib)
    n_test_ben = len(df_ben_test)
    n_unused_ben = len(df_ben_unused)

    # Combine into balanced (50-50%) sets
    df_calib = pd.concat([df_mal_calib, df_ben_calib]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    df_calib["split_role"] = "calibration"

    df_test = pd.concat([df_mal_test, df_ben_test]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    df_test["split_role"] = "inference_testing"

    df_ben_unused["split_role"] = "unused_benign"

    print("\n--- Cohort Summary ---")
    print(f"  • Calibration Set (Doubled with Benign):      {len(df_calib)} images ({n_calib_mal} Malignant / {n_calib_ben} Benign)")
    print(f"  • Inference Testing Set (Doubled with Benign): {len(df_test)} images ({n_test_mal} Malignant / {n_test_ben} Benign)")
    print(f"  • Unused Benign Images Tracked:               {n_unused_ben} images")
    print(f"  • Total Dataset Accounted:                   {len(df_calib) + len(df_test) + n_unused_ben} / {len(df_raw)} images (100.0%)")

    # ---------------------------------------------------------
    # Step 2: Save All CSVs for All Sets in run_detection folder
    # ---------------------------------------------------------
    print("\n--- [Step 2: Saving CSVs for All Sets] ---")
    split_files = {
        "calibration_set.csv": df_calib,
        "calibration_malignant.csv": df_mal_calib,
        "calibration_benign.csv": df_ben_calib,
        "inference_testing_set.csv": df_test,
        "inference_testing_malignant.csv": df_mal_test,
        "inference_testing_benign.csv": df_ben_test,
        "unused_benign_images.csv": df_ben_unused,
        "all_malignant_images.csv": df_all_mal,
        "all_benign_images.csv": df_all_ben,
    }

    for fname, df_data in split_files.items():
        # Save in run root and splits/
        path_root = os.path.join(run_dir, fname)
        path_splits = os.path.join(run_splits_dir, fname)
        df_data.to_csv(path_root, index=False)
        df_data.to_csv(path_splits, index=False)
        print(f"    - Saved {fname:<30} ({len(df_data):>6} rows)")

    summary_df = pd.DataFrame([
        {"set_name": "calibration_set", "total_samples": len(df_calib), "malignant_count": n_calib_mal, "benign_count": n_calib_ben, "malignant_pct": 50.0},
        {"set_name": "inference_testing_set", "total_samples": len(df_test), "malignant_count": n_test_mal, "benign_count": n_test_ben, "malignant_pct": 50.0},
        {"set_name": "unused_benign_images", "total_samples": n_unused_ben, "malignant_count": 0, "benign_count": n_unused_ben, "malignant_pct": 0.0},
        {"set_name": "all_malignant_images", "total_samples": n_all_mal, "malignant_count": n_all_mal, "benign_count": 0, "malignant_pct": 100.0},
        {"set_name": "all_benign_images", "total_samples": n_all_ben, "malignant_count": 0, "benign_count": n_all_ben, "malignant_pct": 0.0},
    ])
    summary_df.to_csv(os.path.join(run_dir, "all_dataset_split_summary.csv"), index=False)
    summary_df.to_csv(os.path.join(run_splits_dir, "all_dataset_split_summary.csv"), index=False)
    print(f"[✓] Split CSVs successfully saved.")

    # ---------------------------------------------------------
    # Step 3: Compute Empirical Site Stats from Calibration Set AND Unused Benign
    # ---------------------------------------------------------
    print("\n--- [Step 3: Finding site_statistics from Calibration Set AND Unused Benign Images] ---")
    # Strictly zero data leakage: df_test is completely excluded from site statistics estimation
    df_site_pool = pd.concat([df_calib, df_ben_unused]).reset_index(drop=True)
    print(f"[*] Reference site statistics pool size: {len(df_site_pool)} images (Calibration + Unused Benign)")

    site_stats = compute_site_stats_from_pool(df_site_pool, data_dir=data_dir, sample_size=SITE_STATS_SAMPLE_SIZE)

    with open(os.path.join(run_results_dir, "site_stats.json"), "w") as f:
        json.dump(site_stats, f, indent=2)
    with open(os.path.join(run_dir, "site_stats.json"), "w") as f:
        json.dump(site_stats, f, indent=2)
    print(f"[✓] Empirical site_stats saved -> {os.path.join(run_results_dir, 'site_stats.json')}")

    preprocessor = MammogramPreprocessor(target_size=TARGET_IMG_SIZE, site_stats=site_stats)

    # ---------------------------------------------------------
    # Step 4: Cache Verification
    # ---------------------------------------------------------
    print("\n--- [Step 4: Image Cache Verification] ---")
    search_cache_dirs = [
        os.path.join(cache_root_dir, "train"),
        os.path.join(cache_root_dir, "val"),
        os.path.join(cache_root_dir, "test")
    ]
    existing_keys = {f[:-4] for d in search_cache_dirs if os.path.exists(d) for f in os.listdir(d) if f.endswith(".npy")}

    df_eval_all = pd.concat([df_calib, df_test]).reset_index(drop=True)
    missing = [r for _, r in df_eval_all.iterrows() if f"{r['patient_id']}_{r['image_id']}" not in existing_keys]

    if len(missing) > 0:
        print(f"[*] Preprocessing {len(missing)} missing images across {PROCESS_POOL_WORKERS} workers...")
        calib_cache_target = os.path.join(cache_root_dir, "train")
        os.makedirs(calib_cache_target, exist_ok=True)
        tasks = []
        for r in missing:
            pid = str(r["patient_id"])
            imid = str(r["image_id"])
            dest_p = os.path.join(calib_cache_target, f"{pid}_{imid}.npy")
            tasks.append((pid, imid, str(r.get("laterality", "R")), str(r.get("view", "MLO")), int(r.get("site_id", 1)), data_dir, dest_p, preprocessor.target_size, site_stats))

        with ProcessPoolExecutor(max_workers=PROCESS_POOL_WORKERS) as ex:
            list(ex.map(_precache_worker, tasks))
        print(f"[✓] Preprocessing finished.")
    else:
        print(f"[✓] All {len(df_eval_all)} calibration and testing images are present in the disk cache.")

    # ---------------------------------------------------------
    # Step 5: Instantiating Pretrained Mammo-CLIP Detector
    # ---------------------------------------------------------
    print("\n--- [Step 5: Instantiating Pretrained Mammo-CLIP Model] ---")
    model = RetinaNet_efficientnet(
        num_classes=1,
        model_type=MODEL_ARCH,
        clip_chk_pt=clip_chk_pt_path,
        freeze_backbone=FREEZE_BACKBONE
    ).to(device)

    # Strictly freeze backbone
    for p in model.efficientnet.parameters():
        p.requires_grad = False
    model.eval()

    # ---------------------------------------------------------
    # Step 6: Use Calibration Set to Calibrate Pre-trained Model
    # ---------------------------------------------------------
    print(f"\n--- [Step 6: Calibrating Model on Calibration Set ({len(df_calib)} images)] ---")
    calib_dataset = MammoDetectionDiskDataset(df_calib, search_cache_dirs, data_dir, preprocessor)
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=BATCH_SIZE_INFER,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    t0_calib = time.time()
    calib_feats_list, calib_targets_list = [], []
    print(f"[*] Extracting frozen C5 representations on calibration cohort...")
    with torch.no_grad():
        for c_imgs, c_targets, _, _ in calib_loader:
            c_imgs = c_imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                _, C3, C4, C5 = model.efficientnet(c_imgs)
                pooled_c5 = F.adaptive_avg_pool2d(C5, (1, 1)).flatten(1)
            calib_feats_list.append(pooled_c5.cpu().numpy())
            calib_targets_list.append(c_targets.numpy())

    X_calib = np.vstack(calib_feats_list)
    y_calib = np.concatenate(calib_targets_list)

    print(f"[*] Fitting calibrated linear classification probe on {X_calib.shape[0]} samples (dim: {X_calib.shape[1]})...")
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
    clf.fit(X_calib, y_calib)

    probe_w = torch.from_numpy(clf.coef_).float().to(device)
    probe_b = torch.from_numpy(clf.intercept_).float().to(device)

    model.probe = nn.Linear(2048, 1).to(device)
    model.probe.weight.data.copy_(probe_w)
    model.probe.bias.data.copy_(probe_b)

    calib_probs = clf.predict_proba(X_calib)[:, 1]

    # Threshold Sensitivity Analysis on Calibration Set
    sens_curve_path = os.path.join(run_results_dir, "sensitivity_threshold_curve.png")
    print(f"\n[*] Running Threshold Sensitivity Analysis on calibration cohort...")
    sensitivity_results = run_threshold_sensitivity_analysis(
        y_true=y_calib,
        y_probs=calib_probs,
        beta=CLINICAL_BETA,
        save_plot_path=sens_curve_path
    )
    opt_threshold = float(sensitivity_results["calibrated_optimal_threshold"])
    print(f"\n[★] Calibrated Optimal Threshold: {opt_threshold:.4f}")

    # Calibration set performance metrics
    calib_metrics = compute_metrics(y_calib, calib_probs, threshold=opt_threshold)
    print(f"[✓] Calibration Performance @ Threshold {opt_threshold:.4f}:")
    print(f"    - Balanced Accuracy: {calib_metrics['balanced_ACC']:.4f}")
    print(f"    - AUC:               {calib_metrics['AUC']:.4f}")
    print(f"    - Sensitivity:       {calib_metrics['SN']:.4f} ({calib_metrics['TP']} / {calib_metrics['true_label_cancer_cases']})")
    print(f"    - Specificity:       {calib_metrics['SP']:.4f} ({calib_metrics['TN']} / {calib_metrics['true_label_benign_cases']})")
    print(f"    - Precision:         {calib_metrics['precision']:.4f}")

    calib_record = {
        "calibrated_optimal_threshold": opt_threshold,
        "calibration_cohort_size": len(y_calib),
        "calibration_cancer_cases": int(np.sum(y_calib == 1)),
        "calibration_benign_cases": int(np.sum(y_calib == 0)),
        "calibration_metrics": calib_metrics,
        "threshold_curve_table": sensitivity_results["table_rows"]
    }
    with open(os.path.join(run_results_dir, "calibration_metrics.json"), "w") as f:
        json.dump(calib_record, f, indent=2)
    pd.DataFrame([calib_metrics]).to_csv(os.path.join(run_results_dir, "calibration_metrics.csv"), index=False)

    # Save calibrated model checkpoint
    best_ckpt_path = os.path.join(run_checkpoints_dir, "best_mammo_clip_detector.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "probe_w": probe_w.cpu(),
            "probe_b": probe_b.cpu(),
            "opt_threshold": opt_threshold,
            "calibrated_optimal_threshold": opt_threshold,
            "mode": "calibrated"
        },
        best_ckpt_path
    )
    print(f"[✓] Calibrated model saved -> {best_ckpt_path}")

    # ---------------------------------------------------------
    # Step 7: Inference on Held-Out Testing Set (1,854 images)
    # ---------------------------------------------------------
    print(f"\n--- [Step 7: Inference on Held-Out Testing Set ({len(df_test)} images)] ---")
    test_dataset = MammoDetectionDiskDataset(df_test, search_cache_dirs, data_dir, preprocessor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE_INFER,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    t0_infer = time.time()
    infer_scores, infer_targets, infer_pids, infer_imids = [], [], [], []
    raw_detector_scores = []
    infer_bboxes = []

    print(f"[*] Running batched inference across {len(df_test)} exams with AMP...")
    with torch.no_grad():
        for b_idx, (imgs, targets, pids, imids) in enumerate(test_loader, 1):
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                _, C3, C4, C5 = model.efficientnet(imgs)
                pooled_c5 = F.adaptive_avg_pool2d(C5, (1, 1)).flatten(1)
                b_logits = pooled_c5 @ probe_w.T + probe_b
                b_scores = torch.sigmoid(b_logits).squeeze(-1)

                # Raw detector outputs for reference
                features = model.fpn([C3, C4, C5])
                cls_maps = [model.classificationModel(f) for f in features]
                all_cls = torch.cat(cls_maps, dim=1)
                raw_max, _ = torch.max(all_cls.squeeze(-1), dim=1)

                # Spatial projection to locate mass bounding box
                spatial_logits = F.conv2d(C5, probe_w.view(1, 2048, 1, 1), bias=probe_b)
                spatial_probs = torch.sigmoid(spatial_logits)

            b_scores_np = b_scores.cpu().numpy().tolist()
            infer_scores.extend(b_scores_np)
            infer_targets.extend(targets.numpy().tolist())
            infer_pids.extend(pids)
            infer_imids.extend(imids)
            raw_detector_scores.extend(raw_max.cpu().numpy().tolist())

            # Bounding box localization
            sp_np = spatial_probs.squeeze(1).cpu().numpy()
            H_img, W_img = imgs.shape[2], imgs.shape[3]
            sy, sx = H_img / sp_np.shape[1], W_img / sp_np.shape[2]
            for bi in range(len(b_scores_np)):
                m_slice = sp_np[bi]
                y_max, x_max = np.unravel_index(np.argmax(m_slice), m_slice.shape)
                cx, cy = (x_max + 0.5) * sx, (y_max + 0.5) * sy
                bw, bh = W_img * 0.18, H_img * 0.14
                x1 = max(0.0, cx - bw / 2.0)
                y1 = max(0.0, cy - bh / 2.0)
                x2 = min(float(W_img), cx + bw / 2.0)
                y2 = min(float(H_img), cy + bh / 2.0)
                infer_bboxes.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])

            if b_idx % 25 == 0 or b_idx == len(test_loader):
                elapsed = time.time() - t0_infer
                processed = min(b_idx * BATCH_SIZE_INFER, len(df_test))
                rate = processed / max(elapsed, 1e-4)
                print(f"    [Inference] {processed}/{len(df_test)} ({processed/len(df_test)*100:.1f}%) | Speed: {rate:.1f} imgs/s")

    infer_scores = np.array(infer_scores)
    infer_targets = np.array(infer_targets)

    # ---------------------------------------------------------
    # Step 8: Comprehensive Evaluation & Demographic Fairness Audit
    # ---------------------------------------------------------
    print(f"\n--- [Step 8: Testing Metrics @ Calibrated Optimal Threshold ({opt_threshold:.4f})] ---")
    overall_metrics = compute_metrics(infer_targets, infer_scores, threshold=opt_threshold)

    print("\n" + "=" * 80)
    print(f"       EVALUATION METRICS ON HELD-OUT TESTING COHORT ({len(df_test)} EXAMS)")
    print("=" * 80)
    print(f"  Calibrated Optimal Threshold: {opt_threshold:.4f}")
    print(f"  Balanced Accuracy:            {overall_metrics['balanced_ACC']:.4f}")
    print(f"  Area Under Curve (AUC):       {overall_metrics['AUC']:.4f}")
    print(f"  Sensitivity (Recall / SN):    {overall_metrics['SN']:.4f} ({overall_metrics['TP']} / {overall_metrics['true_label_cancer_cases']})")
    print(f"  Specificity (SP):             {overall_metrics['SP']:.4f} ({overall_metrics['TN']} / {overall_metrics['true_label_benign_cases']})")
    print(f"  Precision:                    {overall_metrics['precision']:.4f}")
    print(f"  Clinical F-1.5:               {overall_metrics['f_beta']:.4f}")
    print(f"  Confusion Matrix:             TP={overall_metrics['TP']}, TN={overall_metrics['TN']}, FP={overall_metrics['FP']}, FN={overall_metrics['FN']}")
    print("=" * 80)

    # Save overall metrics
    pd.DataFrame([overall_metrics]).to_csv(os.path.join(run_results_dir, "inference_metrics.csv"), index=False)
    with open(os.path.join(run_results_dir, "inference_metrics.json"), "w") as f:
        json.dump(overall_metrics, f, indent=2)

    # Build predictions dataframe
    df_results = df_test.copy()
    df_results["pred_prob"] = infer_scores
    df_results["pred_label"] = (infer_scores >= opt_threshold).astype(int)
    df_results["pred_decision"] = np.where(df_results["pred_label"] == 1, "MALIGNANT", "BENIGN")
    df_results["raw_detector_score"] = raw_detector_scores
    df_results["pred_bbox"] = infer_bboxes

    pred_cols = ["patient_id", "image_id", "cancer", "pred_prob", "pred_label", "pred_decision", "raw_detector_score", "pred_bbox", "age_group", "site_id", "density_clean"]
    df_results[pred_cols].to_csv(os.path.join(run_results_dir, "inference_predictions.csv"), index=False)
    print(f"[✓] Inference predictions saved -> {os.path.join(run_results_dir, 'inference_predictions.csv')}")

    # Subgroup Demographic Fairness Audit
    subgroup_records = []
    for ag in ["before 20", "20-40", "40-55", "55+"]:
        sub = df_results[df_results["age_group"] == ag]
        if len(sub) > 0:
            m = compute_metrics(sub["cancer"].values, sub["pred_prob"].values, threshold=opt_threshold)
            subgroup_records.append({"group_category": "Age Group", "subgroup": ag, **m})

    for site in [1, 2]:
        sub = df_results[df_results["site_id"] == site]
        if len(sub) > 0:
            m = compute_metrics(sub["cancer"].values, sub["pred_prob"].values, threshold=opt_threshold)
            subgroup_records.append({"group_category": "Site ID", "subgroup": f"Site {site}", **m})

    for d_val in ["A", "B", "C", "D", "Unknown"]:
        sub = df_results[df_results["density_clean"] == d_val]
        if len(sub) > 0:
            m = compute_metrics(sub["cancer"].values, sub["pred_prob"].values, threshold=opt_threshold)
            subgroup_records.append({"group_category": "Breast Density", "subgroup": f"Density {d_val}", **m})

    df_subgroups = pd.DataFrame(subgroup_records)
    df_subgroups.to_csv(os.path.join(run_results_dir, "subgroup_metrics.csv"), index=False)
    with open(os.path.join(run_results_dir, "subgroup_metrics.json"), "w") as f:
        json.dump(subgroup_records, f, indent=2)

    display_cols = ["group_category", "subgroup", "total_samples", "cancer_cases", "balanced_ACC", "AUC", "SP", "SN"]
    print("\n--- Subgroup Fairness Audit ---")
    print(df_subgroups[display_cols].to_string(index=False))

    # ---------------------------------------------------------
    # Step 9: 6-Panel XAI Suite in "benign" & "malignant" Folders
    # ---------------------------------------------------------
    print(f"\n--- [Step 9: Generating 6-Panel XAI Suite (Folders: 'malignant' vs 'benign')] ---")
    xai_pos = df_results[df_results["cancer"] == 1]
    n_pos = min(len(xai_pos), XAI_EVAL_MAX_SAMPLES // 2)
    xai_neg = df_results[df_results["cancer"] == 0].sample(n=n_pos, random_state=RANDOM_SEED)
    df_xai_cohort = pd.concat([xai_pos.head(n_pos), xai_neg]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"[*] Generating full 6-panel XAI reports for {len(df_xai_cohort)} examinations:")
    print(f"    - True Malignant: {n_pos} -> Saving to {xai_malignant_dir}")
    print(f"    - True Benign:    {n_pos} -> Saving to {xai_benign_dir}")

    xai_engine = MammoDetectorXAI(
        model=model,
        clip_chk_pt_path=clip_chk_pt_path,
        concept_vec_path=concept_vec_path,
        device=device
    )
    xai_engine.set_calibrated_probe(probe_w, probe_b)

    t0_xai = time.time()
    xai_records = []

    for i, (_, row) in enumerate(df_xai_cohort.iterrows(), 1):
        pid = str(row["patient_id"])
        imid = str(row["image_id"])
        key = f"{pid}_{imid}"
        cancer_true = int(row.get("cancer", 0))

        # Retrieve base preprocessed image
        base_img = None
        for cdir in search_cache_dirs:
            candidate = os.path.join(cdir, f"{key}.npy")
            if os.path.exists(candidate):
                try:
                    base_img = np.load(candidate)
                    break
                except Exception:
                    pass

        if base_img is None:
            base_img = np.zeros(TARGET_IMG_SIZE, dtype=np.float32)

        input_tensor = torch.from_numpy(np.ascontiguousarray(base_img)).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)

        # 1. Det-CAM
        det_cam = xai_engine.compute_det_cam(input_tensor)

        # 2. Separated Localization vs. Classification Saliency
        cls_sal, loc_sal = xai_engine.compute_separated_saliency(input_tensor)

        # 3. D-RISE
        drise_map = xai_engine.compute_drise(input_tensor, num_masks=XAI_DRISE_MASKS)

        # 4. Mammo-FActOR
        factor_map = xai_engine.compute_mammo_factor(input_tensor)

        # 5. Monte Carlo Prediction Variability
        mc_var_map, score_std = xai_engine.compute_monte_carlo_variability(
            input_tensor, num_passes=XAI_MC_PASSES, noise_std=XAI_MC_NOISE_STD
        )

        # 6. Post-Calibration Bounding Box with calibrated optimal threshold
        bbox_info = xai_engine.extract_predicted_bounding_box(
            input_tensor, opt_threshold=opt_threshold, saliency_map=det_cam
        )
        bbox_info["score_std"] = score_std

        # Folder placement based on true label: "benign" or "malignant"
        target_subfolder = "malignant" if cancer_true == 1 else "benign"
        sample_out_dir = os.path.join(run_xai_dir, target_subfolder)
        fig_save_path = os.path.join(sample_out_dir, f"{pid}_{imid}_xai_report.png")
        geojson_save_path = os.path.join(sample_out_dir, f"{pid}_{imid}_detection.geojson")

        # Render 6-panel audit figure
        visualize_detector_xai(
            base_img=base_img,
            det_cam=det_cam,
            cls_saliency=cls_sal,
            loc_saliency=loc_sal,
            drise_map=drise_map,
            factor_map=factor_map,
            mc_variability=mc_var_map,
            bbox_info=bbox_info,
            true_label=cancer_true,
            save_path=fig_save_path,
            dpi=XAI_REPORT_DPI
        )

        # Save GeoJSON detection bounding box in the appropriate subfolder
        bx = bbox_info["bbox"]
        geojson_data = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[bx[0], bx[1]], [bx[2], bx[1]], [bx[2], bx[3]], [bx[0], bx[3]], [bx[0], bx[1]]]]
                },
                "properties": {
                    "patient_id": pid,
                    "image_id": imid,
                    "cancer_true": cancer_true,
                    "pred_score": bbox_info["score"],
                    "opt_threshold": opt_threshold,
                    "decision": bbox_info["decision"],
                    "mc_score_std": round(score_std, 4),
                    "true_label_folder": target_subfolder
                }
            }]
        }
        with open(geojson_save_path, "w") as f:
            json.dump(geojson_data, f, indent=2)

        xai_records.append({
            "patient_id": pid,
            "image_id": imid,
            "cancer_true": cancer_true,
            "true_label_folder": target_subfolder,
            "pred_score": bbox_info["score"],
            "decision": bbox_info["decision"],
            "bbox_x1": bx[0], "bbox_y1": bx[1], "bbox_x2": bx[2], "bbox_y2": bx[3],
            "mc_score_std": round(score_std, 4),
            "opt_threshold": opt_threshold,
            "report_path": fig_save_path
        })

        if i % 20 == 0 or i == len(df_xai_cohort):
            elapsed = time.time() - t0_xai
            rate = i / max(elapsed, 1e-4)
            print(f"    [XAI Progress] {i}/{len(df_xai_cohort)} exams rendered ({rate:.1f} exams/s | Elapsed: {elapsed:.1f}s)")

    # Save XAI master summary
    df_xai_summary = pd.DataFrame(xai_records)
    df_xai_summary.to_csv(os.path.join(run_xai_dir, "all_xai_summary.csv"), index=False)
    print(f"\n[✓] Master XAI summary saved -> {os.path.join(run_xai_dir, 'all_xai_summary.csv')}")

    # ---------------------------------------------------------
    # Step 10: GPU Memory Audit
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        max_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
        max_res_mb = torch.cuda.max_memory_reserved() / (1024**2)
        print(f"\n--- [Step 10: GPU Memory Audit] ---")
        print(f"[*] Peak GPU VRAM Allocated: {max_alloc_mb:.2f} MB ({max_alloc_mb/1024:.2f} GB / 24 GB)")
        print(f"[*] Peak GPU VRAM Reserved:  {max_res_mb:.2f} MB ({max_res_mb/1024:.2f} GB / 24 GB)")

    print("\n" + "=" * 85)
    print(f"[✓] Mammo-CLIP Detection & Calibration Pipeline successfully completed.")
    print(f"[*] All output artifacts preserved under: {run_dir}")
    print(f"    - Calibration Curve:     {sens_curve_path}")
    print(f"    - Calibrated Checkpoint: {best_ckpt_path}")
    print(f"    - Malignant XAI Reports: {xai_malignant_dir}")
    print(f"    - Benign XAI Reports:    {xai_benign_dir}")
    print("=" * 85)


if __name__ == "__main__":
    main()
