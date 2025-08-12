 # py code beginning

# streamlit_deep_only_multilabel_trainer.py
# -------------------------------------------------------------
# Streamlit app to train a Deep-only (DeepFM without FM) model
# for multi-label or multi-class targets on tabular data.
# -------------------------------------------------------------

# === Config Section ===
DEFAULT_PATTERN = r"^(like-|like_|food_|e\d+|.*like.*)$"  # Default regex for multi-label column detection
DEFAULT_HIDDEN = (512, 256, 128)   # Default hidden layers
DEFAULT_DROPOUT = 0.2              # Default dropout
DEFAULT_EPOCHS = 30                # Default number of epochs
DEFAULT_BATCH_SIZE = 256           # Default batch size
DEFAULT_LR = 2e-3                  # Default learning rate
DEFAULT_WEIGHT_DECAY = 1e-5        # Default weight decay
DEFAULT_THRESHOLD = 0.2            # Default decision threshold for multi-label
DEFAULT_ARTIFACTS_DIR = "artifacts"  # Fallback output directory if no preset override
DEFAULT_PAGE_TITLE = "Deep-only (DeepFM) Trainer"
DEFAULT_MAX_EMB_DIM = 16           # Cap embedding dimensions for small data
DEFAULT_AUTO_TUNE_THRESHOLD = True # Auto-tune threshold on validation for multi-label

# Presets for your three tasks (Option A)
PRESETS = {
    "None":   {"pattern": DEFAULT_PATTERN,        "artifacts": DEFAULT_ARTIFACTS_DIR},
    "Places": {"pattern": r"^(like[-_]?p\d+)$", "artifacts": "outputs/places"},
    "Foods":  {"pattern": r"^(like[-_]?e\d+)$", "artifacts": "outputs/foods"},
    "Gifts":  {"pattern": r"^(like[-_]?g\d+)$", "artifacts": "outputs/gifts"},
}

import io
import os
import re
import json
import time
import math
import zipfile
import numpy as np
import pandas as pd
import streamlit as st

# ML imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, accuracy_score, average_precision_score, roc_auc_score
except Exception as e:
    st.error("PyTorch / scikit-learn are required. Please install: pip install torch scikit-learn pandas numpy")
    raise

# Optional iterative stratification for multi-label
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    HAS_ITERSTRAT = True
except Exception:
    HAS_ITERSTRAT = False

st.set_page_config(page_title=DEFAULT_PAGE_TITLE, layout="wide")
st.title("🔧 Deep‑only (DeepFM) Trainer — Multi‑Label / Multi‑Class")
st.caption("Embeddings + MLP. No FM part. Designed for recommender-style tabular data.")

# ------------------------------
# Utilities
# ------------------------------

def is_categorical(series: pd.Series) -> bool:
    if series.dtype == "object" or pd.api.types.is_string_dtype(series):
        return True
    nunique = series.nunique(dropna=True)
    return nunique > 0 and nunique <= max(50, int(0.05 * len(series)))

@st.cache_data(show_spinner=False)
def load_table(file_bytes: bytes, filename: str, sheet: str | None):
    """Always return a DataFrame.
    - If Excel and no sheet provided -> load first sheet (sheet_name=0)
    - Guard against dict return by picking the first sheet.
    """
    if filename.lower().endswith((".xlsx", ".xls")):
        sheet_arg = 0 if (sheet is None or str(sheet).strip() == "") else sheet
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_arg)
        if isinstance(df, dict):  # safety: fall back to first sheet
            first_key = next(iter(df))
            df = df[first_key]
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))
    return df

# ------------------------------
# Components: Model & Preprocess
# ------------------------------
class Preprocessor:
    def __init__(self, cat_cols, num_cols):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.cat_maps = {}
        self.num_means = {}
        self.num_stds = {}

    def fit(self, df: pd.DataFrame):
        for c in self.cat_cols:
            vals = df[c].astype("category").cat.categories.tolist()
            self.cat_maps[c] = {v: i + 1 for i, v in enumerate(vals)}  # 0 for unknown
        for c in self.num_cols:
            col = pd.to_numeric(df[c], errors="coerce")
            m = float(col.mean()) if col.notna().any() else 0.0
            s = float(col.std(ddof=0)) if col.notna().any() else 1.0
            if s == 0.0:
                s = 1.0
            self.num_means[c] = m
            self.num_stds[c] = s

    def transform(self, df: pd.DataFrame):
        cats = []
        for c in self.cat_cols:
            m = self.cat_maps[c]
            arr = df[c].map(m).fillna(0).astype(int).to_numpy().reshape(-1, 1)
            cats.append(arr)
        X_cat = np.hstack(cats) if cats else np.zeros((len(df), 0), dtype=np.int64)

        nums = []
        for c in self.num_cols:
            col = pd.to_numeric(df[c], errors="coerce").fillna(self.num_means[c])
            z = (col - self.num_means[c]) / self.num_stds[c]
            nums.append(z.to_numpy().reshape(-1, 1))
        X_num = np.hstack(nums) if nums else np.zeros((len(df), 0), dtype=np.float32)
        return X_cat.astype(np.int64), X_num.astype(np.float32)

class DeepOnlyRec(nn.Module):
    def __init__(self, cat_cards, num_dim, out_dim, hidden=DEFAULT_HIDDEN, pdrop=DEFAULT_DROPOUT):
        super().__init__()
        self.embeddings = nn.ModuleList()
        self.emb_dims = []
        for card in cat_cards:
            if card > 0:
                dim = min(DEFAULT_MAX_EMB_DIM, max(4, int(round(1.6 * (card ** 0.56)))))
                self.embeddings.append(nn.Embedding(card + 1, dim))  # +1 for unknown index=0
                self.emb_dims.append(dim)
            else:
                self.embeddings.append(None)
                self.emb_dims.append(0)
        input_dim = sum(self.emb_dims) + num_dim
        layers = []
        for h in hidden:
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(pdrop)]
            input_dim = h
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.out = nn.Linear(input_dim, out_dim)

    def forward(self, Xc, Xn):
        if Xc.numel() > 0:
            embs = [emb(Xc[:, i]) for i, emb in enumerate(self.embeddings) if emb is not None]
            embs = torch.cat(embs, dim=1) if len(embs) else torch.zeros((Xc.size(0), 0), device=Xc.device)
        else:
            embs = torch.zeros((Xn.size(0), 0), device=Xn.device)
        x = torch.cat([embs, Xn], dim=1) if Xn.numel() > 0 else embs
        h = self.mlp(x)
        return self.out(h)  # logits

class TabDS(Dataset):
    def __init__(self, Xc, Xn, y):
        self.Xc = torch.from_numpy(Xc).long()
        self.Xn = torch.from_numpy(Xn).float()
        self.y  = torch.from_numpy(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.Xc[idx], self.Xn[idx], self.y[idx]

# ------------------------------
# Sidebar: Data, Presets & Settings
# ------------------------------
with st.sidebar:
    st.header("📄 Data")
    uploaded = st.file_uploader("Upload Excel/CSV", type=["xlsx", "xls", "csv"], accept_multiple_files=False)
    sheet_name = st.text_input("Excel sheet (optional)", value="")
    load_btn = st.button("Load data", use_container_width=True)

    st.divider()
    st.header("🎛 Preset")
    preset = st.selectbox("Task preset", list(PRESETS.keys()), index=1)  # default to Places
    preset_cfg = PRESETS[preset]

    st.divider()
    st.header("🎯 Target Type")
    target_mode = st.radio(
        "What are we predicting?",
        ["Multi-label (multiple foods per user)", "Multi-class (exactly one food)"]
    )

    st.caption("Multi-label → BCEWithLogitsLoss; Multi-class → CrossEntropyLoss")

    # Artifacts path (preset-driven but editable)
    artifacts_dir = st.text_input("Artifacts output folder", value=preset_cfg["artifacts"]) or DEFAULT_ARTIFACTS_DIR

# Main area
if uploaded and (load_btn or "df" in st.session_state):
    if load_btn:
        file_bytes = uploaded.read()
        try:
            df = load_table(file_bytes, uploaded.name, sheet_name or None)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            st.stop()
        st.session_state.df = df
    else:
        df = st.session_state.df

    # Column list & preset label candidates
all_cols = df.columns.tolist()
if target_mode.startswith("Multi-label"):
    preset_pattern = PRESETS.get(preset, {}).get("pattern", DEFAULT_PATTERN)
    try:
        preset_label_cols = [c for c in all_cols if re.search(preset_pattern, c, re.IGNORECASE)]
    except re.error:
        preset_label_cols = []
else:
    preset_label_cols = []

# Preview with optional label hiding
# --- Preview & feature options ---

all_cols = df.columns.tolist()
if target_mode.startswith("Multi-label"):
    preset_pattern = PRESETS.get(preset, {}).get("pattern", DEFAULT_PATTERN)
    try:
        preset_label_cols = [c for c in all_cols if re.search(preset_pattern, c, re.IGNORECASE)]
    except re.error:
        preset_label_cols = []
else:
    preset_label_cols = []

# Preview (optionally hide label columns)
st.subheader("👀 Preview")
hide_labels = st.checkbox(
    "Hide label columns in preview",
    value=True,
    help="Hide columns matched by the preset pattern (like-p*, like-e*, like-g*)."
)
df_view = df.drop(columns=preset_label_cols, errors="ignore") if hide_labels else df
st.dataframe(df_view.head(30), use_container_width=True)

# Feature suggestion (exclude label columns from options)
feature_options_all = [c for c in all_cols if c not in preset_label_cols]
suggested_cat = (
    [c for c in feature_options_all if is_categorical(df[c])]
    if feature_options_all else []
)
suggested_num = (
    [c for c in feature_options_all if c not in suggested_cat and pd.api.types.is_numeric_dtype(df[c])]
    if feature_options_all else []
)

st.subheader("🧱 Features")
    with st.expander("Select categorical & numeric features", expanded=True):
        cat_cols = st.multiselect(
            "Categorical columns",
            options=feature_options_all,
            default=suggested_cat
        )
        num_cols = st.multiselect(
            "Numeric columns",
            options=[c for c in feature_options_all if c not in cat_cols],
            default=suggested_num
        )

    # Target configuration
    st.subheader("🎯 Target configuration")
    pattern_default = preset_cfg["pattern"] if target_mode.startswith("Multi-label") else DEFAULT_PATTERN
    if target_mode.startswith("Multi-label"):
        with st.expander("Choose label columns (binary) OR a delimited text column", expanded=True):
            pattern = st.text_input("Auto-pick pattern (regex)", value=pattern_default, help="Use presets: Places/Foods/Gifts")
            try:
                suggested = [c for c in all_cols if re.search(pattern, c, re.IGNORECASE)]
            except re.error:
                st.warning("Invalid regex; using default pattern.")
                suggested = [c for c in all_cols if re.search(DEFAULT_PATTERN, c, re.IGNORECASE)]
            label_cols = st.multiselect("Binary label columns (multi-hot)", options=all_cols, default=suggested)
            st.markdown("**OR**")
            text_label_col = st.selectbox("Delimited text label column", options=["(none)"] + all_cols, index=0)
            delim = st.text_input("Delimiter (if using text column)", value=",")
    else:
        with st.expander("Pick the single target column (categorical)", expanded=True):
            default_idx = 0
            for i, c in enumerate(all_cols):
                if re.search(r"most.*food|food.*choice|most.*place|most.*gift", c, re.IGNORECASE):
                    default_idx = i + 1
                    break
            target_single = st.selectbox("Target column", options=["(none)"] + all_cols, index=default_idx)

    st.subheader("⚙️ Hyperparameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=int(DEFAULT_EPOCHS), step=1)
        batch_size = st.number_input("Batch size", min_value=16, max_value=4096, value=int(DEFAULT_BATCH_SIZE), step=16)
    with c2:
        lr = st.number_input("Learning rate", min_value=1e-5, max_value=1e-1, value=float(DEFAULT_LR), step=1e-5, format="%.5f")
        weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=1e-2, value=float(DEFAULT_WEIGHT_DECAY), step=1e-5, format="%.5f")
    with c3:
        hidden_str = st.text_input("Hidden layers", value=",".join(str(x) for x in DEFAULT_HIDDEN))
        dropout = st.slider("Dropout", min_value=0.0, max_value=0.8, value=float(DEFAULT_DROPOUT), step=0.05)

    threshold = st.slider("Decision threshold (multi-label)", 0.05, 0.95, float(DEFAULT_THRESHOLD), 0.05)
    auto_tune_thresh = st.checkbox("Auto‑tune threshold on validation", value=DEFAULT_AUTO_TUNE_THRESHOLD)

    start = st.button("🚀 Start training", type="primary")

    # ------------------------------
    # Training
    # ------------------------------
    if start:
        # Parse hidden layers
        try:
            hidden = tuple(int(x.strip()) for x in hidden_str.split(',') if x.strip())
            assert len(hidden) > 0
        except Exception:
            st.error("Hidden layers must be a comma-separated list of integers, e.g., 512,256,128")
            st.stop()

        # Build labels
        if target_mode.startswith("Multi-label"):
            if label_cols:
                Y = df[label_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
                class_names = label_cols
            elif text_label_col and text_label_col != "(none)":
                col = df[text_label_col].astype(str).fillna("")
                tokens = set()
                for s in col:
                    parts = [p.strip() for p in s.split(delim) if p.strip()]
                    tokens.update(parts)
                class_names = sorted(tokens)
                idx = {c: i for i, c in enumerate(class_names)}
                Y = np.zeros((len(df), len(class_names)), dtype=np.float32)
                for r, s in enumerate(col):
                    parts = [p.strip() for p in s.split(delim) if p.strip()]
                    for p in parts:
                        if p in idx:
                            Y[r, idx[p]] = 1.0
            else:
                st.error("For multi-label, select binary label columns or a text column + delimiter.")
                st.stop()
        else:
            if not target_single or target_single == "(none)":
                st.error("Please select a single target column for multi-class.")
                st.stop()
            classes = sorted(df[target_single].dropna().astype(str).unique())
            cls_to_id = {c: i for i, c in enumerate(classes)}
            Y = df[target_single].astype(str).map(cls_to_id).fillna(0).astype(int).values
            class_names = classes

        # 🔒 Auto-exclude label columns from features
        if target_mode.startswith("Multi-label") and label_cols:
            label_set = set(class_names)
        elif not target_mode.startswith("Multi-label"):
            label_set = {target_single}
        else:
            label_set = set()
        cat_cols = [c for c in cat_cols if c not in label_set]
        num_cols = [c for c in num_cols if c not in label_set]
        if len(cat_cols) == 0 and len(num_cols) == 0:
            st.error("No features selected after excluding label columns. Please select some non-label features.")
            st.stop()

        # 🔎 Label stats
        with st.expander("🔎 Label stats", expanded=False):
            pos_counts = np.asarray(Y).sum(axis=0)
            st.write({name: int(cnt) for name, cnt in zip(class_names, pos_counts)})
            st.write("Avg labels per sample:", float(np.asarray(Y).mean()))

        # Preprocess features
        pre = Preprocessor(cat_cols, num_cols)
        pre.fit(df)
        Xc, Xn = pre.transform(df)

        # Train/val split
        if target_mode.startswith("Multi-label"):
            if HAS_ITERSTRAT:
                msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                tr_idx, va_idx = next(msss.split(np.zeros(len(Y)), Y))
                Xc_tr, Xc_va = Xc[tr_idx], Xc[va_idx]
                Xn_tr, Xn_va = Xn[tr_idx], Xn[va_idx]
                Y_tr,  Y_va  = Y[tr_idx],  Y[va_idx]
            else:
                Xc_tr, Xc_va, Xn_tr, Xn_va, Y_tr, Y_va = train_test_split(
                    Xc, Xn, Y, test_size=0.2, random_state=42
                )
            out_dim = Y.shape[1]
        else:
            Xc_tr, Xc_va, Xn_tr, Xn_va, Y_tr, Y_va = train_test_split(
                Xc, Xn, Y, test_size=0.2, random_state=42, stratify=Y
            )
            out_dim = len(class_names)

        # Model
        cat_cards = [len(pre.cat_maps[c]) for c in cat_cols]
        model = DeepOnlyRec(cat_cards, Xn_tr.shape[1], out_dim, hidden=hidden, pdrop=dropout)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Data loaders
        if target_mode.startswith("Multi-label"):
            tr_ds = TabDS(Xc_tr, Xn_tr, Y_tr.astype(np.float32))
            va_ds = TabDS(Xc_va, Xn_va, Y_va.astype(np.float32))
        else:
            tr_ds = TabDS(Xc_tr, Xn_tr, Y_tr.astype(np.int64))
            va_ds = TabDS(Xc_va, Xn_va, Y_va.astype(np.int64))
        tr_dl = DataLoader(tr_ds, batch_size=int(batch_size), shuffle=True, num_workers=0)
        va_dl = DataLoader(va_ds, batch_size=max(512, int(batch_size)), shuffle=False, num_workers=0)

        # Loss & Optim
        if target_mode.startswith("Multi-label"):
            N = Y_tr.shape[0]
            P = np.clip(Y_tr.sum(axis=0), 1.0, None)
            pos_weight = torch.tensor((N - P) / P, dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

        # Training loop
        progress = st.progress(0)
        log_area = st.empty()
        best_score = -1.0
        best_state = None
        lines = []

        for epoch in range(1, int(epochs) + 1):
            model.train()
            total_loss = 0.0
            for Xc_b, Xn_b, y_b in tr_dl:
                Xc_b, Xn_b = Xc_b.to(device), Xn_b.to(device)
                opt.zero_grad()
                logits = model(Xc_b, Xn_b)
                y_b = y_b.to(device)
                loss = criterion(logits, y_b)
                loss.backward()
                opt.step()
                total_loss += loss.item() * (y_b.size(0) if target_mode.startswith("Multi-label") else y_b.shape[0])
            train_loss = total_loss / len(tr_ds)

            # Eval
            model.eval()
            with torch.no_grad():
                all_logits, all_true = [], []
                for Xc_b, Xn_b, y_b in va_dl:
                    Xc_b, Xn_b = Xc_b.to(device), Xn_b.to(device)
                    logits = model(Xc_b, Xn_b)
                    all_logits.append(logits.cpu().numpy())
                    all_true.append(y_b.numpy())
                logits = np.concatenate(all_logits)
                y_true = np.concatenate(all_true)

                if target_mode.startswith("Multi-label"):
                    probs = 1.0 / (1.0 + np.exp(-logits))
                    used_threshold = float(threshold)
                    if auto_tune_thresh:
                        ts = np.linspace(0.05, 0.5, 10)
                        best_t, best_micro = used_threshold, 0.0
                        for t in ts:
                            preds_t = (probs >= t).astype(int)
                            micro_t = f1_score(y_true, preds_t, average="micro", zero_division=0)
                            if micro_t > best_micro:
                                best_t, best_micro = t, micro_t
                        used_threshold = float(best_t)
                    preds = (probs >= used_threshold).astype(int)
                    micro_f1 = f1_score(y_true, preds, average="micro", zero_division=0)
                    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
                    ap_list = []
                    for j in range(probs.shape[1]):
                        try:
                            ap = average_precision_score(y_true[:, j], probs[:, j])
                        except Exception:
                            ap = float("nan")
                        ap_list.append(ap)
                    valid = [a for a in ap_list if not math.isnan(a)]
                    mAP = float(np.mean(valid)) if valid else float("nan")
                    score_display = f"microF1={micro_f1:.4f} | macroF1={macro_f1:.4f} | mAP={mAP:.4f} | thr={used_threshold:.2f}"
                    score_for_model = mAP if not math.isnan(mAP) else micro_f1
                else:
                    preds = logits.argmax(axis=1)
                    acc = accuracy_score(y_true, preds)
                    try:
                        auc = roc_auc_score(y_true, logits, multi_class="ovr")
                    except Exception:
                        auc = float("nan")
                    score_display = f"acc={acc:.4f} | auc(ovr)={auc:.4f}"
                    score_for_model = acc

            line = f"Epoch {epoch:02d} | loss={train_loss:.4f} | {score_display}"
            lines.append(line)
            log_area.code("\n".join(lines), language="text")
            progress.progress(epoch / int(epochs))

            if score_for_model > best_score:
                best_score = score_for_model
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        st.success("Training complete!")

        # ------------------------------
        # Save artifacts
        # ------------------------------
        os.makedirs(artifacts_dir, exist_ok=True)

        # Task-aware filenames
        task_slug = (preset.lower() if preset in ("Places","Foods","Gifts") else "generic").replace(" ", "_")
        stamp = time.strftime("%Y%m%d-%H%M%S")

        model_path   = os.path.join(artifacts_dir, f"{task_slug}_model_{stamp}.pt")
        pre_path     = os.path.join(artifacts_dir, f"{task_slug}_preprocess_{stamp}.pkl")
        classes_path = os.path.join(artifacts_dir, f"{task_slug}_class_names_{stamp}.json")
        cfg_used_path= os.path.join(artifacts_dir, f"{task_slug}_config_{stamp}.json")
        log_path     = os.path.join(artifacts_dir, f"{task_slug}_log_{stamp}.txt")
        zip_name     = f"{task_slug}_artifacts_{stamp}.zip"

        # Model
        torch.save(best_state if best_state is not None else model.state_dict(), model_path)

        # Preprocess
        with open(pre_path, "wb") as f:
            import pickle
            pickle.dump(
                {
                    "cat_cols": pre.cat_cols,
                    "num_cols": pre.num_cols,
                    "cat_maps": pre.cat_maps,
                    "num_means": pre.num_means,
                    "num_stds": pre.num_stds,
                },
                f,
            )

        # Class names
        with open(classes_path, "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)

        # Feature config used
        with open(cfg_used_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categorical_cols": pre.cat_cols,
                    "numeric_cols": pre.num_cols,
                    "target_mode": target_mode,
                    "class_names": class_names,
                    "hidden": list(hidden),
                    "dropout": float(dropout),
                    "max_emb_dim": int(DEFAULT_MAX_EMB_DIM),
                    "auto_tuned_threshold": bool(auto_tune_thresh),
                    "preset": preset,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Training log
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("
".join(lines))

        # Zip bundle for download
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(model_path,   arcname=os.path.basename(model_path))
            z.write(pre_path,     arcname=os.path.basename(pre_path))
            z.write(classes_path, arcname=os.path.basename(classes_path))
            z.write(cfg_used_path,arcname=os.path.basename(cfg_used_path))
            z.write(log_path,     arcname=os.path.basename(log_path))
        buf.seek(0)

        st.download_button(
            "⬇️ Download artifacts (ZIP)",
            data=buf,
            file_name=zip_name,
            mime="application/zip",
            use_container_width=True,
        )",
            data=buf,
            file_name="deep_only_artifacts.zip",
            mime="application/zip",
            use_container_width=True,
        )

else:
    st.info("Upload your dataset on the left and click **Load data** to begin.")

