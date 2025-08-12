# Deep-only (DeepFM-style) Tabular Trainer — Multi-Label / Multi-Class

A Streamlit app for training **Deep-only** (embeddings + MLP; no FM term) models on tabular data.  
Designed for your dataset to predict **Places**, **Foods**, and **Gifts** from user features.

## Highlights
- Upload **Excel/CSV** (sheet picker supported)
- **Presets** for Places / Foods / Gifts
- Multi-**label** (BCEWithLogits) or Multi-**class** (CrossEntropy)
- Auto-detect & edit **categorical / numeric** features
- **Auto-exclude** label columns from features
- **Label stats** panel (per-class positives)
- **Multilabel stratified split** (iterstrat) when available
- **Auto-tune threshold** on validation (multi-label)
- Task-aware artifact filenames (e.g., `places_model_YYYYMMDD-HHMM.pt`)
