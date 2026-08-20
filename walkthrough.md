# Walkthrough: Refactor `megarag-mmkg-kaggle.ipynb` & Fix MinerU BaseModel State Dict Error

We refactored [megarag-mmkg-kaggle.ipynb](file:///d:/Github/reproduce_MegaRAG/megarag-mmkg-kaggle.ipynb) to reproduce MegaRAG cleanly on Kaggle using the repository `https://github.com/linhnguyen15492/reproduce_MegaRAG.git` containing `MegaRAG`, `LightRAG`, and `MinerU`.

## Root Cause & Fix for Step 6.1 `BaseModel` Error

### Issue
When running `magic-pdf` in Step 6.1, MinerU failed with:
```
Error(s) in loading state_dict for BaseModel:
Missing key(s) in state_dict: "backbone.conv.conv.weight", "backbone.conv.bn.weight", ...
```

### Cause
MinerU's OCR detection module (`BaseOCRV20`) expects PP-OCRv3 MobileNet/LCNet weights for `ch_PP-OCRv3_det_infer.pth` and `en_PP-OCRv3_det_infer.pth`. Previously, the alias mapping erroneously copied `ch_PP-OCRv5_det_infer.pth` (which uses an incompatible HGNet architecture) to `v3_det`, resulting in state_dict weight mismatch.

### Fix
In Step 3, the OCR alias mapping was corrected:
- `ch_PP-OCRv3_det_infer.pth`, `en_PP-OCRv3_det_infer.pth`, and `ch_PP-OCRv4_det_infer.pth` are linked to `Multilingual_PP-OCRv3_det_infer.pth` (which possesses the exact matching MobileNet/LCNet architecture).
- `en_PP-OCRv4_rec_infer.pth` and `latin_PP-OCRv3_rec_infer.pth` are linked to `ch_PP-OCRv4_rec_infer.pth`.

## Key Structure of the Notebook

1. **Step 1**: System setup and unified clone of `reproduce_MegaRAG`.
2. **Step 2**: Dependency installation and editable installs (`pip install -e`) for `MinerU`, `LightRAG`, `MegaRAG`.
3. **Step 3**: MinerU weights download and corrected OCR architecture alias mapping.
4. **Step 4**: OpenAI API Key and `MegaRAG/env.sh` configuration.
5. **Step 5**: Data loading from Kaggle Input datasets (`/kaggle/input/datasets`).
6. **Step 6**: MMKG Construction in 4 separate modular steps using `<pdf_name>_run/dumps/` and `<pdf_name>_run/exp/`:
   - **Step 6.1**: Parse PDF with MinerU.
   - **Step 6.2**: Convert PDF pages to images (`pdf2img.py`).
   - **Step 6.3**: Build page assets manifest (`build_page_assets.py`).
   - **Step 6.4**: Construct MMKG and embeddings (`construct_mmkg.py`).
7. **Step 7**: Query with MegaRAG (`query_mmkg.py`).
8. **Step 8**: View and analyze query results and graph structure.

## Verification Results

- All 11 code cells syntax-checked via Python `ast.parse`: **PASSED**
- JSON Notebook schema validated: **PASSED (24 total cells)**
