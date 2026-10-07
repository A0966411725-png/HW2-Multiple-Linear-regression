# CLAUDE.md — HW2 作業討論用（繳交前必須刪除）

> ⚠️ **這個檔案只用於繳交前和 Claude 討論作業。**
> **繳交前的最後一步：刪除本檔（`git rm CLAUDE.md` 後 commit + push），而且不要放進 zip。**
> 只要使用者提到「準備繳交 / 打包 / 最後檢查」，就先提醒他刪掉這個檔案。

## 作業背景
- 課程作業 HW2：多元線性回歸，依循 CRISP-DM 流程，延伸自 HW1（https://github.com/A0966411725-png/HW1-CWA-Weather）。
- 學號：**7115064191**；檔名一律用 `7115064191_hw2.*`。
- 資料集：Kaggle「House Sales in King County, USA」https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
  （`kc_house_data.csv`，21,613 筆，18 個特徵，目標 `price`；檔案是從 GitHub 鏡像 Shreyas3108/house-price-prediction 下載的，內容與 Kaggle 原檔相同）。
- 作業要求重點：10–20 個特徵的 Kaggle 資料集、線性回歸、**特徵選擇**、**模型評估**、**含信賴區間／預測區間的預測圖**、CRISP-DM 六個步驟、GPT 輔助內容（對話匯出成 PDF）、NotebookLM 摘要（100 字以上）、網路主流解法比較。
- 評分：文件說明 50%（CRISP-DM 25%、GPT 與 NotebookLM 15%、資料來源與脈絡 10%）；結果呈現 50%（模型可執行、特徵選擇與評估 25%、結果美觀有說服力 15%、預測圖與評估指標 10%）。

## 繳交內容
- 主程式：`7115064191_hw2.py`
- 報告：`7115064191_hw2_report.pdf`（由 `report.html` 用 Chrome headless 轉成 PDF）
- 壓縮檔：`7115064191_hw2.zip`（被 .gitignore 排除，需在本機重新產生）
- 選擇性：GitHub 連結 https://github.com/A0966411725-png/HW2-Multiple-Linear-regression ，README.md 整理流程與成果

## 檔案結構
| 檔案 | 用途 |
|---|---|
| `7115064191_hw2.py` | 完整 CRISP-DM 流程，輸出 figures/、results/、model/ |
| `predict.py` | 部署示範：輸入房屋條件 → 預測價 + 95% CI/PI |
| `report.html` → `7115064191_hw2_report.pdf` | 報告（第 8 節 AI 對話、第 9 節 NotebookLM 摘要） |
| `ai_conversation.html` → `7115064191_ai_conversation.pdf` | AI 對話紀錄（AI 工具是 Claude Code，不是 ChatGPT，報告裡已如實註明） |
| `figures/` | 11 張圖（01–11） |
| `results/` | metrics.json、model_comparison.csv、feature_selection.csv、m3_ols_summary.txt、run_log.txt |
| `model/house_price_mlr.pkl` | M4 模型（已 remove_data，約 60 KB） |

## 執行
```bash
pip install -r requirements.txt
python 7115064191_hw2.py        # Mac 用 python3；約 30 秒
python predict.py --sqft_living 2000 --grade 8 --zipcode 98052 --lat 47.68
```

## 目前結果（測試集 4,284 筆，seed 42）
| 模型 | 特徵數 | R²(log) | CV R² | MAPE | 95% PI 覆蓋率 |
|---|---|---|---|---|---|
| M1 單純線性回歸 | 1 | 0.453 | 0.455 | 33.1% | – |
| M2 MLR 全部特徵 | 17 | 0.775 | 0.772 | 19.8% | – |
| M3 MLR 特徵選擇後 | 12 | 0.761 | 0.761 | 20.3% | 94.4% |
| M4 M3 + zipcode one-hot | 12+69 | 0.878 | 0.879 | 13.7% | 94.0% |
| HistGradientBoosting（對照） | 18 | 0.905 | 0.903 | 11.6% | – |

特徵選擇：Backward Elimination + RFECV + LassoCV 投票（≥2 票保留），再以 VIF ≤ 5 修剪（每次取 VIF 最高的特徵和它最相關的夥伴，刪掉與目標相關較弱者）。

## 同步規則（Mac ↔ Windows）
- 開始工作前先 `git pull`，做完 `git add -A && git commit && git push`。
- 不要兩台電腦同時改同一個檔案。
- Mac 路徑：`/Users/ian/本地/HW2-Multiple Linear regression `（名稱結尾有一個空格）
- Windows 路徑：`C:\Users\a0966\OneDrive\Desktop\HW2-Multiple-Linear-regression`
- 修改程式或數字後，要同步更新 `report.html`、README.md，並重新產生 PDF。

## 繳交前檢查清單
1. 重新執行主程式，確認數字和報告、README 一致。
2. 重新產生報告 PDF。
3. **刪除 `CLAUDE.md`**（`git rm CLAUDE.md`，commit + push）。
4. 重新壓縮 `7115064191_hw2.zip`，確認裡面沒有 CLAUDE.md。
