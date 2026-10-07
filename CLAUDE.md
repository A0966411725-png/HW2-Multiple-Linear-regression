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
| `7115064191_hw2.py` | 完整 CRISP-DM 流程，輸出 figures/、results/、model/，並自動寫 `results/run_log.txt` |
| `predict.py` | 部署示範：輸入房屋條件 → 預測價 + 95% CI/PI |
| `report.html` → `7115064191_hw2_report.pdf` | 報告（10 節：CRISP-DM 1–6、7 主流解法比較、8 GPT 輔助內容、9 NotebookLM 摘要、10 結論與限制），約 19 頁 |
| `ai_conversation.html` → `7115064191_ai_conversation.pdf` | AI 對話紀錄，分「第一段 Mac」「第二段 Windows」（AI 工具是 Claude Code，不是 ChatGPT，報告裡已如實註明） |
| `figures/` | 15 張圖（01–15；12 forward stepwise、13 PI 校準、14 殘差地圖、15 AR 補充） |
| `results/` | metrics.json、model_comparison.csv、feature_selection.csv、forward_stepwise.csv、pi_calibration.csv、autoregression.json、m3_ols_summary.txt、run_log.txt |
| `model/house_price_mlr.pkl` | M4 模型（已 remove_data，約 60 KB） |

## 執行
```bash
pip install -r requirements.txt
python 7115064191_hw2.py        # Mac 用 python3；約 1 分鐘（含 RF 的 5-fold CV）
python predict.py --sqft_living 2000 --grade 8 --zipcode 98052 --lat 47.68
```
- **Windows 注意**：這台 Windows 有「智慧型應用程式控制」，最新版 pandas 3.x / scikit-learn 的 DLL 會被封鎖。
  已在專案內建 `.venv`（被 .gitignore 排除），裝的是 `pandas==2.2.3`、`scikit-learn==1.5.2`，執行請用 `.venv/Scripts/python`。
- 報告 PDF 產生方式（Windows）：
  `"C:\Program Files\Google\Chrome\Application\chrome.exe" --headless=new --no-pdf-header-footer --print-to-pdf=<絕對路徑>\7115064191_hw2_report.pdf file:///<絕對路徑>/report.html`
  （Mac 用 Google Chrome 同樣參數）。report.html 字型已含 PingFang TC（Mac）與 Microsoft JhengHei（Windows）。

## 目前結果（測試集 4,284 筆，seed 42；Mac 與 Windows 重跑結果一致）
| 模型 | 特徵數 | R²(log) | CV R² | MAPE | 95% PI 覆蓋率 |
|---|---|---|---|---|---|
| M1 單純線性回歸 | 1 | 0.453 | 0.455 | 33.1% | – |
| M2 MLR 全部特徵 | 17 | 0.775 | 0.772 | 19.8% | – |
| M3 MLR 特徵選擇後 | 12 | 0.761 | 0.761 | 20.3% | 94.4% |
| M4 M3 + zipcode one-hot | 12+69 | 0.878 | 0.879 | 13.7% | 94.0% |
| Random Forest（對照） | 18 | 0.892 | 0.889 | 12.3% | – |
| HistGradientBoosting（對照） | 18 | 0.905 | 0.903 | 11.6% | – |

特徵選擇：Backward Elimination + RFECV + LassoCV 投票（≥2 票保留），再以 VIF ≤ 5 修剪（每次取 VIF 最高的特徵和它最相關的夥伴，刪掉與目標相關較弱者）。Forward Stepwise（AIC）只當佐證、不投票。
- Random Forest 在不同平台/版本的數字會有極小差異（第 3 位小數），報告採用 Windows 這次的數字。

其他補強結果（2026-10-07 Windows 加入，都已寫進報告）：
- M3 假設檢定：DW 1.99、最大 VIF 4.44、條件數 4.4；Breusch-Pagan p<0.001（異質變異）→ HC3 穩健 SE 下 11/12 仍顯著（只有 log_sqft_lot p=0.078）；Jarque-Bera p<0.001（峰度 4.0）。
- PI 校準（50/68/80/90/95/99%）：M3 52.6/71.1/81.7/90.2/94.4/98.1；M4 57.4/73.5/83.4/90.5/94.0/97.6。
- 殘差地圖：殘差 SD M3 0.258 → M4 0.185。
- AR(3) 每週中位價：預測 8 週 MAPE 4.6%，比 naive 4.3% 差 → 報告說明 AR 不適合當主模型。
- 成功標準檢核：M4 全部達成；M3 的 MAPE 20.3% 未達 ≤20%（報告已如實寫出）。

## 評分標準對照（老師的截圖）
- 文件 50%：CRISP-DM 完整（25%）→ 第 1–6 節 + 流程圖 + 成功標準檢核；GPT 與 NotebookLM（15%）→ 第 8、9 節 + 對話 PDF；資料來源與脈絡（10%）→ 開頭資料來源框 + 欄位字典。
- 結果 50%：可執行 + 特徵選擇與評估（25%）；美觀有說服力（15%）；Kaggle 名次/預測圖/指標（10%）→ 此資料集沒有 Kaggle 排行榜，第 7 節改用網路公開解法 R² 對照。

## 同步規則（Mac ↔ Windows）
- 開始工作前先 `git pull`，做完 `git add -A && git commit && git push`。
- 不要兩台電腦同時改同一個檔案。
- Mac 路徑：`/Users/ian/本地/HW2-Multiple Linear regression `（名稱結尾有一個空格）
- Windows 路徑：`D:\Claude code\物聯網HW2-Multiple Linear regression`（**不要用 OneDrive 桌面那份舊 clone**）
- 修改程式或數字後，要同步更新 `report.html`、README.md、`ai_conversation.html`，並重新產生兩份 PDF。
- 之後在任一台電腦有新的 AI 對話內容，也要補進報告第 8 節與 `ai_conversation.html`。

## 待使用者處理（Claude 無法代做）
- 作業要求「對話以 pdfCrowd 或其他方式匯出成 PDF」：目前的 `7115064191_ai_conversation.pdf` 是整理版。若要更保險，請使用者在 Claude 桌面版把兩段原始對話各自 Export，一起放進 zip。

## 繳交前檢查清單
1. 重新執行主程式，確認數字和報告、README 一致。
2. 重新產生報告 PDF 與對話紀錄 PDF。
3. **刪除 `CLAUDE.md`**（`git rm CLAUDE.md`，commit + push）。
4. 重新壓縮 `7115064191_hw2.zip`（不要包含 CLAUDE.md、.venv、.git），確認裡面沒有 CLAUDE.md。
