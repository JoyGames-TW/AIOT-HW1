# AIOT-HW1: Linear Regression Analysis and Interactive Streamlit App

## 專案概述

這個專案是人工智慧與物聯網（AIOT）課程的作業一，專注於線性回歸分析。專案包含兩個主要部分：

1. **原始 Python 腳本 (`app.py`)**：使用 scikit-learn 對糖尿病資料集進行線性回歸建模，包括基線模型和正則化模型（Ridge、Lasso、ElasticNet）。

2. **互動式 Streamlit 應用程式 (`streamlit_app.py`)**：提供圖形化介面，讓用戶可以即時調整參數生成合成數據，訓練模型，並視覺化結果。

## Demo

體驗線上 Demo：[https://aiot-hw1.streamlit.app/](https://aiot-hw1.streamlit.app/)

## 功能特色

### 原始腳本功能
- 使用 sklearn 的糖尿病資料集
- 實現完整的機器學習管道：資料預處理、模型訓練、評估
- 比較多種線性回歸模型的性能
- 保存訓練好的模型到 `linreg_pipeline.joblib`

### Streamlit 應用功能
- **參數設定**：
  - `n`: 樣本數量 (100-5000)
  - `a`: 線性關係斜率 (-10 到 10)
  - `noise`: 噪聲方差 (0-1000)
- **即時更新**：調整參數時圖表和結果自動重新計算
- **模型比較**：同時顯示 LinearRegression、Ridge、Lasso、ElasticNet 的評估指標
- **視覺化**：
  - 散點圖顯示實際數據點
  - 預測線
  - 標籤前五個預測誤差最大的點（最差點）
- **響應式佈局**：左側參數設定，右側結果顯示

## 安裝與運行

### 環境需求
- Python 3.7+
- 推薦使用虛擬環境

### 安裝步驟

1. **克隆倉庫**
   ```bash
   git clone https://github.com/JoyGames-TW/AIOT-HW1.git
   cd AIOT-HW1
   ```

2. **安裝依賴**
   ```bash
   pip install -r requirements.txt
   ```

### 運行方式

#### 運行原始腳本
```bash
python app.py
```
這將載入糖尿病資料集，訓練模型，並在控制台輸出評估結果。

#### 運行 Streamlit 應用
```bash
streamlit run streamlit_app.py
```
應用程式將在 http://localhost:8501 啟動。

## 檔案說明

- `app.py`: 原始線性回歸分析腳本
- `streamlit_app.py`: Streamlit 互動式應用程式
- `requirements.txt`: Python 依賴包列表
- `Log.md`: 專案開發日誌
- `linreg_pipeline.joblib`: 保存的線性回歸模型
- `README.MD`: 本文件

## 技術細節

### 使用的技術棧
- **Python**: 主要程式語言
- **scikit-learn**: 機器學習框架
- **Streamlit**: 網頁應用框架
- **matplotlib**: 圖表繪製
- **pandas & numpy**: 資料處理
- **joblib**: 模型序列化

### 模型評估指標
- R² (決定係數)
- MAE (平均絕對誤差)
- RMSE (均方根誤差)

### 資料處理流程
1. 資料載入與預處理
2. 特徵標準化
3. 訓練/測試分割
4. 模型訓練與調參
5. 性能評估
6. 結果視覺化

## 使用指南

### Streamlit 應用操作
1. 啟動應用後，在左側調整參數滑桿
2. 觀察右側的模型結果表格
3. 查看散點圖，注意標記的最差預測點
4. 調整參數觀察變化

### 參數解釋
- **樣本數量 (n)**: 影響數據集大小，更多樣本通常提高模型穩定性
- **斜率 (a)**: 決定線性關係的陡峭程度
- **噪聲方差**: 控制數據的隨機波動程度，較高值使預測更困難

## 開發日誌

詳細的開發過程記錄在 `Log.md` 文件中，包括每次修改和功能添加。

## 作者

JoyGames-TW

## 授權

此專案為教育用途，遵循適當的開源授權。