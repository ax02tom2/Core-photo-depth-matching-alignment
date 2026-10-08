# 岩心箱照片校正與成果輸出工具 — 跨 AI 程式結構說明

> 文件用途：提供其他 AI / 程式代理人 / 開發者快速理解本專案的**程式結構、資料流、影像處理邏輯、UI 互動、輸出規則與已知限制**。
>
> 文件對應程式版本：**2.11**
>
> 目前影像偵測器版本字串：`2.6-safe-hough-full-tray`
>
> 重要：本工具目前**沒有串接 GPT、Gemini、Claude 或其他 AI 視覺模型**。自動偵測完全以 Python + OpenCV + 幾何 / 顏色影像處理完成。若未來增加 AI，應在本文件新增獨立的 AI 模組與資料流說明，不應誤認目前 OpenCV 演算法為生成式 AI。

---

## 1. 專案定位

本工具是以 Streamlit 製作的「岩心箱照片校正與成果輸出」Web App，主要工作流程是：

```text
使用者上傳多張岩心箱照片
        ↓
檔名自然排序
        ↓
EXIF 修正 + 直式轉橫式
        ↓
OpenCV 自動尋找岩心箱四角
        ↓
使用者檢查／必要時拖曳四個角點
        ↓
依每張照片的個別留邊設定做透視校正
        ↓
最後一箱依孔號終深只保留實際深度槽數
        ↓
PDF / Word 成果輸出
```

核心目標不是辨識岩性，也不是做地質 AI 分析；目前主要是：

1. 將拍攝角度歪斜的岩心箱照片校正成可比較的平面成果圖。
2. 保留岩心箱內的有效範圍，避免裁掉岩心。
3. 依孔號中的終深（例如 `H25-1B(50m)`）自動處理最後一箱。
4. 產生接近既有成果 Word / PDF 版型的正式成果文件。

---

## 2. 專案檔案結構

目前 v2.11 主要結構如下：

```text
project/
├─ app.py
├─ core.py
├─ requirements.txt
└─ point_editor/
   └─ index.html
```

### `app.py`

Streamlit 的主程式與 UI 控制器。

負責：

- Streamlit 頁面設定
- 表頭資料輸入
- 圖片上傳
- 照片排序與目前照片選擇
- 每張照片的旋轉狀態
- 每張照片的角點／裁切狀態
- 自動偵測呼叫
- 拖曳式四角點編輯元件呼叫
- 校正後成果預覽
- 最後深度的有效槽數計算
- PDF / Word 輸出按鈕與下載
- Web UI CSS 美編

### `core.py`

影像處理與輸出核心模組。

負責：

- 圖片前處理
- 點位排序
- 藍色岩心箱偵測
- 箱框幾何估計
- 四角點偵測
- 透視校正
- 最後一箱槽數判定
- 半箱裁切
- 成果圖編號
- 表頭圖片製作
- PDF / Word 生成

### `point_editor/index.html`

自訂 Streamlit Component 的前端畫布。

用途是讓使用者在原圖上直接拖曳四個黃色控制點。

它是純 HTML / JavaScript Canvas，不使用第三方 AI 或影像模型。

### `requirements.txt`

目前依賴：

```text
streamlit
opencv-python-headless
numpy
pillow
reportlab
python-docx
```

---

## 3. app.py 架構

### 3.1 程式版本

```python
APP_VERSION = "2.11"
DETECTOR_VERSION = getattr(core, "DETECTOR_VERSION", "unknown")
```

`APP_VERSION` 是 UI 程式版本。

`DETECTOR_VERSION` 用於影像偵測快取的版本隔離，避免演算法更新後 Streamlit cache 繼續使用舊偵測結果。

---

### 3.2 Streamlit Component

```python
_COMPONENT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "point_editor"
)
point_editor = components.declare_component(
    "point_editor",
    path=_COMPONENT_DIR
)
```

此 component 用 `point_editor/index.html` 實作拖曳角點。

Python 端以：

```python
point_editor(...)
```

傳入：

- base64 圖片
- 原始影像寬高
- 四個點
- 顯示寬高

前端拖曳完成後回傳 JSON 格式的四點座標。

---

## 4. app.py 的主要狀態管理

目前使用 `st.session_state` 儲存「照片級」狀態。

主要 key：

```text
manual
rot
crop_cfg
edit_nonce
crop_nonce
state_version
sel_photo
pdf
 docx
```

### `manual`

儲存每張照片的手動四角點。

key 是：

```python
(photo_name, rotation)
```

因此手動調整是**逐張獨立**，不是全域四角點。

### `rot`

儲存每張照片的額外旋轉角度：

```text
0 / 90 / 180 / 270
```

### `crop_cfg`

儲存每張照片自己的裁切／留邊設定：

```python
{
    "mgx": 0.0,
    "mgy": 0.0,
    "source_pad": 0.0,
}
```

用途：

- `mgx`：成果圖左右留邊比例
- `mgy`：成果圖上下留邊比例
- `source_pad`：來源四角向外微擴，避免切到岩心

**重要：這些值是逐張照片獨立保存。**

### `edit_nonce` / `crop_nonce`

用於強制 Streamlit widget 更新，解決「按重設但舊 widget state 不變」的問題。

---

## 5. 圖片進入系統的流程

### `load_image()`

位於 `core.py`。

```python
load_image(file_or_bytes, max_side=3200, portrait_dir="ccw")
```

步驟：

1. `PIL.Image.open()` 讀取。
2. `ImageOps.exif_transpose()` 修正 EXIF 方向。
3. 最大邊超過 `3200px` 時縮圖。
4. 若影像仍為直式，按 UI 指定方向旋轉 90°。
5. 回傳 contiguous NumPy RGB array。

目的：把不同手機／相機 EXIF 方向統一後再進入 OpenCV。

---

## 6. 照片排序

`natural_key()`：

```python
return [
    int(t) if t.isdigit() else t.lower()
    for t in re.split(r"(\d+)", s)
]
```

用途是自然排序，例如：

```text
H25-1B_1.jpg
H25-1B_2.jpg
H25-1B_10.jpg
```

而不是字典序的：

```text
1, 10, 2
```

---

## 7. 自動四角點偵測：核心概念

### 最重要的設計觀念

**偵測目標是「岩心箱框」，不是「岩心本身」。**

這點對本專案非常重要。

尤其最後一箱例如：

```text
H25-1B(50m)
```

只有 49、50 兩槽有資料時，不能因為畫面上只有兩槽岩心，就把「岩心範圍」當成「箱框範圍」。

正確概念應該是：

```text
先找完整岩心箱幾何
        ↓
四個角代表完整箱框
        ↓
做透視校正
        ↓
最後再依終深把 51、52 等超出深度的槽裁掉
```

目前 detector 的核心資料來源是岩心箱的**藍色箱體結構**。

---

## 8. `detect_inner()` 的整體策略

`core.py`：

```python
def detect_inner(img, full_aspect=3.1):
```

目前策略：

### 第一階段：藍色箱體幾何

```text
_blue_mask_strong()
        ↓
_hough_horizontal_groups()
        ↓
_fit_outer_and_inner_blue()
```

如果成功：

```text
完整箱體內框
        ↓
_finish_full_tray_quad()
        ↓
order_pts()
```

### 第二階段：Fallback

若直接藍色箱體幾何失敗：

```text
_fallback_inner_from_holes()
        ↓
_finish_full_tray_quad()
        ↓
幾何合理性檢查
```

最後若都失敗：

```python
return None, True
```

app.py 會把該照片顯示成：

```text
⚠ 需設定
```

而不是讓整個 Streamlit 頁面停止。

---

## 9. 藍色遮罩

### `_blue_mask()`

較早期的偵測路徑使用：

```python
HSV
H = 85~118
S = 80~255
V = 60~255
```

再搭配 morphology。

### `_blue_mask_strong()`

目前主要的箱體偵測使用較寬容範圍：

```python
H = 78~128
S = 55~255
V = 40~255
```

然後：

- morphological close
- morphological open

原因是岩心箱藍色在不同光線下可能偏亮、偏暗、反光或有泥土遮蔽，所以不能只用非常窄的 HSV 範圍。

---

## 10. Hough 橫線偵測

函式：

```python
_hough_horizontal_groups(mask)
```

目的不是直接找最終四角，而是先找：

> 岩心箱內一組規律、近似平行的藍色橫桿／槽線

利用：

```python
cv2.Canny()
cv2.HoughLinesP()
```

再依：

- 線段長度
- 水平角度
- y 位置
- 線段間距
- 間距變異
- 群組數量

將線段分組。

設計理由：

開啟的箱蓋通常只有少數幾條藍線；真正岩心槽區會形成多條規律橫線，因此可以利用群組結構排除部分箱蓋干擾。

### Cloud-safe Hough

此函式目前特別防禦 OpenCV 不同版本可能回傳不同 ndarray shape 的情況，會先 flatten 成 4 個座標。

如果 Hough 發生：

- `ValueError`
- `TypeError`
- `cv2.error`

不讓整個 Web App 崩潰，而改走 fallback。

這是 `DETECTOR_VERSION = "2.6-safe-hough-full-tray"` 的主要安全性之一。

---

## 11. 外框與內框的關係

### `_fit_outer_and_inner_blue()`

先找箱體外框的主要上下線，再根據藍色箱壁實際內緣找內框。

### `_inner_quad_from_blue()`

這一步的設計目的非常重要：

`TRAY_INSET` **不是直接拿來算最終角點**。

它主要用作「搜尋位置的先驗範圍」。

程式仍要由藍色箱壁實際內緣的像素／線條估計真正內角。

---

## 12. 角點順序

`order_pts()` 統一將 4 點整理成：

```text
左上、右上、右下、左下
```

因此使用者在拖曳時：

- 不要求操作順序
- 只要有四個角點
- 校正時由程式重新排序

這個排序函式是所有透視轉換的共同入口。

---

## 13. 自動偵測不把「歪斜」本身判成異常

目前 `detect_inner()` 的設計是：

> 歪斜本身不是失敗條件。

只要箱體幾何線索足夠，就返回 `bad=False`。

這是因為實際岩心現場照片經常存在：

- 相機傾斜
- 箱體透視變形
- 左右高度不一致
- 上下邊不平行

本工具本來就是要處理這些情況，所以不應把「有歪斜」直接標為「異常」。

`bad=True` 主要表示：

- 自動幾何不足
- 四角結果不可靠
- fallback 仍無法取得合理箱框

---

## 14. 手動拖曳角點

### Python 端

`_draggable_points()`：

```python
point_editor(
    image=...,
    points=...,
    image_width=...,
    image_height=...,
    display_width=...,
    display_height=...
)
```

目前顯示寬度會跟可用空間與原圖尺寸控制，不再固定為必然超出欄位的寬度。

### 前端 `point_editor/index.html`

使用 HTML Canvas：

```text
原圖
 + 紅色四邊框
 + 黃色四個控制點
```

使用 `pointerdown / pointermove / pointerup` 做拖曳。

拖曳時：

```text
畫面座標
   ↓
除以顯示縮放比例
   ↓
還原為原始圖片座標
   ↓
限制在 0 ~ image_width-1 / 0 ~ image_height-1
   ↓
回傳 4 個點
```

### 互動結果

Python 收到回傳後若變化大於 0.5 px：

```python
st.session_state["manual"][ck] = edited.astype(np.float32)
st.rerun()
```

因此右側成果會立即重新計算。

---

## 15. 自動角點與手動角點的優先順序

`get_pts()`：

```text
若該照片有 manual
    ↓
直接使用手動角點

否則
    ↓
使用自動偵測

自動失敗
    ↓
回傳 None + bad=True
```

因此：

> **手動角點是最高優先權。**

這也是現階段處理難例照片最可靠的方式。

---

## 16. 透視校正

函式：

```python
warp_points(img, pts, out_w=2400, aspect=3.12, margin=MARGIN, source_pad=0.004)
```

流程：

```text
輸入原圖
   ↓
四個來源角點
   ↓
source_pad 向外微擴
   ↓
目標長方形
   ↓
cv2.getPerspectiveTransform()
   ↓
cv2.warpPerspective()
```

### `source_pad`

作用是避免角點剛好壓在岩心邊緣時，把部分岩心切掉。

### `margin`

目標畫布內的留邊比例：

```text
mgx → 左右
mgy → 上下
```

這些設定在 v2.11 是「每張照片各自保存」。

---

## 17. 成果裁切不是單純矩形裁切

這是後續 AI 修改時非常重要的認知。

目前成果圖主要由：

```text
四角透視校正
```

建立。

因此：

> 如果四角點本身錯誤，輸出看起來可能會有「被拉伸」現象。

這不是單純 crop 的問題，而是透視變換的來源四邊定義錯誤。

所以除非需求明確是要改「最後半箱裁切」，否則不要只修改 `crop_partial()` 來解決透視失真；應先檢查 `detect_inner()` / manual points / `warp_points()`。

---

## 18. 最後一箱與終深邏輯

### 固定槽數

目前專案的固定規則是：

```text
1 槽 = 1m
1 箱 = 4 槽 = 4m
```

### `_parse_end_depth()`

從孔號，例如：

```text
H25-1B(50m)
```

解析：

```text
終深 = 50m
```

### `_depth_total_rows()`

由：

```text
終深 - 起始深度
```

除以每槽 1m 得到有效槽數。

目前起始深度在 UI 中固定為 `0`。

### `output_files_effective()`

根據總槽數反推出真正需要的圖片數。

例如：

```text
50m
→ 50 槽
→ 4 槽 / 箱
→ ceil(50/4) = 13 箱照片
```

所以如果使用者上傳 13 張以上照片，超出終深所需數量的照片會被排除。

---

## 19. 最後半箱 `crop_partial()`

例如：

```text
49m
50m
51m ← 不存在於 50m 孔
52m ← 不存在
```

最後一箱原始影像仍應先代表完整箱體，但輸出時只保留前兩槽。

`crop_partial(im, k, n=4, margin=MARGIN)`：

```text
k = 1 → 1 槽高度
k = 2 → 2 槽高度
k = 3 → 3 槽高度
k = 4 → 完整箱
```

**重要：`crop_partial()` 不應把 2 槽重新拉伸成 4 槽。**

它的目的是「刪除後面不存在／超出終深的空槽」。

---

## 20. 最後一箱無法從孔號判讀終深時

若孔號沒有 `(50m)` 這種格式：

```text
effective_last_rows()
```

會嘗試對最後一張校正後影像使用：

```python
detect_occupied_rows()
```

判斷實際有幾槽。

### `detect_occupied_rows()`

主要分析每槽中央區域：

- 飽和度
- 明度
- 灰色／低飽和程度
- 暗部比例
- Canny edge
- texture / 灰階標準差

再將每槽轉成 score。

前兩槽當作有岩心的基準，後續連續槽若分數明顯降低，視為空槽。

這是一個傳統影像 heuristic，不是 AI 分類器。

---

## 21. UI 狀態標籤

目前照片狀態：

```text
✓ 自動OK
⚠ 需檢查
✋ 手動
⚠ 需設定
```

含義：

### `✓ 自動OK`

自動偵測成功，幾何上沒有觸發目前的 warning 條件。

**注意：不是人工品質保證。**

### `⚠ 需檢查`

自動偵測有結果，但程式判定可靠度不足。

### `✋ 手動`

該照片已有手動角點。

### `⚠ 需設定`

沒有有效四角點，需要使用者拖曳修正。

---

## 22. UI 的人工檢查原則

目前 v2.11 並沒有強制使用者逐張點「我已確認」。

而是在頁面上提供明顯提醒：

> 自動抓角點只供初判。產生成果前，請先檢查每張照片左側箱框與右側「校正後（成果用）」預覽。

這是目前的工作定位：

```text
OpenCV 自動初判
        ↓
使用者視覺檢查
        ↓
必要時拖曳修正
        ↓
正式輸出
```

如果未來需要「所有照片都必須按確認才能輸出」，應新增獨立的 `confirmed` state，而不要用 `bad` 代替人工確認。

---

## 23. 照片切換 UI

三個控制項：

```text
上一張 | 下一張 | 選擇照片
```

使用：

```python
st.columns(3)
```

並用 CSS 統一：

- 高度
- 字體
- 邊框
- 白底
- 陰影

`sel_photo` 是目前照片名稱。

上一張／下一張只是改變 `sel_photo`；不直接改影像資料。

---

## 24. 每張照片的裁切設定

`_photo_cfg(name, rot)` 建立：

```python
{
    "mgx": 0.0,
    "mgy": 0.0,
    "source_pad": 0.0,
}
```

### 目前 UI 預設

```text
左右留邊 = 0%
上下留邊 = 0%
角點向外留邊 = 0%
```

當使用者調整時，只影響目前所選照片。

「重設本張裁切留邊」會：

```text
pop(current_photo_cfg)
        ↓
增加 crop_nonce
        ↓
rerun
        ↓
回到 0 / 0 / 0
```

---

## 25. 「恢復本張自動角點」

按鈕執行：

```python
st.session_state["manual"].pop(ck, None)
_bump_nonce("edit_nonce", ck)
st.rerun()
```

因此它應該：

```text
刪除本張手動角點
↓
下一輪重新取得自動角點
↓
畫面重新建立 point editor
```

如果修改此功能，不應只改 `manual.pop()`；還要考慮 component key / nonce，否則瀏覽器端可能保留舊 state。

---

## 26. PDF 輸出

`build_pdf()` 使用：

- ReportLab
- A4
- 固定邊界
- 成果表頭
- 每箱一列
- 右側箱號
- 最後鑽探結束文字

主要版面常數：

```python
PAGE_MX = 2.0
PAGE_MY = 1.0
HEADER_W = 15.4
HEADER_H = 3.17
BOX_W = 15.15
BOX_H = 4.85
NUM_COL_W = 1.0
FULL_ASPECT = 3.12
```

### 影像高度

目前用：

```python
_image_height_cm(im)
```

按實際圖片比例計算高度。

因此半箱圖片不應強制塞回完整箱高。

---

## 27. Word 輸出

`build_docx()` 使用：

- `python-docx`
- A4
- 固定表格欄寬
- 表頭圖片
- 岩心成果圖
- 右側箱號

重點是：

```text
圖片寬度固定
高度依影像比例
```

目前刻意避免使用不同寬高縮放同一張圖片，以降低 Word 內再次非等比例拉伸的風險。

---

## 28. 表頭製作

`make_header()`：

- 使用上傳的告示牌照片，或程式內嵌的 `BOARD_B64`
- 對工程名稱、孔號、日期、深度等欄位覆寫文字
- 工程名稱會自動縮字以適應欄位

預設使用者輸入值目前是：

```text
工程名稱：空白
孔號：空白
日期：空白
```

日期欄位由 Streamlit `date_input` 選取，再轉成：

```text
YYYY.MM.DD
```

---

## 29. `box_image()` 完整資料流

這個函式可以視為「單張成果圖」的核心入口：

```text
get_img(f)
   ↓
取得每張照片自己的 crop_cfg
   ↓
get_pts(f)
   ↓
取得 manual 或 automatic 四角
   ↓
_warp_compat()
   ↓
warp_points()
   ↓
Image.fromarray()
   ↓
如果是最後一個有效輸出照片
   ↓
effective_last_rows()
   ↓
crop_partial()
   ↓
回傳成果圖
```

因此任何影像品質問題，優先追查：

```text
get_pts()
→ detect_inner()
→ warp_points()
→ crop_partial()
```

而不是先修改 PDF / Word。

---

## 30. `output_files_effective()` 與「50m 後不輸出」

這是最後深度控制的主要入口。

假設：

```text
孔號：H25-1B(50m)
每槽：1m
每箱：4槽
```

則：

```text
總有效槽數 = 50
需要圖片數 = ceil(50 / 4) = 13
```

第 13 張就是：

```text
49
50
```

下方的：

```text
51
52
```

不應出現在正式成果中。

---

## 31. 目前 AI 狀態

### 沒有 AI API

目前沒有：

- OpenAI API
- Gemini API
- Anthropic API
- Azure AI Vision
- Claude Vision
- YOLO
- SAM
- GroundingDINO
- 雲端 AI 圖片分析

### 有的是 Computer Vision

主要使用：

- HSV segmentation
- morphology
- contour
- convex hull
- Hough Lines
- robust line fitting
- geometric intersection
- perspective transform
- heuristic texture / edge scoring

所以若其他 AI 要理解本專案，應稱為：

> **OpenCV-based computer vision image rectification tool**

而不是：

> AI vision detector

---

## 32. 已知限制與容易出錯的位置

### 限制 A：角點偵測依賴藍色箱體

如果：

- 箱體藍色被泥土嚴重遮蔽
- 強烈反光
- 光線造成藍色偏移
- 箱體部分不在照片中

自動偵測可能失敗。

### 限制 B：最後一箱若孔號沒有終深

會改用 heuristic `detect_occupied_rows()`。

這可能誤判，因此孔號最好保有 `(50m)` 這種終深資訊。

### 限制 C：照片非常歪時

目前有幾何與 Hough fallback，但仍可能需要手動拖曳。

### 限制 D：岩心不是箱框

若未來再修改 detector，不能讓「岩心的灰色／褐色區域」直接替代完整箱框幾何，否則會再次出現 49～50 只框到有岩心兩槽的問題。

### 限制 E：透視變換會放大角點錯誤

四角點錯一點，經 `warpPerspective` 後可能變成明顯拉伸，因此「自動四角點品質」比後端 PDF/Word 排版更重要。

---

## 33. 目前最重要的資料模型

可將每張照片視為：

```python
PhotoState = {
    "name": str,
    "rotation": 0 | 90 | 180 | 270,
    "points": np.ndarray(shape=(4,2)),
    "point_source": "auto" | "manual" | "none",
    "crop": {
        "mgx": float,
        "mgy": float,
        "source_pad": float,
    },
}
```

這不是目前程式裡正式的 dataclass，而是給其他 AI 理解資料關係的**概念模型**。

---

## 34. 建議其他 AI 修改本專案時的原則

### 修改四角點偵測

優先修改：

```text
core.py
  ├─ _blue_mask_strong
  ├─ _hough_horizontal_groups
  ├─ _fit_outer_and_inner_blue
  ├─ _inner_quad_from_blue
  ├─ _fallback_inner_from_holes
  └─ detect_inner
```

### 修改使用者拖曳

修改：

```text
app.py
point_editor/index.html
```

### 修改逐張留邊

修改：

```text
app.py
_photo_cfg()
_warp_compat()
box_image()
```

### 修改最後一箱

先檢查：

```text
_parse_end_depth()
_depth_total_rows()
output_files_effective()
effective_last_rows()
box_image()
crop_partial()
```

### 修改 Word / PDF

最後才進入：

```text
build_pdf()
build_docx()
```

因為輸出端通常只是呈現上游已決定的成果圖。

---

## 35. 不要直接改全域比例來修單張照片

目前 `crop_cfg` 是逐張的，因此如果某一張 25～28 照片歪斜：

**不建議**直接把：

```text
TRAY_INSET
FULL_ASPECT
MARGIN
```

改成專門適應這張照片。

正確方式應優先是：

```text
自動角點
↓
檢查
↓
該照片手動拖曳四角
↓
該照片 crop_cfg 微調
```

這樣不會破壞其他正常照片。

---

## 36. Streamlit Cache 注意事項

目前：

```python
@st.cache_data
```

用於圖片讀取與自動偵測。

`_detect()` 特別帶入：

```python
detector_version
```

這是為了讓 detector 版本變更時，cache key 也跟著改變。

如果修改了偵測演算法但忘記增加 detector version，可能出現：

> 程式碼已經改了，但畫面看起來還在用舊角點。

這是未來除錯時非常重要的檢查點。

---

## 37. 部署方式

### 本機

安裝：

```bash
pip install -r requirements.txt
```

啟動：

```bash
streamlit run app.py
```

### Streamlit Cloud

GitHub repository 至少需要：

```text
app.py
core.py
requirements.txt
point_editor/index.html
```

不需要提交：

```text
__pycache__/
*.pyc
```

### 部署注意

`point_editor/index.html` 是自訂 Streamlit component 的必要檔案，不能漏掉。

---

## 38. 跨 AI 交接時應先讀哪些檔案

如果另一個 AI 要接手本專案，建議順序：

### 第一優先

```text
README_跨AI程式結構說明_v2.11.md
app.py
core.py
```

### 第二優先

```text
point_editor/index.html
requirements.txt
```

### 第三優先

再看測試照片、輸出 PDF、debug 圖。

原因是測試圖是案例資料；`app.py/core.py` 才是系統真實邏輯。

---

## 39. 建議 AI 接手後的第一輪檢查

不要先大改程式，先建立以下測試矩陣：

| 測試案例 | 期待結果 |
|---|---|
| 正常水平箱 | 自動四角正確、成果無拉伸 |
| 轻微歪斜 | 自動校正，不應直接判異常 |
| 21～28 類型歪斜 | 優先確認完整箱框，不要抓成岩心框 |
| 49～50 半箱 | 四角應是完整箱框；最後只保留 2 槽 |
| 1m 最後箱 | 只保留 1 槽 |
| 2m 最後箱 | 只保留 2 槽 |
| 3m 最後箱 | 只保留 3 槽 |
| 4m | 完整 4 槽 |
| 自動偵測失敗 | UI 不應崩潰，可手動拖曳 |
| 手動角點 | 應覆蓋自動角點 |
| 重設自動角點 | 應回到自動結果 |
| 重設本張裁切留邊 | 應只重設當前照片 |
| Word | 可開啟且圖片不應再次非等比例拉伸 |
| PDF | 可正常產生並符合終深 |

---

## 40. 未來若加入 AI 視覺模型

如果未來要真正加入 AI，建議不要直接把 AI 塞進 `detect_inner()`。

建議新增：

```text
ai_detector.py
```

架構改成：

```text
原圖
  ↓
OpenCV 初始候選框
  ↓
AI Vision / Detection Model
  ↓
完整岩心箱框 4 點
  ↓
幾何 sanity check
  ↓
人工預覽
  ↓
warp_points()
  ↓
PDF / Word
```

AI 的輸入輸出應該定義清楚，例如：

```python
{
    "bbox": [x1, y1, x2, y2],
    "corners": [[x,y], [x,y], [x,y], [x,y]],
    "confidence": 0.97,
    "source": "ai",
}
```

而不是讓 AI 直接產生最終 JPG。

這樣可保留：

- 可解釋性
- 手動修正
- OpenCV fallback
- 幾何檢查
- 不同 AI provider 的可替換性

---

## 41. 最重要的跨 AI 結論

本專案可以用一句話描述：

> **這是一個以 Streamlit 為 UI、OpenCV 為自動箱框偵測核心、Canvas 拖曳點為人工修正工具、透視變換為照片校正核心、ReportLab/python-docx 為成果輸出後端的岩心箱照片整理工具。**

### 最重要的資料流

```text
Input Photo
  ↓
load_image()
  ↓
rotate_extra()
  ↓
get_pts()
  ├─ manual points
  └─ detect_inner()
       ↓
       blue tray geometry
       ↓
       Hough / line fitting / fallback
  ↓
warp_points()
  ↓
per-photo crop configuration
  ↓
last-tray depth handling
  ↓
box_image()
  ↓
build_pdf() / build_docx()
```

### 最重要的品質原則

```text
不要把岩心本身當成岩心箱框。
不要用全域裁切參數修正單張照片。
不要把歪斜本身當成錯誤。
不要讓單張偵測失敗造成整個 Streamlit 頁面崩潰。
最後一箱應先定位完整箱框，再依終深裁掉不存在的槽。
手動四角點應永遠能覆蓋自動結果。
```

---

## 42. 開發者快速索引

| 需求 | 主要檔案 | 函式 |
|---|---|---|
| 上傳照片 | `app.py` | uploader / sorting |
| EXIF / 轉橫 | `core.py` | `load_image()` |
| 單張旋轉 | `app.py` / `core.py` | `cur_rot()`, `rotate_extra()` |
| 自動箱框 | `core.py` | `detect_inner()` |
| 藍色箱體 | `core.py` | `_blue_mask_strong()` |
| Hough 線段 | `core.py` | `_hough_horizontal_groups()` |
| 內框 | `core.py` | `_inner_quad_from_blue()` |
| fallback | `core.py` | `_fallback_inner_from_holes()` |
| 四點排序 | `core.py` | `order_pts()` |
| 手動拖點 | `app.py` + `point_editor/index.html` | `_draggable_points()` |
| 透視校正 | `core.py` | `warp_points()` |
| 單張留邊 | `app.py` | `_photo_cfg()` |
| 最後槽數 | `app.py` / `core.py` | `_depth_total_rows()`, `detect_occupied_rows()` |
| 半箱裁切 | `core.py` | `crop_partial()` |
| 單張成果圖 | `app.py` | `box_image()` |
| 表頭 | `core.py` | `make_header()` |
| PDF | `core.py` | `build_pdf()` |
| Word | `core.py` | `build_docx()` |
| UI CSS | `app.py` | `<style>` 區塊 |

---

## 43. 文件與程式一致性說明

本文件以目前可取得的 **v2.11 程式結構**為基準，定位是「跨 AI / 開發者交接文件」，不是一般使用者操作手冊。

若後續修改：

- `APP_VERSION`
- `DETECTOR_VERSION`
- 檔案結構
- 影像偵測流程
- 最後深度規則
- Word / PDF 版型
- 是否加入 AI API

應同步更新本文件，避免其他 AI 根據舊文件誤判程式行為。
