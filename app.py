import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from core import (natural_key, load_image, detect_box, warp, split_rows,
                  draw_corners, build_pdf)

st.set_page_config(page_title="岩心照片校正與成果 PDF", layout="wide")
st.title("岩心箱照片：透視校正 → 切列 → 套疊成果 PDF")

# ---------------- 側邊欄參數 ----------------
with st.sidebar:
    st.header("成果表頭")
    company = st.text_input("公司", "青山工程顧問有限公司")
    project = st.text_input("工程名稱", "114 年度鵠鵠崙地區潛在大規模崩塌調查監測計畫")
    hole = st.text_input("孔號", "H25-1B(50m)")
    date = st.text_input("日期", "114.3.29")
    start_depth = st.number_input("起始深度 (m)", 0, 500, 0)

    st.header("切列設定")
    rows_per_box = st.number_input("每箱列數（照片1 = 4 列）", 1, 10, 4)
    row_m = st.number_input("每列代表深度 (m)", 1, 5, 1)
    per_page = st.number_input("每頁箱數", 5, 30, 20)
    aspect = st.number_input("校正後整箱 寬/高（0 = 由角點自動估算）", 0.0, 10.0, 2.87, 0.01)
    st.caption("內側裁切（%）：去掉藍色箱緣")
    c1, c2 = st.columns(2)
    left = c1.slider("左", 0, 15, 3) / 100
    right = c2.slider("右", 0, 15, 4) / 100
    top = c1.slider("上", 0, 25, 9) / 100
    bottom = c2.slider("下", 0, 25, 8) / 100
    skip_warp = st.checkbox("照片已經是正的，不做透視校正", False)

files = st.file_uploader("依深度由淺到深上傳岩心箱照片（可多選，依檔名自然排序）",
                         type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if not files:
    st.info("請先上傳照片。每張照片為一個岩心箱（4 列 = 4 個箱號）。")
    st.stop()

files = sorted(files, key=lambda f: natural_key(f.name))
st.session_state.setdefault("manual", {})
st.session_state.setdefault("clicks", {})


@st.cache_data(show_spinner=False)
def _load(name, data):
    import io
    return load_image(io.BytesIO(data))


imgs = {f.name: _load(f.name, f.getvalue()) for f in files}


def get_pts(name):
    img = imgs[name]
    if name in st.session_state["manual"]:
        return st.session_state["manual"][name]
    p = detect_box(img)
    if p is None:
        h, w = img.shape[:2]
        p = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    return p


def rectify(name):
    img = imgs[name]
    if skip_warp:
        return img
    return warp(img, get_pts(name), aspect or None)


# ---------------- 逐張檢查 / 手動點角 ----------------
st.subheader("1. 檢查校正結果")
sel = st.selectbox("選擇照片", [f.name for f in files])
colA, colB = st.columns(2)
img = imgs[sel]
with colA:
    st.caption("原圖與偵測到的四角（紅框）")
    st.image(draw_corners(img, get_pts(sel)), use_container_width=True)
    with st.expander("自動偵測不準？手動點選四個角"):
        st.caption("依序點：左上 → 右上 → 右下 → 左下（箱子外緣）。")
        scale = 900 / img.shape[1]
        small = Image.fromarray(img).resize((900, int(img.shape[0] * scale)))
        clicks = st.session_state["clicks"].setdefault(sel, [])
        disp = np.array(small)
        for q in clicks:
            disp[max(0, int(q[1]) - 6):int(q[1]) + 6, max(0, int(q[0]) - 6):int(q[0]) + 6] = (255, 0, 0)
        v = streamlit_image_coordinates(Image.fromarray(disp), key=f"clk_{sel}_{len(clicks)}")
        if v and len(clicks) < 4 and (v["x"], v["y"]) not in [tuple(c) for c in clicks]:
            clicks.append((v["x"], v["y"]))
            st.rerun()
        st.write(f"已點 {len(clicks)}/4")
        b1, b2 = st.columns(2)
        if b1.button("套用手動角點", disabled=len(clicks) != 4):
            st.session_state["manual"][sel] = np.array(clicks, np.float32) / scale
            st.rerun()
        if b2.button("清除重點"):
            st.session_state["clicks"][sel] = []
            st.session_state["manual"].pop(sel, None)
            st.rerun()
with colB:
    st.caption("校正後（已拉正）")
    st.image(rectify(sel), use_container_width=True)

rows_prev = split_rows(rectify(sel), rows_per_box, left, right, top, bottom)
st.caption("切列預覽")
for r in rows_prev:
    st.image(r, use_container_width=True)

# ---------------- 產生 PDF ----------------
st.subheader("2. 套疊成果 PDF")
st.write(f"共 {len(files)} 張照片 → 預計 {len(files) * rows_per_box} 個箱號。")
if st.button("產生 PDF", type="primary"):
    all_rows = []
    bar = st.progress(0.0)
    for i, f in enumerate(files):
        all_rows += split_rows(rectify(f.name), rows_per_box, left, right, top, bottom)
        bar.progress((i + 1) / len(files))
    pdf = build_pdf(all_rows, company, project, hole, date, start_depth, per_page, row_m)
    st.session_state["pdf"] = pdf
    st.success(f"完成：{len(all_rows)} 個箱號")

if "pdf" in st.session_state:
    st.download_button("下載成果 PDF", st.session_state["pdf"],
                       file_name=f"{hole}_岩心照片.pdf", mime="application/pdf")
