import io

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from core import (natural_key, load_image, rotate_extra, detect_inner, warp_points,
                  crop_partial, draw_corners, build_pdf, build_docx, auto_detect_filled_rows)

st.set_page_config(page_title="岩心照片校正與成果輸出", layout="wide")
st.title("岩心箱照片：轉橫 → 以箱內四個內角校正 → 套疊成果（Word / PDF）")

# ---------------- 側邊欄 ----------------
with st.sidebar:
    st.header("成果表頭")
    project = st.text_input("工程名稱", "114 年度鵠鵠崙地區潛在大規模崩塌調查監測計畫")
    hole = st.text_input("孔號", "H25-1B(50m)")
    date = st.text_input("日期", "114.3.29")
    start_depth = st.number_input("起始深度 (m)", 0, 500, 0)
    board_up = st.file_uploader("自訂表頭告示牌照片（選填，預設用範例 Word 的）", type=["jpg", "jpeg", "png"])

    st.header("轉向 / 校正")
    portrait_dir = st.radio("直式照片（90°拍攝）轉橫的方向", ["逆時針", "順時針"], horizontal=True)
    st.caption("轉橫後標籤 H25-1B 應在箱子上緣右側、箱號 1–4 在右邊且由上往下。若相反，在下方用「額外旋轉」單張修正。")
    global_rot = st.selectbox("全部照片額外旋轉（逆時針）", [0, 90, 180, 270], index=0)
    aspect = st.number_input("成果圖整箱 寬/高（範例 Word ≈ 3.12）", 1.5, 5.0, 3.12, 0.01)

    st.header("進階微調與自動化")
    inset_adj = st.slider("紅框向外微調 %", -5.0, 5.0, 1.5, 0.5) / 100.0
    st.caption("預設向外擴 1.5%，若紅框切到岩心請調大數值；若抓到太多藍色邊框請調小。")
    auto_last = st.checkbox("自動偵測最後一箱實際列數", True)
    
    st.header("成果裁切")
    st.caption("成果圖 = 箱內四個內角拉成長方形，外側只多留一點點邊（同範例 Word 的裁法）。")
    mgx = st.slider("左右多留 %", 0.0, 3.0, 0.4, 0.1) / 100
    mgy = st.slider("上下多留 %", 0.0, 3.0, 0.6, 0.1) / 100
    rows_per_box = st.number_input("每箱列數", 1, 10, 4)
    row_m = st.number_input("每列代表深度 (m)", 1, 5, 1)
    per_page = st.number_input("每頁箱號數", 4, 40, 20, 4)
    
    if not auto_last:
        last_rows = st.number_input("最後一箱實際有岩心的列數（0 = 滿箱）", 0, 10, 0,
                                    help="例：只鑽到 50m，最後一箱只有 49、50 兩列 → 填 2。")
    else:
        last_rows = 0
        
    end_mark = st.checkbox("最後加「鑽探結束」", True)
    skip_warp = st.checkbox("照片已是正的，不做透視校正", False)

files = st.file_uploader("依深度由淺到深上傳岩心箱照片（可多選，依檔名自然排序）",
                         type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if not files:
    st.info("請先上傳照片。每張照片為一個岩心箱（4 列 = 4 個箱號）。")
    st.stop()
files = sorted(files, key=lambda f: natural_key(f.name))
for k in ("manual", "clicks", "rot"):
    st.session_state.setdefault(k, {})
pdir = "ccw" if portrait_dir == "逆時針" else "cw"


@st.cache_data(show_spinner=False, max_entries=60)
def _load(name, data, pdir):
    return load_image(io.BytesIO(data), portrait_dir=pdir)


@st.cache_data(show_spinner=False, max_entries=120)
def _detect(name, data, pdir, rot, inset_adj):
    img = rotate_extra(_load(name, data, pdir), rot)
    return detect_inner(img, inset_adj=inset_adj)


def cur_rot(name):
    return st.session_state["rot"].get(name, global_rot)


def get_img(f):
    return rotate_extra(_load(f.name, f.getvalue(), pdir), cur_rot(f.name))


def get_pts(f):
    """回傳 (四角點, 是否貼邊/未偵測到, 是否手動)"""
    key = (f.name, cur_rot(f.name))
    if key in st.session_state["manual"]:
        return st.session_state["manual"][key], False, True
    p, touch = _detect(f.name, f.getvalue(), pdir, cur_rot(f.name), inset_adj)
    if p is None:
        img = get_img(f)
        h, w = img.shape[:2]
        return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32), True, False
    return p, touch, False


def box_image(f):
    img = get_img(f)
    if not skip_warp:
        pts, _, _ = get_pts(f)
        img = warp_points(img, pts, aspect=aspect, margin=(mgx, mgy))
    im = Image.fromarray(img)
    
    # 處理最後一箱
    if f.name == files[-1].name:       
        if auto_last:
            filled = auto_detect_filled_rows(img, rows_per_box)
            im = crop_partial(im, filled, rows_per_box, (mgx, mgy))
            st.session_state["detected_last_rows"] = filled
        elif last_rows:
            im = crop_partial(im, last_rows, rows_per_box, (mgx, mgy))
            st.session_state["detected_last_rows"] = last_rows
            
    return im


# ---------------- 1. 逐張檢查 ----------------
st.subheader("1. 檢查每張照片的校正結果")
status = {}
for f in files:
    _, bad, man = get_pts(f)
    status[f.name] = "手動" if man else ("⚠ 需確認" if bad else "自動OK")
st.caption("狀態：" + " ".join(f"{n}：{s}" for n, s in status.items()))

names = [f.name for f in files]
if st.session_state.get("sel_photo") not in names:
    st.session_state["sel_photo"] = names[0]


def _step(d):
    k = names.index(st.session_state["sel_photo"]) + d
    st.session_state["sel_photo"] = names[max(0, min(len(names) - 1, k))]


nb1, nb2, nb3 = st.columns([1, 1, 6])
nb1.button("◀ 上一張", on_click=_step, args=(-1,))
nb2.button("下一張 ▶", on_click=_step, args=(1,))
sel_name = nb3.selectbox("選擇照片", names, key="sel_photo")
st.caption(f"目前這張：{status[sel_name]}")
sel = next(f for f in files if f.name == sel_name)

r1, r2 = st.columns([1, 3])
rot_val = r1.selectbox("此張額外旋轉（逆時針）", [0, 90, 180, 270],
                       index=[0, 90, 180, 270].index(cur_rot(sel.name)), key=f"rot_{sel.name}")
if rot_val != cur_rot(sel.name):
    st.session_state["rot"][sel.name] = rot_val
    st.rerun()

img = get_img(sel)
pts, bad, man = get_pts(sel)
if bad and not man:
    r2.warning("自動偵測的四個內角不可靠（找不到岩心槽或貼近照片邊緣），請檢查紅框；不準就用下方手動點四個內角。")

colA, colB = st.columns(2)
with colA:
    st.caption("已轉橫的原圖與箱內四個內角（紅框，受左側微調連動）")
    st.image(draw_corners(img, pts), use_container_width=True)
with colB:
    st.caption("校正後（成果用）")
    b_img = box_image(sel)
    st.image(b_img, use_container_width=True)
    if sel.name == files[-1].name and auto_last:
        detected = st.session_state.get("detected_last_rows", rows_per_box)
        st.info(f"自動偵測最後一箱包含岩心列數為： {detected} 列")

with st.expander("角點不準？手動點選箱內四個內角"):
    st.caption("依序點：左上 → 右上 → 右下 → 左下（箱子內側的四個角，藍色箱緣內緣）。")
    scale = 900 / img.shape[1]
    small = np.array(Image.fromarray(img).resize((900, int(img.shape[0] * scale))))
    ck = (sel.name, cur_rot(sel.name))
    clicks = st.session_state["clicks"].setdefault(ck, [])
    for q in clicks:
        small[max(0, int(q[1]) - 7):int(q[1]) + 7, max(0, int(q[0]) - 7):int(q[0]) + 7] = (255, 0, 0)
    v = streamlit_image_coordinates(Image.fromarray(small), key=f"clk_{sel.name}_{ck[1]}_{len(clicks)}")
    if v and len(clicks) < 4 and (v["x"], v["y"]) not in [tuple(c) for c in clicks]:
        clicks.append((v["x"], v["y"]))
        st.rerun()
    st.write(f"已點 {len(clicks)}/4")
    b1, b2 = st.columns(2)
    if b1.button("套用手動角點", disabled=len(clicks) != 4):
        st.session_state["manual"][ck] = np.array(clicks, np.float32) / scale
        st.rerun()
    if b2.button("清除，回到自動"):
        st.session_state["clicks"][ck] = []
        st.session_state["manual"].pop(ck, None)
        st.rerun()

# ---------------- 2. 輸出 ----------------
st.subheader("2. 輸出成果（版面同範例 Word）")
actual_last_rows = st.session_state.get("detected_last_rows", last_rows) if auto_last else last_rows
total_calc_rows = ((len(files) - 1) * rows_per_box + actual_last_rows) if (actual_last_rows and actual_last_rows < rows_per_box) else len(files) * rows_per_box

st.write(f"共 {len(files)} 張照片 → {total_calc_rows} 個箱號，每頁 {per_page} 個箱號。")

if st.button("產生 Word / PDF", type="primary"):
    boxes, bar = [], st.progress(0.0, "處理照片中…")
    for i, f in enumerate(files):
        boxes.append(box_image(f))
        bar.progress((i + 1) / len(files))
    kw = dict(hole=hole, date=date, project=project, start_depth=start_depth,
              rows_per_box=rows_per_box, per_page=per_page, row_m=row_m,
              board=board_up if board_up else None, end_mark=end_mark,
              total_rows=total_calc_rows)
    st.session_state["pdf"] = build_pdf(boxes, **kw)
    st.session_state["docx"] = build_docx(boxes, **kw)
    bar.empty()
    st.success("完成")

if "pdf" in st.session_state:
    d1, d2 = st.columns(2)
    d1.download_button("下載 PDF", st.session_state["pdf"], file_name=f"{hole}_岩心照片.pdf",
                       mime="application/pdf")
    d2.download_button("下載 Word", st.session_state["docx"], file_name=f"{hole}_岩心照片.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
