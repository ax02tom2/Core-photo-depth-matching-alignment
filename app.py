import base64
import io
import inspect
import json
import os
import re

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components

import core

APP_VERSION = "2.5"
DETECTOR_VERSION = getattr(core, "DETECTOR_VERSION", "unknown")

natural_key = core.natural_key
load_image = core.load_image
rotate_extra = core.rotate_extra
detect_inner = core.detect_inner
warp_points = core.warp_points
crop_partial = core.crop_partial
draw_corners = core.draw_corners
build_pdf = core.build_pdf
build_docx = core.build_docx

_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "point_editor")
point_editor = components.declare_component("point_editor", path=_COMPONENT_DIR)

st.set_page_config(page_title="岩心照片校正與成果輸出", layout="wide")
st.title("岩心箱照片：自動抓角點 → 拖曳微調 → 成果輸出")


def _parse_end_depth(text):
    m = re.search(r"[\(（]\s*(\d+(?:\.\d+)?)\s*m\s*[\)）]", str(text), re.I)
    return float(m.group(1)) if m else None


# ---------------- 側邊欄 ----------------
with st.sidebar:
    st.header("成果表頭")
    project = st.text_input("工程名稱", "114 年度鵠鵠崙地區潛在大規模崩塌調查監測計畫")
    hole = st.text_input("孔號", "H25-1B(50m)")
    date = st.text_input("日期", "114.3.29")
    start_depth = st.number_input("起始深度 (m)", 0, 500, 0)
    board_up = st.file_uploader(
        "自訂表頭告示牌照片（選填）", type=["jpg", "jpeg", "png"]
    )

    st.header("轉向 / 校正")
    portrait_dir = st.radio(
        "直式照片（90°拍攝）轉橫的方向",
        ["逆時針", "順時針"],
        horizontal=True,
    )
    global_rot = st.selectbox("全部照片額外旋轉（逆時針）", [0, 90, 180, 270], index=0)
    aspect = st.number_input("成果圖整箱寬／高", 1.5, 5.0, 3.12, 0.01)

    st.header("最後深度")
    row_m = st.number_input("每槽代表深度 (m)", 0.1, 5.0, 1.0, 0.1)
    rows_per_box = st.number_input("每箱槽數", 1, 10, 4)
    partial_depth_m = st.number_input(
        "最後一箱保留深度 (m)（0 = 依孔號終深自動）",
        0.0,
        float(rows_per_box * row_m),
        0.0,
        0.1,
        help="每槽 1m 時：1m=1槽、2m=2槽。H25-1B(50m) 留白時會自動保留 49、50。",
    )
    per_page = st.number_input("每頁箱號數", 4, 40, 20, 4)
    end_mark = st.checkbox("最後加「鑽探結束」", True)
    skip_warp = st.checkbox("照片已是正的，不做透視校正", False)

files = st.file_uploader(
    "依深度由淺到深上傳岩心箱照片（可多選，依檔名自然排序）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)
if not files:
    st.info("請先上傳照片。每張照片為一個岩心箱。")
    st.stop()

files = sorted(files, key=lambda f: natural_key(f.name))
for k in ("manual", "rot", "crop_cfg"):
    st.session_state.setdefault(k, {})
st.session_state.setdefault("state_version", "2.5")
if st.session_state.get("state_version") != "2.5":
    st.session_state["manual"].clear()
    st.session_state["rot"].clear()
    st.session_state["crop_cfg"].clear()
    st.session_state["state_version"] = "2.5"

pdir = "ccw" if portrait_dir == "逆時針" else "cw"


@st.cache_data(show_spinner=False, max_entries=60)
def _load(name, data, pdir):
    return load_image(io.BytesIO(data), portrait_dir=pdir)


@st.cache_data(show_spinner=False, max_entries=120)
def _detect(name, data, pdir, rot, detector_version):
    img = rotate_extra(_load(name, data, pdir), rot)
    return detect_inner(img)


def cur_rot(name):
    return st.session_state["rot"].get(name, global_rot)


def get_img(f):
    return rotate_extra(_load(f.name, f.getvalue(), pdir), cur_rot(f.name))


def _photo_cfg(name, rot):
    ck = (name, rot)
    cfg = st.session_state["crop_cfg"].setdefault(
        ck,
        {"mgx": 0.004, "mgy": 0.006, "source_pad": 0.004},
    )
    return cfg


def _expand_source_quad(pts, pad):
    q = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    if pad <= 0:
        return q
    c = q.mean(axis=0)
    return c + (q - c) * (1.0 + float(pad))


def _warp_compat(img, pts, cfg):
    src = _expand_source_quad(pts, float(cfg.get("source_pad", 0.004)))
    margin = (float(cfg.get("mgx", 0.004)), float(cfg.get("mgy", 0.006)))
    try:
        params = inspect.signature(warp_points).parameters
        if "source_pad" in params:
            return warp_points(img, src, aspect=aspect, margin=margin, source_pad=0.0)
        return warp_points(img, src, aspect=aspect, margin=margin)
    except (TypeError, ValueError):
        return warp_points(img, src, aspect=aspect, margin=margin)


def get_pts(f):
    """回傳 (四角點, 自動判定是否需檢查, 是否手動)。"""
    key = (f.name, cur_rot(f.name))
    if key in st.session_state["manual"]:
        return st.session_state["manual"][key], False, True
    p, bad = _detect(f.name, f.getvalue(), pdir, cur_rot(f.name), DETECTOR_VERSION)
    if p is None:
        return None, True, False
    return p, bool(bad), False


def _depth_total_rows():
    end_depth = _parse_end_depth(hole)
    if end_depth is None or row_m <= 0 or end_depth < start_depth:
        return None
    return max(0, int(round((end_depth - float(start_depth)) / float(row_m))))


def detect_occupied_rows(im, n=4):
    fn = getattr(core, "detect_occupied_rows", None)
    if callable(fn):
        try:
            return int(fn(np.asarray(im), n))
        except Exception:
            pass
    return n


def effective_last_rows():
    if partial_depth_m > 0:
        return max(1, min(int(rows_per_box), int(np.ceil(partial_depth_m / row_m))))

    depth_rows = _depth_total_rows()
    if depth_rows is not None:
        rem = depth_rows % int(rows_per_box)
        return int(rem or rows_per_box)

    last = files[-1]
    img = get_img(last)
    pts, failed, _ = get_pts(last)
    if failed or pts is None:
        return int(rows_per_box)
    if not skip_warp:
        img = _warp_compat(img, pts, _photo_cfg(last.name, cur_rot(last.name)))
    return detect_occupied_rows(img, int(rows_per_box))


def total_rows_effective():
    depth_rows = _depth_total_rows()
    if partial_depth_m > 0:
        return max(0, (len(files) - 1) * int(rows_per_box) + effective_last_rows())
    if depth_rows is not None:
        return int(depth_rows)
    return (len(files) - 1) * int(rows_per_box) + effective_last_rows()


def output_files_effective():
    total = total_rows_effective()
    if total <= 0:
        return files
    need = int(np.ceil(total / float(rows_per_box)))
    return files[: min(len(files), need)]


def box_image(f):
    img = get_img(f)
    cfg = _photo_cfg(f.name, cur_rot(f.name))
    if not skip_warp:
        pts, failed, _ = get_pts(f)
        if failed or pts is None:
            raise ValueError("這張照片沒有可用的四個角點")
        img = _warp_compat(img, pts, cfg)
    im = Image.fromarray(img)
    out_files = output_files_effective()
    if out_files and f.name == out_files[-1].name:
        k = effective_last_rows()
        if k < rows_per_box:
            im = crop_partial(im, k, rows_per_box, (cfg["mgx"], cfg["mgy"]))
    return im


def _image_data_uri(img):
    b = io.BytesIO()
    img.convert("RGB").save(b, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode("ascii")


def _draggable_points(img, pts, key):
    display_w = min(900, int(img.shape[1]))
    display_h = int(round(img.shape[0] * display_w / img.shape[1]))
    result = point_editor(
        image=_image_data_uri(Image.fromarray(img)),
        points=np.asarray(pts, dtype=float).tolist(),
        image_width=int(img.shape[1]),
        image_height=int(img.shape[0]),
        display_width=display_w,
        display_height=display_h,
        key=key,
    )
    if result is None:
        return None
    try:
        if isinstance(result, str):
            result = json.loads(result)
        arr = np.asarray(result, dtype=np.float32).reshape(4, 2)
        return arr if np.all(np.isfinite(arr)) else None
    except Exception:
        return None


# ---------------- 1. 檢查 ----------------
st.subheader("1. 照片檢查")
st.error("⚠ 重要：自動抓角點只供初判。請在產生成果前，務必逐張看過左側 4 個角點與右側「校正後」結果。")

out_files = output_files_effective()
status = {}
for f in out_files:
    pts, bad, man = get_pts(f)
    if pts is None:
        status[f.name] = "⚠ 需設定"
    elif man:
        status[f.name] = "✋ 手動"
    elif bad:
        status[f.name] = "⚠ 需檢查"
    else:
        status[f.name] = "✓ 自動OK"

c = st.columns(4)
c[0].metric("✓ 自動OK", sum(v == "✓ 自動OK" for v in status.values()))
c[1].metric("⚠ 需檢查", sum(v == "⚠ 需檢查" for v in status.values()))
c[2].metric("✋ 手動", sum(v == "✋ 手動" for v in status.values()))
c[3].metric("⚠ 需設定", sum(v == "⚠ 需設定" for v in status.values()))

names = [f.name for f in out_files]
if not names:
    st.error("沒有可輸出的照片。")
    st.stop()
if st.session_state.get("sel_photo") not in names:
    st.session_state["sel_photo"] = names[0]


def _step(d):
    k = names.index(st.session_state["sel_photo"]) + d
    st.session_state["sel_photo"] = names[max(0, min(len(names) - 1, k))]


nb1, nb2, nb3 = st.columns([1, 1, 6])
nb1.button("◀ 上一張", on_click=_step, args=(-1,))
nb2.button("下一張 ▶", on_click=_step, args=(1,))
sel_name = nb3.selectbox("選擇照片", names, key="sel_photo")
sel = next(f for f in out_files if f.name == sel_name)
ck = (sel.name, cur_rot(sel.name))

cfg = _photo_cfg(sel.name, cur_rot(sel.name))
q1, q2, q3 = st.columns(3)
with q1:
    cfg["mgx"] = st.slider(
        "本張左右多留 %", 0.0, 3.0, float(cfg["mgx"] * 100), 0.1,
        key=f"mgx_{sel.name}_{ck[1]}",
    ) / 100
with q2:
    cfg["mgy"] = st.slider(
        "本張上下多留 %", 0.0, 3.0, float(cfg["mgy"] * 100), 0.1,
        key=f"mgy_{sel.name}_{ck[1]}",
    ) / 100
with q3:
    cfg["source_pad"] = st.slider(
        "本張角點向外安全留邊 %", 0.0, 2.0, float(cfg["source_pad"] * 100), 0.1,
        key=f"pad_{sel.name}_{ck[1]}",
    ) / 100

rot_val = st.selectbox(
    "此張額外旋轉（逆時針）", [0, 90, 180, 270],
    index=[0, 90, 180, 270].index(cur_rot(sel.name)), key=f"rot_{sel.name}",
)
if rot_val != cur_rot(sel.name):
    st.session_state["rot"][sel.name] = rot_val
    st.rerun()

img = get_img(sel)
pts, bad, man = get_pts(sel)

if man:
    st.info("✋ 本張已使用手動拖曳角點")
elif bad:
    st.warning("⚠ 自動結果需檢查")
elif pts is not None:
    st.success("✓ 自動結果")
else:
    st.error("⚠ 找不到四個角點")

colA, colB = st.columns(2)
with colA:
    st.caption("拖曳左圖 4 個黃色角點：左上 → 右上 → 右下 → 左下")
    if pts is not None:
        edited = _draggable_points(img, pts, key=f"point_editor_{sel.name}_{ck[1]}")
        if edited is not None and np.max(np.abs(edited - np.asarray(pts))) > 0.5:
            st.session_state["manual"][ck] = edited.astype(np.float32)
            st.rerun()
    else:
        st.image(img, use_container_width=True)

with colB:
    st.caption("校正後（成果用）")
    if pts is not None:
        try:
            st.image(box_image(sel), use_container_width=True)
        except Exception as e:
            st.error("校正預覽失敗")
            st.caption(str(e))
    else:
        st.info("沒有可用角點")

b1, b2 = st.columns(2)
if b1.button("恢復本張自動角點"):
    st.session_state["manual"].pop(ck, None)
    st.rerun()
if b2.button("重設本張裁切留邊"):
    st.session_state["crop_cfg"].pop(ck, None)
    st.rerun()

out_files = output_files_effective()
if out_files and sel.name == out_files[-1].name:
    last_k = effective_last_rows()
    end_depth = _parse_end_depth(hole)
    if partial_depth_m > 0:
        st.info(f"最後一箱：{partial_depth_m:g}m = {last_k} 槽（1m=1槽）")
    elif end_depth is not None:
        st.info(f"終深 {end_depth:g}m → 最後一箱保留 {last_k} 槽；超過終深的槽直接不輸出")
    else:
        st.info(f"最後一箱：自動判定 {last_k} 槽")

# ---------------- 2. 輸出 ----------------
st.subheader("2. 輸出成果")
_total_rows = total_rows_effective()
out_files = output_files_effective()
st.write(f"輸出 {len(out_files)} 張照片 / {_total_rows} 個箱號，每頁 {per_page} 個箱號。")
st.warning("⚠ 自動判定僅供初判；產生 Word / PDF 前，請務必逐張檢查照片。")
extra = max(0, len(files) - len(out_files))
if extra:
    st.info(f"已自動排除終深後的 {extra} 張照片。")

if st.button("產生 Word / PDF", type="primary"):
    boxes = []
    bar = st.progress(0.0, "處理照片中…")
    failed = None
    for i, f in enumerate(out_files):
        try:
            boxes.append(box_image(f))
        except Exception as e:
            failed = (f.name, e)
            break
        bar.progress((i + 1) / max(1, len(out_files)))

    if failed:
        bar.empty()
        st.error(f"處理 {failed[0]} 失敗")
        st.code(f"{type(failed[1]).__name__}: {failed[1]}")
    else:
        kw = dict(
            hole=hole,
            date=date,
            project=project,
            start_depth=start_depth,
            rows_per_box=rows_per_box,
            per_page=per_page,
            row_m=row_m,
            board=board_up if board_up else None,
            end_mark=end_mark,
            total_rows=_total_rows,
        )
        st.session_state.pop("pdf", None)
        st.session_state.pop("docx", None)
        try:
            st.session_state["pdf"] = build_pdf(boxes, **kw)
        except Exception as e:
            st.error("PDF 輸出失敗")
            st.code(f"{type(e).__name__}: {e}")
        try:
            st.session_state["docx"] = build_docx(boxes, **kw)
        except Exception as e:
            st.error("Word 輸出失敗")
            st.code(f"{type(e).__name__}: {e}")
        bar.empty()
        if "pdf" in st.session_state and "docx" in st.session_state:
            st.success("✓ Word / PDF 產生完成")

if "pdf" in st.session_state:
    d1, d2 = st.columns(2)
    d1.download_button(
        "下載 PDF", st.session_state["pdf"],
        file_name=f"{hole}_岩心照片.pdf", mime="application/pdf",
    )
    if "docx" in st.session_state:
        d2.download_button(
            "下載 Word", st.session_state["docx"],
            file_name=f"{hole}_岩心照片.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    else:
        d2.warning("Word 尚未產生")

st.caption(f"程式版本 {APP_VERSION}｜角點偵測 {DETECTOR_VERSION}")
