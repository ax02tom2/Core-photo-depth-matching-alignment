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

APP_VERSION = "2.11"
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

st.set_page_config(page_title="岩心照片校正與成果輸出", layout="wide", initial_sidebar_state="expanded")

# ---------------- 視覺樣式 ----------------
st.markdown("""
<style>
:root{--navy:#17324d;--blue:#0b6fb3;--blue2:#eaf4fb;--teal:#0f7c7c;--teal2:#eaf8f7;--amber:#a96b00;--amber2:#fff7e6;--red:#b42318;--red2:#fff0ee;--line:#d8e1ea;--bg:#f4f7fa;}
[data-testid="stAppViewContainer"]{background:linear-gradient(180deg,#f7f9fb 0%,#eef3f7 100%);}
[data-testid="stHeader"]{background:rgba(255,255,255,.75);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#f6f9fc 0%,#edf3f8 100%);border-right:1px solid var(--line);}
.block-container{padding-top:1.1rem;padding-bottom:2rem;max-width:1500px;}
h1{color:var(--navy);letter-spacing:.02em;}
.section-card{border:1px solid var(--line);border-radius:14px;padding:13px 16px;margin:4px 0 12px 0;background:white;box-shadow:0 4px 14px rgba(26,51,77,.06);}
.section-card.blue{background:linear-gradient(135deg,#eaf4fb 0%,#f7fbff 100%);border-color:#a8cbe5;}
.section-card.teal{background:linear-gradient(135deg,#eaf8f7 0%,#f7fcfc 100%);border-color:#a6d8d5;}
.section-card.amber{background:linear-gradient(135deg,#fff7e6 0%,#fffdf8 100%);border-color:#ecd49d;}
.section-title{font-size:1.05rem;font-weight:800;color:var(--navy);margin:0 0 3px 0;}
.section-sub{font-size:.88rem;color:#506273;margin:0;}
.hero-preview{border:2px solid #0b6fb3;border-radius:16px;padding:14px 16px 12px 16px;background:linear-gradient(145deg,#eaf4fb 0%,#ffffff 58%,#f6fbff 100%);box-shadow:0 8px 24px rgba(11,111,179,.14);}
.hero-preview .hero-title{font-size:1.28rem;font-weight:900;color:#0b5488;margin:0 0 2px 0;}
.hero-preview .hero-sub{font-size:.9rem;color:#486276;margin:0 0 8px 0;}
.action-card{border:2px solid #0f7c7c;border-radius:14px;padding:11px 14px;background:linear-gradient(135deg,#eaf8f7 0%,#ffffff 100%);box-shadow:0 6px 18px rgba(15,124,124,.10);margin:8px 0 10px 0;}
.action-card .action-title{font-size:1.02rem;font-weight:900;color:#0f6565;margin:0 0 2px 0;}
.action-card .action-sub{font-size:.87rem;color:#4a6464;margin:0;}
.notice-card{border-left:5px solid #d97706;border-radius:10px;padding:9px 12px;background:var(--amber2);color:#7a4b00;margin:8px 0 14px 0;font-weight:700;}
.status-ok{border-left:5px solid #15803d;border-radius:10px;padding:8px 12px;background:#ecfdf3;color:#166534;font-weight:700;}
.status-warn{border-left:5px solid #d97706;border-radius:10px;padding:8px 12px;background:#fff7ed;color:#9a3412;font-weight:700;}
.status-bad{border-left:5px solid #b42318;border-radius:10px;padding:8px 12px;background:#fff0ee;color:#9f1d1a;font-weight:700;}
.small-note{font-size:.82rem;color:#657789;}
.nav-caption-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.6rem;margin:2px 0 4px 0;color:#17324d;font-size:.86rem;font-weight:800;}
.nav-caption-row div{min-height:1.1rem;}
[data-testid="stHorizontalBlock"] [data-testid="stButton"] button,
[data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] [data-baseweb="select"] > div{min-height:46px;height:46px;box-sizing:border-box;border-radius:10px;}
[data-testid="stHorizontalBlock"] [data-testid="stButton"] button{font-size:1rem;font-weight:800;padding:0 .8rem;}
[data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] [data-baseweb="select"] > div{display:flex;align-items:center;font-size:1.08rem;font-weight:900;border:2.5px solid #17324d;background:#ffffff;color:#17324d;box-shadow:0 4px 12px rgba(23,50,77,.12);padding-top:0;padding-bottom:0;}
[data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within{border-color:#0b6fb3;box-shadow:0 0 0 3px rgba(11,111,179,.16),0 4px 12px rgba(23,50,77,.12);}
[data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdownContainer"]{color:#17324d;font-weight:900;}
.nav-label{height:46px;display:flex;align-items:center;padding:0 .2rem;color:#17324d;font-size:.92rem;font-weight:900;}
button[kind="primary"]{min-height:46px;font-size:1.05rem;font-weight:800;box-shadow:0 5px 16px rgba(11,111,179,.18);}
[data-testid="stMetric"]{background:rgba(255,255,255,.78);border:1px solid var(--line);border-radius:12px;padding:8px 10px;}
</style>
""", unsafe_allow_html=True)
st.title("岩心箱照片校正與成果輸出")
st.markdown('<div class="section-card blue"><div class="section-title">岩心箱照片校正工作區</div><div class="section-sub">先看左側箱框，再以右側「校正後（成果用）」作為最終成果預覽。</div></div>', unsafe_allow_html=True)


def _parse_end_depth(text):
    m = re.search(r"[\(（]\s*(\d+(?:\.\d+)?)\s*m\s*[\)）]", str(text), re.I)
    return float(m.group(1)) if m else None


# ---------------- 側邊欄 ----------------
with st.sidebar:
    st.header("成果表頭")
    project = st.text_input("工程名稱", "")
    hole = st.text_input("孔號", "")
    date_value = st.date_input("日期", value=None, format="YYYY.MM.DD")
    date = date_value.strftime("%Y.%m.%d") if date_value else ""
    start_depth = 0
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

    # 深度與槽數固定為：每槽 1m、每箱 4 槽。
    # 最後深度直接從「孔號」中的 (50m) 之類文字自動判讀；不再提供容易混淆的最後深度輸入。
    ROW_M = 1.0
    rows_per_box = 4
    st.header("版面")
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
for k in ("manual", "rot", "crop_cfg", "edit_nonce", "crop_nonce"):
    st.session_state.setdefault(k, {})
st.session_state.setdefault("state_version", "2.11")
if st.session_state.get("state_version") != "2.11":
    st.session_state["manual"].clear()
    st.session_state["rot"].clear()
    st.session_state["crop_cfg"].clear()
    st.session_state["edit_nonce"].clear()
    st.session_state["crop_nonce"].clear()
    st.session_state["state_version"] = "2.10"

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
        {"mgx": 0.0, "mgy": 0.0, "source_pad": 0.0},
    )
    return cfg


def _nonce(store_name, key):
    return int(st.session_state[store_name].get(key, 0))


def _bump_nonce(store_name, key):
    st.session_state[store_name][key] = _nonce(store_name, key) + 1


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
    try:
        p, bad = _detect(f.name, f.getvalue(), pdir, cur_rot(f.name), DETECTOR_VERSION)
    except Exception:
        # 自動偵測失敗時，不讓整個 Streamlit 頁面中斷。
        # 直接把本張標成需檢查，讓使用者用拖曳四點修正。
        return None, True, False
    if p is None:
        return None, True, False
    return p, bool(bad), False


def _depth_total_rows():
    end_depth = _parse_end_depth(hole)
    if end_depth is None or end_depth < start_depth:
        return None
    return max(0, int(round((end_depth - float(start_depth)) / ROW_M)))


def detect_occupied_rows(im, n=4):
    fn = getattr(core, "detect_occupied_rows", None)
    if callable(fn):
        try:
            return int(fn(np.asarray(im), n))
        except Exception:
            pass
    return n


def effective_last_rows():
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
st.markdown('<div class="notice-card">⚠ 重要提醒：自動抓角點只供初判。產生成果前，請先檢查每張照片左側箱框與右側「校正後（成果用）」預覽。</div>', unsafe_allow_html=True)

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


# 照片切換：三個控制項做成同尺寸、同高度，避免「選擇照片」顯得太小。
st.markdown('<div class="nav-caption-row"><div>照片切換</div><div></div><div>照片選擇</div></div>', unsafe_allow_html=True)
nb1, nb2, nb3 = st.columns(3, gap="small")
with nb1:
    st.button("◀ 上一張", on_click=_step, args=(-1,), use_container_width=True)
with nb2:
    st.button("下一張 ▶", on_click=_step, args=(1,), use_container_width=True)
with nb3:
    st.markdown('<div class="nav-label">選擇照片</div>', unsafe_allow_html=True)
    sel_name = st.selectbox("照片檔案", names, key="sel_photo", label_visibility="collapsed")
sel = next(f for f in out_files if f.name == sel_name)
ck = (sel.name, cur_rot(sel.name))

cfg = _photo_cfg(sel.name, cur_rot(sel.name))
with st.container(border=True):
    st.markdown('<div class="section-card teal" style="margin-top:0"><div class="section-title">本張調整</div><div class="section-sub">只會影響目前這一張照片。</div></div>', unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        cfg["mgx"] = st.slider(
            "左右留邊 %", 0.0, 3.0, float(cfg["mgx"] * 100), 0.1,
            key=f"mgx_{sel.name}_{ck[1]}_{_nonce('crop_nonce', ck)}",
        ) / 100
    with q2:
        cfg["mgy"] = st.slider(
            "上下留邊 %", 0.0, 3.0, float(cfg["mgy"] * 100), 0.1,
            key=f"mgy_{sel.name}_{ck[1]}_{_nonce('crop_nonce', ck)}",
        ) / 100
    with q3:
        cfg["source_pad"] = st.slider(
            "角點向外留邊 %", 0.0, 2.0, float(cfg["source_pad"] * 100), 0.1,
            key=f"pad_{sel.name}_{ck[1]}_{_nonce('crop_nonce', ck)}",
        ) / 100
    with q4:
        st.markdown("&nbsp;")
        reset_crop = st.button("重設本張裁切留邊", key=f"reset_crop_{sel.name}_{ck[1]}_{_nonce('crop_nonce', ck)}")

if reset_crop:
    st.session_state["crop_cfg"].pop(ck, None)
    _bump_nonce("crop_nonce", ck)
    st.rerun()

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
    st.error("⚠ 自動偵測失敗，請拖曳四個黃色角點")

colA, colB = st.columns([1, 1], gap="large")
with colA:
    st.markdown('<div class="section-card"><div class="section-title">拖曳黃色角點調整箱框</div><div class="section-sub">調整左側箱框後，右側成果預覽會同步更新。</div></div>', unsafe_allow_html=True)
    if pts is not None:
        edited = _draggable_points(img, pts, key=f"point_editor_{sel.name}_{ck[1]}_{_nonce('edit_nonce', ck)}")
        if edited is not None and np.max(np.abs(edited - np.asarray(pts))) > 0.5:
            st.session_state["manual"][ck] = edited.astype(np.float32)
            st.rerun()
    else:
        st.image(img, use_container_width=True)

with colB:
    st.markdown('<div class="hero-preview"><div class="hero-title">校正後（成果用）</div><div class="hero-sub">這裡就是最後會放進 Word / PDF 的成果樣式。</div></div>', unsafe_allow_html=True)
    if pts is not None:
        try:
            st.image(box_image(sel), use_container_width=True)
        except Exception as e:
            st.error("校正預覽失敗")
            st.caption(str(e))
    else:
        st.info("沒有可用角點")

reset_auto = st.button("恢復本張自動角點", key=f"reset_auto_{sel.name}_{ck[1]}_{_nonce('edit_nonce', ck)}")
if reset_auto:
    st.session_state["manual"].pop(ck, None)
    _bump_nonce("edit_nonce", ck)
    st.rerun()

out_files = output_files_effective()
if out_files and sel.name == out_files[-1].name:
    last_k = effective_last_rows()
    end_depth = _parse_end_depth(hole)
    if end_depth is not None:
        st.info(f"終深 {end_depth:g}m → 最後一箱保留 {last_k} 槽；超過終深的槽直接不輸出")
    else:
        st.info(f"最後一箱：自動判定 {last_k} 槽")

# ---------------- 2. 輸出 ----------------
st.markdown('<div class="action-card"><div class="action-title">📄 產生 Word / PDF</div><div class="action-sub">確認每張照片的校正結果後，按下按鈕產生正式成果。</div></div>', unsafe_allow_html=True)
_total_rows = total_rows_effective()
out_files = output_files_effective()
st.write(f"輸出 {len(out_files)} 張照片 / {_total_rows} 個箱號，每頁 {per_page} 個箱號。")
st.markdown('<div class="notice-card">⚠ 產生成果前請先檢查每張照片；需要修正時可直接拖曳黃色角點。</div>', unsafe_allow_html=True)
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
            row_m=ROW_M,
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

st.caption(f"程式版本 {APP_VERSION}")
