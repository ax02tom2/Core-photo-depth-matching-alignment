import io
import inspect
import re

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

import core

# 直接從 core 取必要函式；這樣即使雲端暫時還殘留舊版 core.py，app 也能繼續運作。
natural_key = core.natural_key
load_image = core.load_image
rotate_extra = core.rotate_extra
detect_inner = core.detect_inner
warp_points = core.warp_points
crop_partial = core.crop_partial
draw_corners = core.draw_corners
build_pdf = core.build_pdf
build_docx = core.build_docx

st.set_page_config(page_title="岩心照片校正與成果輸出", layout="wide")
st.title("岩心箱照片：轉橫 → 箱內四角校正 → 成果輸出")

APP_VERSION = "2.4"


def _parse_end_depth(text):
    """從孔號自動讀出括號內的終深，例如 H25-1B(50m) -> 50。"""
    m = re.search(r'[\(（]\s*(\d+(?:\.\d+)?)\s*m\s*[\)）]', str(text), re.I)
    if not m:
        return None
    return float(m.group(1))


def _depth_total_rows():
    """有終深時直接以終深控制成果範圍，避免最後空槽被硬算進去。"""
    end_depth = _parse_end_depth(hole)
    if end_depth is None or row_m <= 0 or end_depth < start_depth:
        return None
    rows = int(round((end_depth - float(start_depth)) / float(row_m)))
    return max(0, rows)


# ---------------- 側邊欄 ----------------
with st.sidebar:
    st.header("成果表頭")
    project = st.text_input("工程名稱", "114 年度鵠鵠崙地區潛在大規模崩塌調查監測計畫")
    hole = st.text_input("孔號", "H25-1B(50m)")
    date = st.text_input("日期", "114.3.29")
    start_depth = st.number_input("起始深度 (m)", 0, 500, 0)
    board_up = st.file_uploader(
        "自訂表頭告示牌照片（選填，預設用範例 Word 的）",
        type=["jpg", "jpeg", "png"],
    )

    st.header("轉向 / 校正")
    portrait_dir = st.radio(
        "直式照片（90°拍攝）轉橫的方向",
        ["逆時針", "順時針"],
        horizontal=True,
    )
    st.caption(
        "轉橫後標籤 H25-1B 應在箱子上緣右側、箱號 1–4 在右邊且由上往下。"
    )
    global_rot = st.selectbox("全部照片額外旋轉（逆時針）", [0, 90, 180, 270], index=0)
    aspect = st.number_input("成果圖整箱 寬/高", 1.5, 5.0, 3.12, 0.01)

    st.header("成果裁切")
    st.caption("先做四角透視校正，再在成果畫布留極小邊；不是單純裁切。最後一箱會按終深裁掉後續空槽。")
    mgx = st.slider("左右多留 %", 0.0, 3.0, 0.4, 0.1) / 100
    mgy = st.slider("上下多留 %", 0.0, 3.0, 0.6, 0.1) / 100
    source_pad = st.slider(
        "內角向箱外安全外擴 %",
        0.0,
        2.0,
        0.4,
        0.1,
        help="只增加來源安全邊；不改變最後成果比例。",
    ) / 100

    per_page = st.number_input("每頁箱號數", 4, 40, 20, 4)
    rows_per_box = st.number_input("每箱列數", 1, 10, 4)
    row_m = st.number_input("每槽代表深度 (m)", 0.1, 5.0, 1.0, 0.1)
    partial_depth_m = st.number_input(
        "最後一箱實際深度 (m)（0 = 自動）",
        0.0,
        float(rows_per_box * row_m),
        0.0,
        0.1,
        help="每槽 1m 時：1m = 1 槽、2m = 2 槽。H25-1B(50m) 不填時會依終深自動保留最後 2 槽。",
    )
    end_mark = st.checkbox("最後加「鑽探結束」", True)
    skip_warp = st.checkbox("照片已是正的，不做透視校正", False)

files = st.file_uploader(
    "依深度由淺到深上傳岩心箱照片（可多選，依檔名自然排序）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)
if not files:
    st.info("請先上傳照片。每張照片為一個岩心箱（4 列 = 4 個箱號）。")
    st.stop()

files = sorted(files, key=lambda f: natural_key(f.name))
for k in ("manual", "clicks", "rot", "confirmed"):
    st.session_state.setdefault(k, {})
pdir = "ccw" if portrait_dir == "逆時針" else "cw"


@st.cache_data(show_spinner=False, max_entries=60)
def _load(name, data, pdir):
    return load_image(io.BytesIO(data), portrait_dir=pdir)


@st.cache_data(show_spinner=False, max_entries=120)
def _detect(name, data, pdir, rot):
    img = rotate_extra(_load(name, data, pdir), rot)
    return detect_inner(img)


def cur_rot(name):
    return st.session_state["rot"].get(name, global_rot)


def get_img(f):
    return rotate_extra(_load(f.name, f.getvalue(), pdir), cur_rot(f.name))


def _expand_source_quad(pts, pad):
    """App 端安全外擴。讓 app 與新舊 core.py 都相容，不會因 source_pad 參數不一致而崩潰。"""
    q = np.array(pts, dtype=np.float32).reshape(4, 2)
    if pad <= 0:
        return q
    c = q.mean(axis=0)
    return c + (q - c) * (1.0 + float(pad))


def _warp_compat(img, pts):
    """與舊版/新版 core.warp_points 相容。

    舊版 warp_points 沒有 source_pad；新版有。安全外擴統一在 app 端做一次，
    新版 core 再傳 source_pad=0，避免重複外擴。
    """
    src = _expand_source_quad(pts, source_pad)
    try:
        params = inspect.signature(warp_points).parameters
        if "source_pad" in params:
            return warp_points(
                img,
                src,
                aspect=aspect,
                margin=(mgx, mgy),
                source_pad=0.0,
            )
        return warp_points(img, src, aspect=aspect, margin=(mgx, mgy))
    except (TypeError, ValueError):
        # 最後一道保險：完全使用舊版呼叫方式。
        return warp_points(img, src, aspect=aspect, margin=(mgx, mgy))


def _local_detect_occupied_rows(im, n=4):
    """最後一箱的相容版自動判斷；core 沒有 detect_occupied_rows 時由 app 執行。"""
    if n <= 1:
        return 1
    try:
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        blue = cv2.inRange(hsv, (85, 60, 45), (118, 255, 255)) > 0
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        H, W = im.shape[:2]
        scores = []
        for k in range(n):
            y0 = int(H * (k / n + 0.08 / n))
            y1 = int(H * ((k + 1) / n - 0.08 / n))
            x0, x1 = int(W * 0.08), int(W * 0.92)
            if y1 <= y0 or x1 <= x0:
                return n
            bm = blue[y0:y1, x0:x1]
            g = gray[y0:y1, x0:x1]
            non_blue = float(1.0 - bm.mean())
            texture = float(min(np.std(g) / 65.0, 1.0))
            edge = float(min(cv2.Canny(g, 40, 120).mean() / 255.0 / 0.12, 1.0))
            scores.append(0.68 * non_blue + 0.20 * texture + 0.12 * edge)
        head = np.array(scores[: min(2, n)], dtype=float)
        adaptive = max(0.27, float(np.median(head)) * 0.42)
        occupied = [s >= adaptive for s in scores]
        k = 0
        for ok in occupied:
            if not ok:
                break
            k += 1
        if k == 0 or all(s >= adaptive * 0.90 for s in scores):
            return n
        return max(1, min(n, k))
    except Exception:
        return n


def detect_occupied_rows(im, n=4):
    fn = getattr(core, "detect_occupied_rows", None)
    if callable(fn):
        try:
            return int(fn(np.asarray(im), n))
        except Exception:
            pass
    return _local_detect_occupied_rows(np.asarray(im), n)


def get_pts(f):
    """回傳 (四角點, 自動抓不到, 是否手動)。

    本版不再替使用者判定「自動OK」。只負責提供自動角點，最後一定由使用者逐張確認。
    """
    key = (f.name, cur_rot(f.name))
    if key in st.session_state["manual"]:
        return st.session_state["manual"][key], False, True
    p, _touch = _detect(f.name, f.getvalue(), pdir, cur_rot(f.name))
    if p is None:
        return None, True, False
    return p, False, False



def effective_last_rows():
    """最後一箱保留幾槽。

    優先順序：手動「最後一箱實際深度(m)」 > 孔號終深 > 最後照片影像判定。
    每槽深度 row_m=1 時，1m 就是 1 槽，2m 就是 2 槽。
    """
    if partial_depth_m > 0:
        rows = int(np.ceil(float(partial_depth_m) / float(row_m)))
        return max(1, min(int(rows_per_box), rows))

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
        img = _warp_compat(img, pts)
    return detect_occupied_rows(img, int(rows_per_box))

def total_rows_effective():
    """成果總列數。手動最後一箱深度優先，其次使用孔號終深。"""
    depth_rows = _depth_total_rows()
    if partial_depth_m > 0:
        full_before = max(0, len(files) - 1) * int(rows_per_box)
        return full_before + int(effective_last_rows())
    if depth_rows is not None:
        return int(depth_rows)
    return (len(files) - 1) * int(rows_per_box) + int(effective_last_rows())


def output_files_effective():
    """只輸出終深以前需要的照片，多出的照片直接排除。"""
    total = total_rows_effective()
    if total <= 0:
        return files
    need = int(np.ceil(total / int(rows_per_box)))
    return files[:min(len(files), need)]


def box_image(f):
    img = get_img(f)
    if not skip_warp:
        pts, failed, _ = get_pts(f)
        if failed or pts is None:
            raise ValueError("這張照片尚未設定可用的四個角點")
        img = _warp_compat(img, pts)
    im = Image.fromarray(img)
    out_files = output_files_effective()
    if out_files and f.name == out_files[-1].name:
        k = effective_last_rows()
        if k < rows_per_box:
            im = crop_partial(im, k, rows_per_box, (mgx, mgy))
    return im


# ---------------- 1. 逐張檢查 ----------------
st.subheader("1. 每張照片都要檢查確認")
st.caption("自動角點只提供初始位置，不自動判定正確與否；所有照片都必須由你確認後，才能產生成果。")

out_files = output_files_effective()
status = {}
valid_pts = {}
confirmed_now = {}
for f in out_files:
    pts, failed, man = get_pts(f)
    ck = (f.name, cur_rot(f.name))
    valid = pts is not None and not failed
    valid_pts[f.name] = (pts, valid, man)
    ok = bool(st.session_state["confirmed"].get(ck, False)) and valid
    confirmed_now[f.name] = ok
    status[f.name] = "已確認" if ok else ("待設定" if not valid else "待確認")

counts = {
    "已確認": sum(v == "已確認" for v in status.values()),
    "待確認": sum(v == "待確認" for v in status.values()),
    "待設定": sum(v == "待設定" for v in status.values()),
}
b1, b2, b3 = st.columns(3)
b1.metric("✓ 已確認", counts["已確認"])
b2.metric("○ 待確認", counts["待確認"])
b3.metric("⚠ 待設定", counts["待設定"])

names = [f.name for f in out_files]
names = [f.name for f in output_files_effective()]
if st.session_state.get("sel_photo") not in names:
    st.session_state["sel_photo"] = names[0]


def _step(d):
    k = names.index(st.session_state["sel_photo"]) + d
    st.session_state["sel_photo"] = names[max(0, min(len(names) - 1, k))]


nb1, nb2, nb3 = st.columns([1, 1, 6])
nb1.button("◀ 上一張", on_click=_step, args=(-1,))
nb2.button("下一張 ▶", on_click=_step, args=(1,))
sel_name = nb3.selectbox("選擇照片", names, key="sel_photo")
sel = next(f for f in files if f.name == sel_name)
sel_status = status[sel_name]

if sel_status == "待設定":
    st.error("⚠ 找不到四個角點，請手動點四角")
elif sel_status == "待確認":
    st.warning("○ 請檢查這張照片的四個角點與校正結果")
else:
    st.success("✓ 這張已確認")

out_files = output_files_effective()
if out_files and sel.name == out_files[-1].name:
    auto_last = effective_last_rows()
    end_depth = _parse_end_depth(hole)
    if partial_depth_m > 0:
        st.info(f"最後一箱：{partial_depth_m:g}m → {auto_last}/{rows_per_box} 槽")
    elif end_depth is not None:
        st.info(f"終深 {end_depth:g}m → 最後一箱 {auto_last}/{rows_per_box} 槽")
    else:
        st.info(f"自動判定：最後一箱 {auto_last}/{rows_per_box} 槽")

r1, r2 = st.columns([1, 3])
rot_val = r1.selectbox(
    "此張額外旋轉（逆時針）",
    [0, 90, 180, 270],
    index=[0, 90, 180, 270].index(cur_rot(sel.name)),
    key=f"rot_{sel.name}",
)
if rot_val != cur_rot(sel.name):
    st.session_state["rot"][sel.name] = rot_val
    st.session_state["confirmed"].pop((sel.name, rot_val), None)
    st.rerun()

img = get_img(sel)
pts, failed, man = get_pts(sel)

colA, colB = st.columns(2)
with colA:
    st.caption("原圖＋四個箱內內角")
    if pts is not None:
        st.image(draw_corners(img, pts), use_container_width=True)
    else:
        st.image(img, use_container_width=True)
        st.error("目前沒有可用角點；請在下方手動點 4 個角。")
with colB:
    st.caption("校正後（成果用）")
    if pts is not None:
        try:
            preview = box_image(sel)
            st.image(preview, use_container_width=True)
        except Exception:
            st.error("目前無法產生校正預覽，請檢查四個角點。")
    else:
        st.info("設定四個角點後，這裡會顯示成果預覽。")

ck = (sel.name, cur_rot(sel.name))
confirm_value = bool(st.session_state["confirmed"].get(ck, False))
if pts is not None:
    confirm_value = st.checkbox(
        "我已確認這張照片的四個角點與校正結果",
        value=confirm_value,
        key=f"confirm_{sel.name}_{ck[1]}",
    )
    st.session_state["confirmed"][ck] = bool(confirm_value)
else:
    st.session_state["confirmed"][ck] = False

with st.expander("角點不準？手動點選箱內四個內角"):
    st.caption("依序點：左上 → 右上 → 右下 → 左下。")
    scale = 900 / img.shape[1]
    small = np.array(Image.fromarray(img).resize((900, int(img.shape[0] * scale))))
    clicks = st.session_state["clicks"].setdefault(ck, [])
    for q in clicks:
        small[
            max(0, int(q[1]) - 7) : int(q[1]) + 7,
            max(0, int(q[0]) - 7) : int(q[0]) + 7,
        ] = (255, 0, 0)
    v = streamlit_image_coordinates(
        Image.fromarray(small),
        key=f"clk_{sel.name}_{ck[1]}_{len(clicks)}",
    )
    if v and len(clicks) < 4 and (v["x"], v["y"]) not in [tuple(c) for c in clicks]:
        clicks.append((v["x"], v["y"]))
        st.rerun()
    st.write(f"已點 {len(clicks)}/4")
    b1, b2 = st.columns(2)
    if b1.button("套用手動角點", disabled=len(clicks) != 4):
        st.session_state["manual"][ck] = np.array(clicks, np.float32) / scale
        st.session_state["clicks"][ck] = []
        st.session_state["confirmed"][ck] = False
        st.rerun()
    if b2.button("清除，回到自動"):
        st.session_state["clicks"][ck] = []
        st.session_state["manual"].pop(ck, None)
        st.session_state["confirmed"][ck] = False
        st.rerun()

# ---------------- 2. 輸出 ----------------
st.subheader("2. 輸出成果")
_total_rows = total_rows_effective()
out_files = output_files_effective()
pending = [f.name for f in out_files if not confirmed_now.get(f.name, False)]
valid_all = all(valid_pts[f.name][1] for f in out_files) if out_files else False
ready = bool(out_files) and valid_all and not pending
st.write(f"輸出 {len(out_files)} 張照片 / {_total_rows} 個箱號，每頁 {per_page} 個箱號。")
if pending:
    st.warning(f"尚有 {len(pending)} 張照片未確認；全部確認後才能產生成果。")
extra = max(0, len(files) - len(out_files))
if extra:
    st.info(f"已自動排除終深後的 {extra} 張照片。")

if st.button("產生 Word / PDF", type="primary", disabled=not ready):
    boxes, bar = [], st.progress(0.0, "處理照片中…")
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
            hole=hole, date=date, project=project, start_depth=start_depth,
            rows_per_box=rows_per_box, per_page=per_page, row_m=row_m,
            board=board_up if board_up else None, end_mark=end_mark,
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
        file_name=f"{hole}_岩心照片.pdf", mime="application/pdf"
    )
    if "docx" in st.session_state:
        d2.download_button(
            "下載 Word", st.session_state["docx"],
            file_name=f"{hole}_岩心照片.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    else:
        d2.warning("Word 尚未產生")

st.caption(f"程式版本 {APP_VERSION}")
