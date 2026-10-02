import io
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

pdfmetrics.registerFont(UnicodeCIDFont("MSung-Light"))

HERE = os.path.dirname(os.path.abspath(__file__))
BOARD_PATH = os.path.join(HERE, "assets", "board.jpg")

# ---- 版面（依範例 Word 量測：A4、左右 2cm、上下 1cm）----
PAGE_MX, PAGE_MY = 2.0, 1.0
HEADER_W, HEADER_H = 15.4, 3.17
BOX_W, BOX_H = 15.15, 4.85
NUM_COL_W = 1.0

SERIF_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSerifCJK-Regular.ttc",
    "C:/Windows/Fonts/mingliu.ttc",
    "C:/Windows/Fonts/msjh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# ------------------------------------------------------------------ 影像前處理
def load_image(file_or_bytes, max_side=3200, portrait_dir="ccw"):
    """讀圖 → 修正 EXIF → 一律轉成橫式（直式照片依 portrait_dir 轉 90°）"""
    im = Image.open(file_or_bytes)
    im = ImageOps.exif_transpose(im).convert("RGB")
    if max(im.size) > max_side:
        r = max_side / max(im.size)
        im = im.resize((int(im.width * r), int(im.height * r)), Image.LANCZOS)
    arr = np.array(im)
    if arr.shape[0] > arr.shape[1]:
        arr = np.rot90(arr, 1 if portrait_dir == "ccw" else -1)
    return np.ascontiguousarray(arr)


def rotate_extra(img, deg):
    """額外手動旋轉：0 / 90(逆時針) / 180 / 270"""
    return np.ascontiguousarray(np.rot90(img, (deg // 90) % 4))


def order_pts(pts):
    """排序為 左上、右上、右下、左下"""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _reduce_to_quad(hull):
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.005, 0.12, 60):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) <= 4:
            if len(ap) == 4:
                return ap.reshape(4, 2).astype(np.float32)
            break
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def _extreme_quad(hull):
    """凸包上最靠近四個角落的點（x+y、x-y 的極值），比多邊形近似更貼近真實角"""
    p = hull.reshape(-1, 2).astype(np.float32)
    sm_, df = p.sum(axis=1), p[:, 0] - p[:, 1]
    return np.array([p[np.argmin(sm_)], p[np.argmax(df)], p[np.argmax(sm_)], p[np.argmin(df)]], np.float32)


def _blue_mask(sm):
    hsv = cv2.cvtColor(sm, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (85, 80, 60), (118, 255, 255))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def detect_inner(img):
    """找「岩心所在的那一格」＝藍色岩心箱內側的四個內角（橘框）。
    做法：藍色箱壁/隔板會圍出 4 個「洞」（每個洞 = 一條岩心槽），
    把這幾個洞合起來取外框，再縮成 4 個角。腳、土、旁邊的箱子不會形成這種洞，所以不受影響。
    回傳 (pts, bad)；bad=True 表示不可靠（請手動點）"""
    h, w = img.shape[:2]
    sc = 1000.0 / max(h, w)
    sm = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    sh, sw = sm.shape[:2]
    ms = min(sh, sw)
    mask = _blue_mask(sm)
    k = max(3, int(ms * 0.006))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    if hier is not None:
        for idx, c in enumerate(cnts):
            if hier[0][idx][3] != -1:  # 有父輪廓 = 洞
                a = cv2.contourArea(c)
                if a > 0.004 * sh * sw:
                    holes.append((a, c))
    if not holes:
        return None, True
    holes.sort(key=lambda t: -t[0])
    a0, c0 = holes[0]
    (cx0, cy0), (rw, rh), _ = cv2.minAreaRect(c0)
    minor = min(rw, rh)
    grp = [c0]
    for a, c in holes[1:]:
        (cx, cy), _, _ = cv2.minAreaRect(c)
        if a >= 0.25 * a0 and np.hypot(cx - cx0, cy - cy0) < 4.5 * minor:
            grp.append(c)
    hull = cv2.convexHull(np.vstack(grp))
    quad = _extreme_quad(hull)
    area = cv2.contourArea(quad)
    bad = (len(grp) < 3) or area < 0.05 * sh * sw
    bad = bad or bool(np.any(quad < 2) or np.any(quad[:, 0] > sw - 3) or np.any(quad[:, 1] > sh - 3))
    return order_pts(quad / sc), bool(bad)


def warp_inner(img, pts, out_w=2400, aspect=2.87, inner=(0.042, 0.055, 0.107, 0.07)):
    """把四個內角（岩心槽範圍）映射到輸出圖中的內框位置，
    內框外側保留藍色箱緣（inner = 左、右、上、下 留邊比例）。"""
    l, r, t, b = inner
    out_h = int(out_w / aspect)
    x0, x1 = out_w * l, out_w * (1 - r)
    y0, y1 = out_h * t, out_h * (1 - b)
    dst = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    M = cv2.getPerspectiveTransform(order_pts(pts), dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def draw_corners(img, pts, color=(255, 0, 0)):
    out = img.copy()
    p = np.array(pts, np.int32)
    t = max(3, img.shape[1] // 300)
    cv2.polylines(out, [p.reshape(-1, 1, 2)], True, color, t)
    for q in p:
        cv2.circle(out, tuple(int(v) for v in q), t * 3, (255, 255, 0), -1)
    return out


# ------------------------------------------------------------------ 表頭（告示牌照片 + 填字）
def _font(size):
    for p in SERIF_PATHS:
        if os.path.exists(p):
            for idx in (3, 0):
                try:
                    return ImageFont.truetype(p, size, index=idx)
                except Exception:
                    continue
    return ImageFont.load_default()


def make_header(hole, depth, date, project, board=None):
    """用告示牌照片當底，把孔號/深度/日期/工程名稱蓋上去（同範例 Word）"""
    im = Image.open(board or BOARD_PATH).convert("RGB")
    W, H = im.size
    fill = im.getpixel((int(W * 0.30), int(H * 0.64)))  # 取欄位底色
    d = ImageDraw.Draw(im)

    def cell(x0, y0, x1, y1, text, size, left=False):
        if not text:
            return
        f = _font(size)
        bb = d.textbbox((0, 0), text, font=f)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        x = x0 * W + (12 if left else ((x1 - x0) * W - tw) / 2)
        y = y0 * H + ((y1 - y0) * H - th) / 2 - bb[1]
        d.text((x, y), text, font=f, fill=(20, 20, 20))

    d.rectangle([W * 0.233, H * 0.286, W * 0.969, H * 0.528], fill=fill)
    # 長工程名稱自動縮字
    size = int(H * 0.17)
    while size > 20 and d.textlength(project, font=_font(size)) > W * 0.72:
        size -= 2
    cell(0.233, 0.286, 0.969, 0.528, project, size, left=True)
    cell(0.116, 0.528, 0.49, 0.755, hole, int(H * 0.17))
    cell(0.116, 0.786, 0.49, 0.969, date, int(H * 0.17))
    cell(0.613, 0.528, 0.985, 0.755, depth, int(H * 0.17))
    return im


def _jpeg(im, q=88):
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=q)
    b.seek(0)
    return b


def _pages(boxes, rows_per_box, per_page, start_depth, row_m):
    bpp = per_page // rows_per_box
    for p in range(0, len(boxes), bpp):
        chunk = boxes[p:p + bpp]
        r0 = p * rows_per_box
        n = len(chunk) * rows_per_box
        d0 = start_depth + r0 * row_m
        yield chunk, r0, n, f"{d0}~{d0 + n * row_m}m"


# ------------------------------------------------------------------ PDF
def build_pdf(boxes, hole, date, project, start_depth=0, rows_per_box=4,
              per_page=20, row_m=1, board=None, end_mark=True):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    x0 = PAGE_MX * cm
    for chunk, r0, n, depth in _pages(boxes, rows_per_box, per_page, start_depth, row_m):
        y = H - PAGE_MY * cm
        hi = make_header(hole, depth, date, project, board)
        c.drawImage(ImageReader(_jpeg(hi, 92)), x0, y - HEADER_H * cm, HEADER_W * cm, HEADER_H * cm)
        y -= HEADER_H * cm
        rh = BOX_H * cm / rows_per_box
        for i, im in enumerate(chunk):
            c.drawImage(ImageReader(_jpeg(im)), x0, y - BOX_H * cm, BOX_W * cm, BOX_H * cm)
            c.setFont("Helvetica", 16)
            for k in range(rows_per_box):
                num = r0 + i * rows_per_box + k + 1
                c.drawString(x0 + (HEADER_W + 0.3) * cm, y - (k + 0.5) * rh - 5.5, str(num))
            y -= BOX_H * cm
        if r0 + n >= len(boxes) * rows_per_box and end_mark:
            c.setFont("MSung-Light", 10.5)
            c.drawCentredString(W / 2, y - 0.8 * cm, "鑽探結束")
        c.showPage()
    c.save()
    return buf.getvalue()


# ------------------------------------------------------------------ Word
def build_docx(boxes, hole, date, project, start_depth=0, rows_per_box=4,
               per_page=20, row_m=1, board=None, end_mark=True):
    from docx import Document
    from docx.enum.table import WD_ROW_HEIGHT_RULE
    from docx.enum.text import WD_LINE_SPACING, WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(PAGE_MX)
    sec.top_margin = sec.bottom_margin = Cm(PAGE_MY)
    sec.header_distance = sec.footer_distance = Cm(0)
    st = doc.styles["Normal"]
    st.font.name = "Arial"
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "PMingLiU")
    st.paragraph_format.space_after = Pt(0)
    st.paragraph_format.space_before = Pt(0)

    def tight(p, exact=None):
        pf = p.paragraph_format
        pf.space_after = pf.space_before = Pt(0)
        if exact:
            pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
            pf.line_spacing = Pt(exact)

    def no_margins(tbl):
        pr = tbl._tbl.tblPr
        m = OxmlElement("w:tblCellMar")
        for side in ("top", "left", "bottom", "right"):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:w"), "0")
            e.set(qn("w:type"), "dxa")
            m.append(e)
        pr.append(m)

    first = True
    for chunk, r0, n, depth in _pages(boxes, rows_per_box, per_page, start_depth, row_m):
        sp = doc.add_paragraph()
        tight(sp, 1)
        sp.add_run("").font.size = Pt(1)
        sp.paragraph_format.page_break_before = not first
        first = False

        tbl = doc.add_table(rows=1 + len(chunk), cols=2)
        tbl.autofit = False
        no_margins(tbl)
        widths = (Cm(HEADER_W), Cm(NUM_COL_W))
        lay = OxmlElement("w:tblLayout")
        lay.set(qn("w:type"), "fixed")
        tbl._tbl.tblPr.append(lay)
        for j, wd in enumerate(widths):
            tbl.columns[j].width = wd
            for c_ in tbl.columns[j].cells:
                c_.width = wd
        # 表頭列（合併兩欄）
        hrow = tbl.rows[0]
        hrow.height, hrow.height_rule = Cm(HEADER_H), WD_ROW_HEIGHT_RULE.EXACTLY
        hc = hrow.cells[0].merge(hrow.cells[1])
        p = hc.paragraphs[0]
        tight(p)
        hi = make_header(hole, depth, date, project, board)
        p.add_run().add_picture(_jpeg(hi, 92), width=Cm(HEADER_W), height=Cm(HEADER_H - 0.05))
        rh = BOX_H / rows_per_box
        for i, im in enumerate(chunk):
            row = tbl.rows[i + 1]
            row.height, row.height_rule = Cm(BOX_H), WD_ROW_HEIGHT_RULE.EXACTLY
            for j, wd in enumerate(widths):
                row.cells[j].width = wd
            p = row.cells[0].paragraphs[0]
            tight(p)
            p.add_run().add_picture(_jpeg(im), width=Cm(BOX_W), height=Cm(BOX_H - 0.05))
            cell = row.cells[1]
            for k in range(rows_per_box):
                p = cell.paragraphs[0] if k == 0 else cell.add_paragraph()
                tight(p, rh / 2.54 * 72)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(str(r0 + i * rows_per_box + k + 1))
                r.font.size = Pt(16)
    if end_mark:
        p = doc.add_paragraph()
        tight(p)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run("鑽探結束").font.size = Pt(10.5)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()
