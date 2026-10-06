import base64
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

# ⚠️ 【字體設定】如果您更換系統，可在此新增該系統的中文字型路徑
pdfmetrics.registerFont(UnicodeCIDFont("MSung-Light"))

HERE = os.path.dirname(os.path.abspath(__file__))

# ⚠️ 【參數調整區】可調整 Word 與 PDF 輸出的版面間距與大小
PAGE_MX, PAGE_MY = 2.0, 1.0     # 頁面左右邊界 2.0cm, 上下邊界 1.0cm
HEADER_W, HEADER_H = 15.4, 3.17 # 表頭圖片寬高 (cm)
BOX_W, BOX_H = 15.15, 4.85      # 岩心箱圖片寬高 (cm)
NUM_COL_W = 1.0                 # 深度編號欄位寬度 (cm)

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

def load_image(file_or_bytes, max_side=3200, portrait_dir="ccw"):
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
    return np.ascontiguousarray(np.rot90(img, (deg // 90) % 4))

def order_pts(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)

def _extreme_quad(hull):
    p = hull.reshape(-1, 2).astype(np.float32)
    sm_, df = p.sum(axis=1), p[:, 0] - p[:, 1]
    return np.array([p[np.argmin(sm_)], p[np.argmax(df)], p[np.argmax(sm_)], p[np.argmin(df)]], np.float32)

def _refine_quad(pts, quad, tol_frac=0.05):
    pts = pts.reshape(-1, 2).astype(np.float32)
    q = quad.astype(np.float32)
    short = min(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0]),
                np.linalg.norm(q[2] - q[1]), np.linalg.norm(q[3] - q[2]))
    tol = max(2.0, tol_frac * short)
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = b - a
        L = np.linalg.norm(d)
        u = d / L
        nrm = np.array([-u[1], u[0]])
        rel = pts - a
        t = rel @ u / L
        dist = np.abs(rel @ nrm)
        sel = pts[(dist < tol) & (t > 0.08) & (t < 0.92)]
        if len(sel) < 8:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    out = []
    for i in range(4):
        p1, d1 = lines[(i - 1) % 4]
        p2, d2 = lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return quad
        t = np.linalg.solve(A, p2 - p1)
        out.append(p1 + t[0] * d1)
    out = np.array(out, np.float32)
    if np.linalg.norm(out - q, axis=1).max() > 0.2 * short:
        return quad
    return out

def _blue_mask(sm):
    hsv = cv2.cvtColor(sm, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (85, 80, 60), (118, 255, 255))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

def detect_inner(img):
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
            if hier[0][idx][3] != -1:
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
    quad = _refine_quad(np.vstack([c.reshape(-1, 2) for c in grp]), quad)
    area = cv2.contourArea(quad)
    bad = (len(grp) < 3) or area < 0.05 * sh * sw
    bad = bad or bool(np.any(quad < 2) or np.any(quad[:, 0] > sw - 3) or np.any(quad[:, 1] > sh - 3))
    return order_pts(quad / sc), bool(bad)

def _warp_to(img, pts, out_w, out_h, inner):
    l, r, t, b = inner
    x0, x1 = out_w * l, out_w * (1 - r)
    y0, y1 = out_h * t, out_h * (1 - b)
    dst = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    M = cv2.getPerspectiveTransform(order_pts(pts), dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

def _crop_to_rim(img, inner, shrink=0.003):
    h, w = img.shape[:2]
    l, r, t, b = inner
    hsv = cv2.cvtColor(cv2.resize(img, (w // 2, h // 2)), cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, (85, 70, 50), (118, 255, 255)) > 0
    h2, w2 = m.shape
    x0, x1 = int(w2 * l), int(w2 * (1 - r))
    y0, y1 = int(h2 * t), int(h2 * (1 - b))
    colf = m[y0 + (y1 - y0) // 6: y1 - (y1 - y0) // 6, :].mean(axis=0)
    rowf = m[:, x0 + (x1 - x0) // 6: x1 - (x1 - x0) // 6].mean(axis=1)

    def walk(prof, start, step, limit, thr=0.3):
        i, n = start, 0
        while 0 <= i + step < len(prof) and n < limit:
            if prof[i + step] < thr:
                break
            i += step
            n += 1
        return i

    capx, capy = int(w2 * 0.045), int(h2 * 0.075)
    xl = walk(colf, x0, -1, capx)
    xr = walk(colf, x1, +1, capx)
    yt = walk(rowf, y0, -1, capy)
    yb = walk(rowf, y1, +1, capy)
    sx, sy = int(shrink * w), int(shrink * h)
    return img[max(0, yt * 2 + sy):min(h, yb * 2 - sy), max(0, xl * 2 + sx):min(w, xr * 2 - sx)]

def warp_inner(img, pts, out_w=2400, aspect=3.1, shrink=0.003, valid_rows=4, total_rows=4, is_manual=False):
    """
    影像透視校正：
    1. 若 is_manual=True，嚴禁二次校正，鎖定手動座標！
    2. 未滿一整箱（如 2m），依比例高度縮放，下方補滿底色，不強迫變形。
    """
    # ⚠️ 【參數調整區】內部裁切的安全邊距比例 (左右上下)
    big = (0.07, 0.07, 0.14, 0.14)
    cw = out_w
    ch = int(cw / 2.9)
    
    w1 = _warp_to(img, pts, cw, ch, big)
    
    # 🚨 關鍵除錯點：如果是手動套點，【禁止】執行第二階段偵測，保護原本準確的座標
    if not is_manual:
        p2, bad2 = detect_inner(w1)
        w2 = _warp_to(w1, p2, cw, ch, big) if (p2 is not None and not bad2) else w1
    else:
        w2 = w1
        
    crop = _crop_to_rim(w2, big, shrink)
    oh = int(out_w / aspect)
    
    # 🚨 關鍵除錯點：處理未滿一整箱 (例如 2m，即 2/4 列)，避免被拉成 full height
    if valid_rows < total_rows and total_rows > 0:
        actual_h = int(oh * (valid_rows / total_rows))
        crop_h, crop_w = crop.shape[:2]
        
        # 只取出有效列數的影像部分
        partial_crop = crop[:int(crop_h * (valid_rows / total_rows)), :]
        
        # 依照正確的比例縮放，不強行拉伸
        partial_res = cv2.resize(partial_crop, (out_w, actual_h), interpolation=cv2.INTER_AREA)
        
        # ⚠️ 【參數調整區】空箱部分的背景顏色 (B, G, R)。預設為純黑，如需改深藍可調為 (50, 40, 30)
        bg_color = (0, 0, 0)
        
        # 建立完整的箱子畫布，並將有效內容貼在最上方
        res = np.zeros((oh, out_w, 3), dtype=np.uint8)
        res[:, :] = bg_color
        res[:actual_h, :] = partial_res
    else:
        res = cv2.resize(crop, (out_w, oh), interpolation=cv2.INTER_AREA)
        
    return res

def draw_corners(img, pts, color=(255, 0, 0)):
    out = img.copy()
    p = np.array(pts, np.int32)
    t = max(3, img.shape[1] // 300)
    cv2.polylines(out, [p.reshape(-1, 1, 2)], True, color, t)
    for q in p:
        cv2.circle(out, tuple(int(v) for v in q), t * 3, (255, 255, 0), -1)
    return out

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
    im = Image.open(board if board else io.BytesIO(base64.b64decode(BOARD_B64))).convert("RGB")
    W, H = im.size
    fill = im.getpixel((int(W * 0.30), int(H * 0.64)))
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

BOARD_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBAUEBAYFBQUGBgYHCQ4JCQgICRINDQoOFRIWFhUSFBQXGiEcFxgfGRQUHScdHyIj"
    "JSUlFhwpLCgkKyEkJST/2wBDAQYGBgkICREJCREkGBQYJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQk"
    "JCQkJCQkJCT/wAARCAFIBkADASIAAhEBAxEB/8QAHAAAAQQDAQAAAAAAAAAAAAAAAQACBgcDBAUI/8QAXhAAAQIEAwMFCAgRCwME"
    "AwADAQIDAAQFEQYHIRIxQRNRYXHRFBUWIoGRk9IXMjWSlKGxwSMkJSYzNDZCRVJicnOCg6KyCENEU1RVY3SE4eJGVmSjs8LwGKTx"
    "J2WV/8QAGwEBAQEBAQEBAQAAAAAAAAAAAAECAwQFBgf/xAA4EQACAQIFAgUDAwMEAgIDAAAAAQIDEQQSITFRE1IUMkFhoSIzkQUj"
    "cYGx8BVCU2IGJMHxNEPh/9oADAMBAAIRAxEAPwCeTr9JpC0zE46wwVqsNNVGA1mJhVCCnvq2CneOTV2RhxDRKZUp5pNUlTMtoIAR"
    "tEC54m0bkvl3hRtIU1RZVN+N1a/HHWMYuN5M43ZnbxxQHGwpM8FAi/2NXZD28bUJ1CiidBsbHxFD5oy+B9BCbd7GCBwN+2CnCVCb"
    "QQKXL2JuRY9sVqn7luzWaxtQ3XghM4Su27k1CHO44obNtqbVvto2oxsN4aozQ2UUyWSOPiwFYYojh8alyx57p/3haHuLs13MbUTU"
    "91KL7NUNTjmgqZDiZpezz8kqN4YYoljamS1ubYgqwzRnANumSqutEGqfuLs57OPKAvb2ZtZ2d/0IxiVmNh5LoR3S8VE2ADKjHSbw"
    "nQkFRTSZNO1vs3vheCdB2wrvRJXB0PJwtT9yXZzncx8PtgqVMTFh/gmEMxKCW0rDkzZQv9hPbHScwtRFE7VIk19BbEZBhqihCbUq"
    "TAAsByY0han7i7OQzmNQHtoIcmiR/g/7wxeZmHmlhC1zalHQAM3+eOy3hiiAKtSZEE77NCG+DVF5UHvVIAjceRTeFqfuW7Oc/mFQ"
    "2kbalTOyRf7F/vDDmNQthK/pqxHFodsdlzDtGWkbdLkldbSYf4O0gJAFLkQBw5FMTLT9xdkfZzNoLtwjuwkHX6ENPjgJzRoHLBu0"
    "2FH/AAh2xIG8O0hKTs0uRFzwYSPmgpw9SEruKXJAjjyCeyLateral7jUjb+amH21bJTOqPMGgfnhy806EhAUWpwA7hyYv8sSNVDpQ"
    "NxTJG448gnsh5o1NKT9TpO/PyKeyH7fuT6iMN5q0NxClIYnthO9RbSB8sMbzVoi3AhDE6tRNrBKe2JSKPIkAqkpU20tyKeyHt0qQ"
    "QvaRIyqTbeGUj5on7fuX6iIP5s0NpWyticJvuAT2xrTuZdDfb2jLTiCbcE3+WJuqmSJP2jK+VlPZD1U6TIAMnLaDi0nshalwxeRA"
    "JSLQpगरणp7I+SLUHsc8o3d4e7X44u8pCEFqUIA7k7ofxT8sfNfWc8k3cKk27j0R5TffG4m141q6YJ3wR3w+Y3Hwg05eT1d0E74b"
    "w83iF6w83iFvXjA3iL0N98PmhjLzQ74eXhA3w83iAW88G4O7x80C34wfNAK8oF+mEbwQ3iAGo03Q1B4wnm5oV7zAMJ7/AO6Ebw3n"
    "Cj5oAvfzhX4wvNAv0wo3hAW88A1O8wgveG8M83iFfeLwAbxQz3gw3hLTeMBe8MLTeMFrO7TDBvjBDeMIG+EbwjA3hLzeIBe8MLW8"
    "ILV5oRveMLW8IMTveAGveDFvfCHveDDeL4AN4RgvzhS17c8MW/n44BvGFrN4QLXjCmvLwA3i+MFveDFvO8G8oBDeLwwN4wQvCmv"
    "5oAbxgB4wQrDeG8rxwBivbwhX6YQbzCgvAAbwYt54cTzwhreG8M83iAXhjeGFvGrN4QtrN4QFvOAfNhS3iCneDBvN4wA88G4wN4"
    "d5oQuMAdN4RveG8MO7ghTeG8M83iAXveDDW8MLTeCneG8M83iAMeKAc7obzQ0mFe8MLeG8AbeGEdMOvDeHcvAAvGGrzeGGtvLzQ"
    "0m8K95oAN7w0mGe8MMbwhneCne8ILV74QvvDCNd5hCveaACvvhX6oaFeGEe8MDeEFe8MLWe8w0vN4ga35pAsf/Z"
)
