import io
import re
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

pdfmetrics.registerFont(UnicodeCIDFont("MSung-Light"))  # 繁中內建字型，不需字型檔
FONT = "MSung-Light"
FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "C:/Windows/Fonts/msjh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


def find_font():
    return next((p for p in FONT_PATHS if os.path.exists(p)), None)


def header_image(company, project, hole, date, depth, boxes, w=1600, rh=90):
    """表頭轉成圖片（內嵌，避免 PDF 閱讀器缺中文字型）；找不到字型回傳 None"""
    fp = find_font()
    if not fp:
        return None
    big, mid = ImageFont.truetype(fp, 52), ImageFont.truetype(fp, 36)
    im = Image.new("RGB", (w, rh * 3), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w - 1, rh * 3 - 1], outline="black", width=3)
    d.line([0, rh, w, rh], fill="black", width=3)
    d.line([0, rh * 2, w, rh * 2], fill="black", width=3)
    d.line([w // 2, rh * 2, w // 2, rh * 3], fill="black", width=3)
    tw = d.textlength(company, font=big)
    d.text(((w - tw) / 2, rh // 2 - 30), company, font=big, fill="black")
    d.text((20, rh + 22), "工程名稱：" + project, font=mid, fill="black")
    d.text((20, rh * 2 + 22), f"孔號：{hole}   日期：{date}", font=mid, fill="black")
    d.text((w // 2 + 20, rh * 2 + 22), f"深度：{depth}   箱號：{boxes}", font=mid, fill="black")
    return im


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_image(file_or_bytes, max_side=3200):
    im = Image.open(file_or_bytes)
    im = ImageOps.exif_transpose(im).convert("RGB")
    if max(im.size) > max_side:
        r = max_side / max(im.size)
        im = im.resize((int(im.width * r), int(im.height * r)), Image.LANCZOS)
    return np.array(im)


def order_pts(pts):
    """排序為 左上、右上、右下、左下"""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def detect_box(img):
    """用藍色岩心箱自動找四個角點；失敗回傳 None"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (85, 90, 70), (115, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.05 * img.shape[0] * img.shape[1]:
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
    else:
        pts = cv2.boxPoints(cv2.minAreaRect(hull))
    return order_pts(pts)


def warp(img, pts, aspect=None, out_w=3000):
    """透視校正。aspect = 寬/高；None 則由角點估算"""
    tl, tr, br, bl = order_pts(pts)
    if aspect is None:
        w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        aspect = w / h
    out_h = int(out_w / aspect)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC)


def split_rows(box_img, n_rows=4, left=0.03, right=0.04, top=0.09, bottom=0.08):
    """把校正後的整箱切成 n_rows 列（每列 = 1 個箱號）"""
    h, w = box_img.shape[:2]
    x0, x1 = int(w * left), int(w * (1 - right))
    y0, y1 = int(h * top), int(h * (1 - bottom))
    inner = box_img[y0:y1, x0:x1]
    rh = inner.shape[0] / n_rows
    return [Image.fromarray(inner[int(i * rh):int((i + 1) * rh)]) for i in range(n_rows)]


def draw_corners(img, pts, color=(255, 0, 0)):
    out = img.copy()
    p = np.array(pts, np.int32)
    t = max(3, img.shape[1] // 300)
    cv2.polylines(out, [p.reshape(-1, 1, 2)], True, color, t)
    for q in p:
        cv2.circle(out, tuple(int(v) for v in q), t * 3, (255, 255, 0), -1)
    return out


def build_pdf(rows, company, project, hole, date, start_depth=0, per_page=20, row_m=1):
    """rows: list[PIL.Image]，依箱號順序。回傳 PDF bytes"""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    mx = 36
    n_pages = (len(rows) + per_page - 1) // per_page
    for p in range(n_pages):
        chunk = rows[p * per_page:(p + 1) * per_page]
        d0 = start_depth + p * per_page * row_m
        d1 = d0 + len(chunk) * row_m
        # ---- 表頭 ----
        top = H - 30
        hh = 22
        hi = header_image(company, project, hole, date, f"{d0}~{d1}m",
                          f"{p * per_page + 1}-{p * per_page + len(chunk)}")
        if hi is not None:
            b = io.BytesIO()
            hi.save(b, "PNG")
            b.seek(0)
            c.drawImage(ImageReader(b), mx, top - hh * 3, W - 2 * mx, hh * 3)
        else:  # 無字型時退回 PDF 內建繁中字型
            c.setLineWidth(0.8)
            c.rect(mx, top - hh * 3, W - 2 * mx, hh * 3)
            c.line(mx, top - hh, W - mx, top - hh)
            c.line(mx, top - hh * 2, W - mx, top - hh * 2)
            c.setFont(FONT, 15)
            c.drawCentredString(W / 2, top - hh + 6, company)
            c.setFont(FONT, 10)
            c.drawString(mx + 6, top - hh * 2 + 7, "工程名稱：" + project)
            c.line(W / 2, top - hh * 3, W / 2, top - hh * 2)
            c.drawString(mx + 6, top - hh * 3 + 7, f"孔號：{hole}    日期：{date}")
            c.drawString(W / 2 + 6, top - hh * 3 + 7, f"深度：{d0}~{d1}m    箱號：{p * per_page + 1}-{p * per_page + len(chunk)}")
        # ---- 岩心列 ----
        y_top = top - hh * 3 - 10
        avail = y_top - 40
        slot = avail / per_page
        max_w = W - 2 * mx - 28
        for i, im in enumerate(chunk):
            asp = im.width / im.height
            rh = slot * 0.92
            rw = rh * asp
            if rw > max_w:
                rw, rh = max_w, max_w / asp
            y = y_top - (i + 1) * slot + (slot - rh) / 2
            buf_im = io.BytesIO()
            im.convert("RGB").save(buf_im, "JPEG", quality=88)
            buf_im.seek(0)
            c.drawImage(ImageReader(buf_im), mx, y, rw, rh)
            c.setFont("Helvetica", 10)
            c.drawString(W - mx - 22, y + rh / 2 - 3, str(p * per_page + i + 1))
        if p == n_pages - 1:
            c.setFont(FONT, 10)
            c.drawCentredString(W / 2, 22, "鑽探結束")
        c.showPage()
    c.save()
    return buf.getvalue()
