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

pdfmetrics.registerFont(UnicodeCIDFont("MSung-Light"))

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 版面（依範例 Word 量測：A4、左右 2cm、上下 1cm）----
PAGE_MX, PAGE_MY = 2.0, 1.0
HEADER_W, HEADER_H = 15.4, 3.17
BOX_W, BOX_H = 15.15, 4.85
NUM_COL_W = 1.0
FULL_ASPECT = 3.12  # 滿箱成果圖寬/高（= BOX_W/BOX_H，範例 Word 的顯示比例）

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


def _refine_quad(pts, quad, tol_frac=0.05):
    """四邊各自用直線擬合（Huber），再取相鄰兩線交點 → 比凸包角更準、更不歪"""
    pts = pts.reshape(-1, 2).astype(np.float32)
    q = quad.astype(np.float32)  # tl,tr,br,bl
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
    
    def ang(d):
        return np.arctan2(d[1], d[0])

    def lr_dir(d):  
        return d if d[1] >= 0 else -d

    top_d, bot_d = lines[0][1], lines[2][1]
    top_d = top_d if top_d[0] >= 0 else -top_d
    bot_d = bot_d if bot_d[0] >= 0 else -bot_d
    base = np.arctan2(top_d[1] + bot_d[1], top_d[0] + bot_d[0])  
    perp = base + np.pi / 2
    for idx in (1, 3):
        p0, d0 = lines[idx]
        d0 = lr_dir(d0)
        delta = (ang(d0) - perp + np.pi) % (2 * np.pi) - np.pi
        delta = float(np.clip(delta, -np.deg2rad(2.0), np.deg2rad(2.0)))
        a2 = perp + delta
        lines[idx] = (p0, np.array([np.cos(a2), np.sin(a2)]))
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


def _extend_missing(quad, full_aspect=3.1, thr=1.25):
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if h <= 0 or w / h < full_aspect * thr:
        return quad
    k = (w / full_aspect - h) / h
    return np.array([tl, tr, br + (br - tr) * k, bl + (bl - tl) * k], np.float32)


def _blue_mask(sm):
    hsv = cv2.cvtColor(sm, cv2.COLOR_RGB2HSV)
    # 擴大藍色的偵測範圍 (抵抗污泥與陰影造成的色偏)
    mask = cv2.inRange(hsv, (80, 40, 40), (130, 255, 255))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _detect_holes(img, full_aspect=3.1):
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
    quad = order_pts(quad)
    quad = _extend_missing(quad, full_aspect)
    area = cv2.contourArea(quad)
    bad = area < 0.05 * sh * sw
    bad = bad or bool(np.any(quad < 2) or np.any(quad[:, 0] > sw - 3) or np.any(quad[:, 1] > sh - 3))
    return order_pts(quad / sc), bool(bad)


# 大幅縮減 TRAY_INSET，避免咬進岩心（改為左右縮 0.5%，上下縮 1%）
TRAY_INSET = (0.005, 0.005, 0.01, 0.01)  # 左、右、上、下
MARGIN = (0.004, 0.006)                    


def _warp_to(img, pts, out_w, out_h, inner):
    l, r, t, b = inner
    x0, x1 = out_w * l, out_w * (1 - r)
    y0, y1 = out_h * t, out_h * (1 - b)
    dst = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    M = cv2.getPerspectiveTransform(order_pts(pts), dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def _walk_out(line, start, step, cap, gap=6, find=40):
    n = len(line)
    i, k = start, 0
    while 0 <= i < n and not line[i] and k < find:
        i += step
        k += 1
    if not (0 <= i < n and line[i]):
        return None
    last, miss, k = i, 0, 0
    while 0 <= i + step < n and k < cap:
        i += step
        k += 1
        if line[i]:
            last, miss = i, 0
        else:
            miss += 1
            if miss >= gap:
                break
    return last


def _fit_line_robust(pts):
    pts = np.array(pts, np.float32)
    if len(pts) < 12:
        return None
    for _ in range(2):
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        nrm = np.array([-vy, vx])
        res = np.abs((pts - np.array([x0, y0])) @ nrm)
        keep = res <= max(1.5, 2.5 * np.median(res) + 1)
        if keep.sum() < 12:
            break
        pts = pts[keep]
    return np.array([x0, y0]), np.array([vx, vy])


def _outer_quad(w, big):
    h, wd = w.shape[:2]
    hsv = cv2.cvtColor(cv2.resize(w, (wd // 2, h // 2)), cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, (80, 40, 40), (130, 255, 255)) > 0
    h2, w2 = m.shape
    l, r, t, b = big
    x0, x1 = int(w2 * l), int(w2 * (1 - r))
    y0, y1 = int(h2 * t), int(h2 * (1 - b))
    capx, capy = int(w2 * 0.06), int(h2 * 0.11)
    L, R, T, B = [], [], [], []
    for y in range(y0 + (y1 - y0) // 8, y1 - (y1 - y0) // 8, 2):
        a = _walk_out(m[y], x0, -1, capx)
        if a is not None:
            L.append((a, y))
        a = _walk_out(m[y], x1, +1, capx)
        if a is not None:
            R.append((a, y))
    for x in range(x0 + (x1 - x0) // 8, x1 - (x1 - x0) // 8, 2):
        a = _walk_out(m[:, x], y0, -1, capy)
        if a is not None:
            T.append((x, a))
        a = _walk_out(m[:, x], y1, +1, capy)
        if a is not None:
            B.append((x, a))
    lines = [_fit_line_robust(T), _fit_line_robust(R), _fit_line_robust(B), _fit_line_robust(L)]
    if any(ln is None for ln in lines):
        return None
    q = []
    for i in range(4):
        p1, d1 = lines[(i - 1) % 4]
        p2, d2 = lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        tt = np.linalg.solve(A, p2 - p1)
        q.append(p1 + tt[0] * d1)
    return np.array(q, np.float32) * 2


def detect_inner(img, full_aspect=3.1):
    ph, bad = _detect_holes(img, full_aspect)
    if ph is None:
        return None, True
    cw = 2400
    ch = int(cw / 2.9)
    big = (0.07, 0.07, 0.14, 0.14)
    l, r, t, b = big
    dst = np.array([[cw * l, ch * t], [cw * (1 - r), ch * t],
                    [cw * (1 - r), ch * (1 - b)], [cw * l, ch * (1 - b)]], np.float32)
    M1 = cv2.getPerspectiveTransform(order_pts(ph), dst)
    w1 = cv2.warpPerspective(img, M1, (cw, ch), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    oq = _outer_quad(w1, big)
    if oq is None:
        return ph, True
    
    ow = (np.linalg.norm(oq[1] - oq[0]) + np.linalg.norm(oq[2] - oq[3])) / 2
    oh = (np.linalg.norm(oq[3] - oq[0]) + np.linalg.norm(oq[2] - oq[1])) / 2
    inner_w, inner_h = cw * (1 - l - r), ch * (1 - t - b)
    # 放寬長寬比檢查 (0.8~1.6)，讓嚴重傾斜的照片也能順利利用外框校正，不再輕易回退
    if not (0.8 < ow / inner_w < 1.45 and 0.8 < oh / inner_h < 1.6):
        return ph, True
    H = cv2.getPerspectiveTransform(oq, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
    fl, fr, ft, fb = TRAY_INSET
    inn = np.float32([[fl, ft], [1 - fr, ft], [1 - fr, 1 - fb], [fl, 1 - fb]])
    p_w1 = cv2.perspectiveTransform(inn.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    p = cv2.perspectiveTransform(p_w1.reshape(-1, 1, 2), np.linalg.inv(M1)).reshape(-1, 2)
    p = order_pts(p)
    hh, ww = img.shape[:2]
    bad = bool(bad or np.any(p[:, 0] < 0) or np.any(p[:, 1] < 0)
               or np.any(p[:, 0] > ww - 1) or np.any(p[:, 1] > hh - 1))
    return p, bad


def warp_points(img, pts, out_w=2400, aspect=3.12, margin=MARGIN):
    mx, my = margin
    out_h = int(out_w / aspect)
    x0, x1 = out_w * mx, out_w * (1 - mx)
    y0, y1 = out_h * my, out_h * (1 - my)
    dst = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    M = cv2.getPerspectiveTransform(order_pts(pts), dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def detect_filled_rows(img, rows_per_box=4):
    """自動偵測最後一箱有幾列實際裝有岩心（計算裁切後各列的藍色佔比）"""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (80, 50, 40), (130, 255, 255))
    row_h = h // rows_per_box
    filled_rows = rows_per_box
    
    for i in range(rows_per_box):
        slice_mask = mask[i*row_h:(i+1)*row_h, :]
        blue_ratio = np.sum(slice_mask > 0) / slice_mask.size
        # 若該列的藍底像素超過 22%，判定為空槽
        if blue_ratio > 0.22:
            filled_rows = i
            break
            
    return filled_rows if filled_rows > 0 else rows_per_box


def crop_partial(im, k, n=4, margin=MARGIN):
    if not k or k >= n:
        return im
    mx, my = margin
    H = im.height
    cut = (my + (1 - 2 * my) * k / n + 0.010) * H
    return im.crop((0, 0, im.width, int(min(H, cut))))


def draw_corners(img, pts, color=(255, 0, 0)):
    out = img.copy()
    p = order_pts(pts).astype(np.int32)
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
        d.
