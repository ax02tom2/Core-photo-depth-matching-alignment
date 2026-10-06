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
    y_sorted = pts[np.argsort(pts[:, 1]), :]
    top = y_sorted[:2, :]
    bottom = y_sorted[2:, :]
    tl = top[np.argmin(top[:, 0]), :]
    tr = top[np.argmax(top[:, 0]), :]
    bl = bottom[np.argmin(bottom[:, 0]), :]
    br = bottom[np.argmax(bottom[:, 0]), :]
    return np.array([tl, tr, br, bl], dtype=np.float32)

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
        delta = float(np.clip(delta, -np.deg2rad(15.0), np.deg2rad(15.0)))
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
    mask = cv2.inRange(hsv, (85, 80, 60), (118, 255, 255))
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

TRAY_INSET = (0.021, 0.022, 0.061, 0.074)  
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
    m = cv2.inRange(hsv, (85, 70, 50), (118, 255, 255)) > 0
    h2, w2 = m.shape
    l, r, t, b = big
    x0, x1 = int(w2 * l), int(w2 * (1 - r))
    y0, y1 = int(h2 * t), int(h2 * (1 - b))
    capx, capy = int(w2 * 0.06), int(h2 * 0.11)
    L, R, T, B = [], [], [], []
    for y in range(y0 + (y1 - y0) // 8, y1 - (y1 - y0) // 8, 2):
        a = _walk_out(m[y], x0, -1, capx)
        if a is not None: L.append((a, y))
        a = _walk_out(m[y], x1, +1, capx)
        if a is not None: R.append((a, y))
    for x in range(x0 + (x1 - x0) // 8, x1 - (x1 - x0) // 8, 2):
        a = _walk_out(m[:, x], y0, -1, capy)
        if a is not None: T.append((x, a))
        a = _walk_out(m[:, x], y1, +1, capy)
        if a is not None: B.append((x, a))
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

def detect_inner(img, full_aspect=3.1, inset_adj=0.0):
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
        
    H = cv2.getPerspectiveTransform(oq, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
    fl, fr, ft, fb = TRAY_INSET
    
    fl = max(0.0, fl - inset_adj)
    fr = max(0.0, fr - inset_adj)
    ft = max(0.0, ft - inset_adj)
    fb = max(0.0, fb - inset_adj)
    
    inn = np.float32([[fl, ft], [1 - fr, ft], [1 - fr, 1 - fb], [fl, 1 - fb]])
    p_w1 = cv2.perspectiveTransform(inn.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    p = cv2.perspectiveTransform(p_w1.reshape(-1, 1, 2), np.linalg.inv(M1)).reshape(-1, 2)
    p = order_pts(p)

    # ---------------- 修正重點：極度嚴格的幾何異常檢測 ----------------
    w1_len = np.linalg.norm(p[1] - p[0])
    w2_len = np.linalg.norm(p[2] - p[3])
    h1_len = np.linalg.norm(p[3] - p[0])
    h2_len = np.linalg.norm(p[2] - p[1])
    avg_w = (w1_len + w2_len) / 2.0
    avg_h = (h1_len + h2_len) / 2.0
    
    hh, ww = img.shape[:2]
    img_area = hh * ww
    frame_area = avg_w * avg_h
    
    if avg_h == 0 or avg_w == 0:
        bad = True
    else:
        ratio = avg_w / avg_h
        # 1. 嚴格長寬比例檢測 (標準為 3.12，若框到其他區域會變太胖或太扁)
        if ratio < 2.6 or ratio > 3.6: 
            bad = True
        
        # 2. 面積比例檢測 (紅框佔據整張照片的比例，過大過小皆異常)
        if frame_area / img_area < 0.35 or frame_area / img_area > 0.95:
            bad = True
            
        # 3. 嚴格梯形變形檢測 (上下、左右邊長落差不得大於 8%)
        if abs(w1_len - w2_len) > avg_w * 0.08: bad = True
        if abs(h1_len - h2_len) > avg_h * 0.08: bad = True
        
        # 4. 內角角度檢測 (四個角應接近 90 度)
        for i in range(4):
            v1 = p[i] - p[(i+1)%4]
            v2 = p[(i+2)%4] - p[(i+1)%4]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                if abs(np.dot(v1, v2) / (n1 * n2)) > 0.20:
                    bad = True

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

def crop_partial(im, k, n=4, margin=MARGIN):
    if not k or k >= n:
        return im
    mx, my = margin
    H = im.height
    cut = (my + (1 - 2 * my) * k / n + 0.010) * H
    return im.crop((0, 0, im.width, int(min(H, cut))))

def auto_detect_filled_rows(img, total_rows=4):
    try:
        h, w = img.shape[:2]
        if h == 0 or w == 0: return int(total_rows)
        small = cv2.resize(img, (max(1, w // 4), max(1, h // 4)))
        sh, sw = small.shape[:2]
        s_row_h = sh // total_rows
        if s_row_h == 0: return int(total_rows)
        
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        filled_count = 0
        for i in range(total_rows):
            strip = hsv[i * s_row_h:(i + 1) * s_row_h, :]
            area = strip.shape[0] * strip.shape[1]
            if area == 0: continue
            
            blue_mask = cv2.inRange(strip, (85, 50, 50), (125, 255, 255))
            blue_ratio = np.sum(blue_mask > 0) / area
            
            if blue_ratio < 0.40:
                filled_count = i + 1
                
        return max(1, int(filled_count))
    except Exception:
        return int(total_rows)

def draw_corners(img, pts, color=(255, 0, 0)):
    out = img.copy()
    p = order_pts(pts).astype(np.int32)
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

# ---------------- 修正重點：完整恢復 Base64 編碼，解決空白表頭 ----------------
def make_header(hole, depth, date, project, board=None):
    im = None
    if board is not None:
        try:
            im = Image.open(io.BytesIO(board)).convert("RGB")
        except Exception:
            pass
            
    if im is None:
        # 直接讀取完整的 BOARD_B64
        im = Image.open(io.BytesIO(base64.b64decode(BOARD_B64))).convert("RGB")
            
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

def _text_img(text, size):
    fp = next((p for p in SERIF_PATHS if os.path.exists(p)), None)
    if not fp:
        return None
    f = _font(size)
    bb = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=f)
    im = Image.new("RGB", (bb[2] - bb[0] + 20, bb[3] - bb[1] + 20), "white")
    ImageDraw.Draw(im).text((10 - bb[0], 10 - bb[1]), text, font=f, fill="black")
    return im

def _jpeg(im, q=88):
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=q)
    b.seek(0)
    return b

def _pages(boxes, rows_per_box, per_page, start_depth, row_m, total_rows=0):
    total_rows = total_rows or len(boxes) * rows_per_box
    bpp = int(per_page // rows_per_box)
    if bpp == 0: bpp = 1  
    for p in range(0, len(boxes), bpp):
        chunk = boxes[p:p + bpp]
        r0 = p * rows_per_box
        n = min(len(chunk) * rows_per_box, total_rows - r0)
        d0 = start_depth + r0 * row_m
        yield chunk, r0, n, f"{d0}~{d0 + n * row_m}m"

def build_pdf(boxes, hole, date, project, start_depth=0, rows_per_box=4,
              per_page=20, row_m=1, board=None, end_mark=True, total_rows=0):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    x0 = PAGE_MX * cm
    for chunk, r0, n, depth in _pages(boxes, rows_per_box, per_page, start_depth, row_m, total_rows):
        y = H - PAGE_MY * cm
        hi = make_header(hole, depth, date, project, board)
        c.drawImage(ImageReader(_jpeg(hi, 92)), x0, y - HEADER_H * cm, HEADER_W * cm, HEADER_H * cm)
        y -= HEADER_H * cm
        for i, im in enumerate(chunk):
            h_cm = BOX_H * (im.height * FULL_ASPECT / im.width)   
            c.drawImage(ImageReader(_jpeg(im)), x0, y - h_cm * cm, BOX_W * cm, h_cm * cm)
            c.setFont("Helvetica", 16)
            mx_, my_ = MARGIN
            for k in range(rows_per_box):
                num = r0 + i * rows_per_box + k + 1
                if total_rows and num > total_rows:
                    break
                frac = my_ + (1 - 2 * my_) * (k + 0.5) / rows_per_box   
                c.drawString(x0 + (HEADER_W + 0.3) * cm, y - BOX_H * frac * cm - 5.5, str(num))
            y -= h_cm * cm
        if r0 + n >= (total_rows or len(boxes) * rows_per_box) and end_mark:
            ti = _text_img("鑽探結束", 60)
            if ti is not None:  
                tw = 2.6 * cm
                c.drawImage(ImageReader(ti), W / 2 - tw / 2, y - 1.3 * cm, tw, tw * ti.height / ti.width)
            else:
                c.setFont("MSung-Light", 10.5)
                c.drawCentredString(W / 2, y - 0.8 * cm, "鑽探結束")
        c.showPage()
    c.save()
    return buf.getvalue()

def build_docx(boxes, hole, date, project, start_depth=0, rows_per_box=4,
               per_page=20, row_m=1, board=None, end_mark=True, total_rows=0):
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
    for chunk, r0, n, depth in _pages(boxes, rows_per_box, per_page, start_depth, row_m, total_rows):
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
            h_cm = BOX_H * (im.height * FULL_ASPECT / im.width)
            row.height, row.height_rule = Cm(h_cm), WD_ROW_HEIGHT_RULE.EXACTLY
            for j, wd in enumerate(widths):
                row.cells[j].width = wd
            p = row.cells[0].paragraphs[0]
            tight(p)
            p.add_run().add_picture(_jpeg(im), width=Cm(BOX_W), height=Cm(h_cm - 0.05))
            cell = row.cells[1]
            for k in range(rows_per_box):
                p = cell.paragraphs[0] if k == 0 else cell.add_paragraph()
                tight(p, rh / 2.54 * 72)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                num = r0 + i * rows_per_box + k + 1
                r = p.add_run(str(num) if (not total_rows or num <= total_rows) else "")
                r.font.size = Pt(16)
    if end_mark:
        p = doc.add_paragraph()
        tight(p)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run("鑽探結束").font.size = Pt(10.5)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()

# 原封不動加回的 Base64 預設表頭照片
BOARD_B64 = (
    "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"
)
