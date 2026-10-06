CORE_VERSION = "2.1"
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
    # 左右兩邊（端壁有階梯，點位散）→ 方向限制在「垂直於上下邊」±2° 內，避免整張剪切歪斜
    def ang(d):
        return np.arctan2(d[1], d[0])

    def lr_dir(d):  # 讓方向朝下，便於比較
        return d if d[1] >= 0 else -d

    top_d, bot_d = lines[0][1], lines[2][1]
    top_d = top_d if top_d[0] >= 0 else -top_d
    bot_d = bot_d if bot_d[0] >= 0 else -bot_d
    base = np.arctan2(top_d[1] + bot_d[1], top_d[0] + bot_d[0])  # 上下邊平均方向
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
    if np.linalg.norm(out - q, axis=1).max() > 0.2 * short:  # 擬合發散就不用
        return quad
    return out


def _extend_missing(quad, full_aspect=3.1, thr=1.25):
    """整箱 4 槽的內框寬高比約 3.1。若量到的框比這扁很多，代表下面幾槽是空的（藍色槽底
    不會形成洞，沒被偵測到），依岩心由上往下放的慣例把框往下補足，不要把 2 槽硬拉成 4 槽。"""
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
    quad = _refine_quad(np.vstack([c.reshape(-1, 2) for c in grp]), quad)
    quad = order_pts(quad)
    quad = _extend_missing(quad, full_aspect)
    area = cv2.contourArea(quad)
    bad = area < 0.05 * sh * sw
    bad = bad or bool(np.any(quad < 2) or np.any(quad[:, 0] > sw - 3) or np.any(quad[:, 1] > sh - 3))
    return order_pts(quad / sc), bool(bad)


# 使用者手點的四個點 = 岩心箱「內側四角」。量測：這四點相對於箱子外緣（藍色箱緣最外側）
# 的位置，左 2.1%、右 2.2%、上 6.1%、下 7.4%（四個角彼此一致，與範例 Word 的裁切也吻合）。
TRAY_INSET = (0.021, 0.022, 0.061, 0.074)  # 左、右、上、下（佔外框寬/高的比例）
MARGIN = (0.004, 0.006)                    # 成果圖內框外側多留的邊（x, y 佔成果圖比例）


def _warp_to(img, pts, out_w, out_h, inner):
    l, r, t, b = inner
    x0, x1 = out_w * l, out_w * (1 - r)
    y0, y1 = out_h * t, out_h * (1 - b)
    dst = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    M = cv2.getPerspectiveTransform(order_pts(pts), dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def _walk_out(line, start, step, cap, gap=6, find=40):
    """從 start 往外找到第一個藍色，再沿藍色走到箱子外緣（容許小缺口）；找不到回傳 None"""
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
    """在（已大致拉正的）圖上，沿藍色箱緣找出箱子最外緣四條邊，再取交點。
    外緣是長直線，比端壁階梯、隔板的內角穩定，所以歪斜可以在這裡一次修正。"""
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


def _inner_quad_from_blue(w1, outer_q, expected=TRAY_INSET):
    """直接找藍色箱壁「外側到內側」的第一個藍色帶之內緣。

    TRAY_INSET 只用來限制搜尋位置；最後角點由藍色箱壁的實際內緣決定。
    由外往內掃描並取藍色帶的最後一個像素，可避免把箱內的藍色反光/
    隔板誤當成岩心槽內角。
    """
    h, w = w1.shape[:2]
    hsv = cv2.cvtColor(w1, cv2.COLOR_RGB2HSV)
    blue = cv2.inRange(hsv, (85, 70, 50), (118, 255, 255)) > 0

    tl, tr, br, bl = outer_q
    xmin, xmax = sorted((float(tl[0]), float(tr[0])))
    ymin, ymax = sorted((float(tl[1]), float(bl[1])))
    fl, fr, ft, fb = expected

    ex_t = (tl[1] + tr[1]) / 2 + ft * ((bl[1] + br[1] - tl[1] - tr[1]) / 2)
    ex_b = (bl[1] + br[1]) / 2 - fb * ((bl[1] + br[1] - tl[1] - tr[1]) / 2)
    ex_l = (tl[0] + bl[0]) / 2 + fl * ((tr[0] + br[0] - tl[0] - bl[0]) / 2)
    ex_r = (tr[0] + br[0]) / 2 - fr * ((tr[0] + br[0] - tl[0] - bl[0]) / 2)

    def last_of_first_blue_run(line, start, stop, step):
        i = int(start)
        stop = int(stop)
        while 0 <= i < len(line) and ((i <= stop) if step > 0 else (i >= stop)):
            if line[i]:
                # 沿「外 -> 內」走完整個藍色箱壁，取最後一點 = 內緣。
                while 0 <= i + step < len(line) and (
                    (i + step <= stop) if step > 0 else (i + step >= stop)
                ) and line[i + step]:
                    i += step
                return i
            i += step
        return None

    search = 0.10
    tol = 0.11
    top, right, bottom, left = [], [], [], []

    # 上/下：由外框往箱內掃。
    for x in np.linspace(xmin + .10 * (xmax - xmin),
                          xmax - .10 * (xmax - xmin), 140).astype(int):
        y = last_of_first_blue_run(blue[:, x], ymin + 2, ex_t + search * h, +1)
        if y is not None and abs(y - ex_t) <= tol * h:
            top.append((x, y))

        y = last_of_first_blue_run(blue[:, x], ymax - 2, ex_b - search * h, -1)
        if y is not None and abs(y - ex_b) <= tol * h:
            bottom.append((x, y))

    # 左/右：由外框往箱內掃。
    for y in np.linspace(ymin + .12 * (ymax - ymin),
                         ymax - .12 * (ymax - ymin), 140).astype(int):
        x = last_of_first_blue_run(blue[y, :], xmin + 2, ex_l + search * w, +1)
        if x is not None and abs(x - ex_l) <= tol * w:
            left.append((x, y))

        x = last_of_first_blue_run(blue[y, :], xmax - 2, ex_r - search * w, -1)
        if x is not None and abs(x - ex_r) <= tol * w:
            right.append((x, y))

    lines = [_fit_line_robust(top), _fit_line_robust(right),
             _fit_line_robust(bottom), _fit_line_robust(left)]
    if any(x is None for x in lines):
        return None

    out = []
    for i in range(4):
        p1, d1 = lines[(i - 1) % 4]
        p2, d2 = lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        t = np.linalg.solve(A, p2 - p1)
        out.append(p1 + t[0] * d1)

    q = order_pts(np.array(out, np.float32))
    ew = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    eh = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    ow = (np.linalg.norm(outer_q[1] - outer_q[0]) +
          np.linalg.norm(outer_q[2] - outer_q[3])) / 2
    oh = (np.linalg.norm(outer_q[3] - outer_q[0]) +
          np.linalg.norm(outer_q[2] - outer_q[1])) / 2
    if not (0.70 * ow < ew < 0.99 * ow and 0.65 * oh < eh < 0.99 * oh):
        return None
    if np.any(q[:, 0] < xmin) or np.any(q[:, 0] > xmax):
        return None
    if np.any(q[:, 1] < ymin) or np.any(q[:, 1] > ymax):
        return None
    return q



# ------------------------------------------------------------------ 自動箱框 / 內角偵測（v2.2）
def _blue_mask_strong(img):
    """較寬容的藍色箱體遮罩；岩心箱的藍色本身就是最穩定的幾何線索。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, (78, 55, 40), (128, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def _line_intersection(a, da, b, db):
    A = np.array([da, -db], np.float32).T
    if abs(float(np.linalg.det(A))) < 1e-6:
        return None
    t = np.linalg.solve(A, np.asarray(b, np.float32) - np.asarray(a, np.float32))
    return np.asarray(a, np.float32) + float(t[0]) * np.asarray(da, np.float32)


def _fit_line_loose(points):
    """允許較少點的 robust fit，專門給歪照片的箱邊。"""
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    if len(pts) < 4:
        return None
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0], np.float32), np.array([vx, vy], np.float32)


def _hough_horizontal_groups(mask):
    """找出岩心箱內多條近似平行的藍色橫桿。

    之所以先找「一組橫桿」，是為了把打開的箱蓋排除掉；蓋子通常只有一兩條長線，
    真正的岩心槽區會有 4~6 條規律的橫向藍線，即使整張照片是歪的也不影響。
    """
    H, W = mask.shape[:2]
    edge = cv2.Canny(mask, 40, 120)
    lines = cv2.HoughLinesP(
        edge,
        1,
        np.pi / 180,
        threshold=max(18, int(0.055 * W)),
        minLineLength=max(45, int(0.18 * W)),
        maxLineGap=max(8, int(0.035 * W)),
    )
    if lines is None:
        return []

    raw = []
    for l in lines[:, 0]:
        x1, y1, x2, y2 = map(float, l)
        dx, dy = x2 - x1, y2 - y1
        L = float(np.hypot(dx, dy))
        if L < 0.18 * W:
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        if min(abs(ang), abs(abs(ang) - 180)) > 22:
            continue
        slope = dy / (dx + 1e-6)
        yc = (y1 + y2) * 0.5 + slope * (W * 0.5 - (x1 + x2) * 0.5)
        raw.append((float(yc), float(slope), L, (x1, y1, x2, y2)))
    if not raw:
        return []

    raw.sort(key=lambda x: x[0])
    groups = []
    for item in raw:
        yc, slope, L, seg = item
        placed = False
        for g in groups:
            slope_diff = abs(np.degrees(np.arctan(slope)) - np.degrees(np.arctan(g["slope"])))
            if abs(yc - g["y"]) <= max(7.0, 0.018 * H) and slope_diff <= 8.0:
                g["items"].append(item)
                g["y"] = float(np.mean([z[0] for z in g["items"]]))
                g["slope"] = float(np.mean([z[1] for z in g["items"]]))
                g["length"] += L
                placed = True
                break
        if not placed:
            groups.append({"y": yc, "slope": slope, "items": [item], "length": L})

    groups.sort(key=lambda g: g["y"])
    merged = []
    for g in groups:
        if merged and abs(g["y"] - merged[-1]["y"]) <= max(9.0, 0.022 * H):
            merged[-1]["items"] += g["items"]
            merged[-1]["y"] = float(np.mean([z[0] for z in merged[-1]["items"]]))
            merged[-1]["length"] += g["length"]
        else:
            merged.append(g)

    # 選擇一段有 5 條左右橫線的區域；歪斜造成間距不等沒有關係。
    best = None
    ys = [g["y"] for g in merged]
    min_gap = max(7.0, 0.020 * H)
    max_gap = max(40.0, 0.28 * H)
    for i in range(len(merged)):
        for j in range(i + 3, len(merged)):
            sub = merged[i:j + 1]
            y = [g["y"] for g in sub]
            d = np.diff(y)
            if len(d) < 3:
                continue
            if np.any(d < min_gap) or np.any(d > max_gap):
                continue
            span = y[-1] - y[0]
            if not (0.14 * H <= span <= 0.72 * H):
                continue
            cv = float(np.std(d) / (np.mean(d) + 1e-6))
            if cv > 0.90:
                continue
            # 5 條以上優先；同樣條數時，較寬、較長、較靠下的組較可信。
            score = (
                len(sub) * 2.0
                + 3.0 * span / H
                + 0.35 * sum(g["length"] for g in sub) / (W * len(sub))
                - 1.5 * min(cv, 0.8)
                + 0.45 * y[0] / H
            )
            if best is None or score > best[0]:
                best = (score, sub)
    return best[1] if best else []


def _fit_outer_and_inner_blue(img):
    """直接由藍色箱壁找外框與內緣。

    回傳 (inner_quad, quality)。quality 只表示幾何線索是否充分，並不把「歪」當成錯誤。
    """
    H, W = img.shape[:2]
    mask = _blue_mask_strong(img)
    groups = _hough_horizontal_groups(mask)
    if not groups:
        return None, 0.0

    # 最外的兩條藍線當作箱體上下外框。
    end_lines = []
    for g in (groups[0], groups[-1]):
        pts = []
        for _, _, _, (x1, y1, x2, y2) in g["items"]:
            pts.extend(((x1, y1), (x2, y2)))
        end_lines.append(_fit_line_loose(pts))
    if any(x is None for x in end_lines):
        return None, 0.0

    def y_at(ln, x):
        p, d = ln
        return float(p[1] + d[1] / (d[0] + 1e-6) * (x - p[0]))

    top_ln, bot_ln = end_lines
    top_yc = y_at(top_ln, W * 0.5)
    bot_yc = y_at(bot_ln, W * 0.5)
    if not (0 <= top_yc < bot_yc <= H):
        return None, 0.0

    # 外框左右邊：在上下外框之間，每列取左右最外側連續藍帶，再做 robust fit。
    left_pts, right_pts = [], []
    for y in np.linspace(top_yc + 0.10 * (bot_yc - top_yc),
                         bot_yc - 0.10 * (bot_yc - top_yc), 180).astype(int):
        row = mask[y] > 0
        # 左側只在 0~40% 搜；右側只在 60~100% 搜，避免隔板進來。
        xs = np.flatnonzero(row[:max(1, int(W * 0.40))])
        if len(xs):
            # 取最左側藍帶的最後一點（較接近內緣），外側/內側都可由後續掃描修正。
            left_pts.append((float(xs[0]), float(y)))
        xs = np.flatnonzero(row[min(W - 1, int(W * 0.60)):])
        if len(xs):
            right_pts.append((float(min(W - 1, int(W * 0.60)) + xs[-1]), float(y)))

    left_ln = _fit_line_loose(left_pts)
    right_ln = _fit_line_loose(right_pts)
    if left_ln is None or right_ln is None:
        return None, 0.0

    outer = []
    for a, b in ((top_ln, left_ln), (top_ln, right_ln),
                 (bot_ln, right_ln), (bot_ln, left_ln)):
        q = _line_intersection(a[0], a[1], b[0], b[1])
        if q is None:
            return None, 0.0
        outer.append(q)
    outer = order_pts(np.asarray(outer, np.float32))

    # 由外框往內找「第一段藍色帶的最後一點」= 內緣。
    def last_blue_run(line, start, stop, step, max_gap=4):
        i = int(round(start))
        stop = int(round(stop))
        found = None
        while 0 <= i < len(line) and ((i <= stop) if step > 0 else (i >= stop)):
            if line[i]:
                found = i
                miss = 0
                j = i + step
                while 0 <= j < len(line) and ((j <= stop) if step > 0 else (j >= stop)):
                    if line[j]:
                        found = j
                        miss = 0
                    else:
                        miss += 1
                        if miss >= max_gap:
                            break
                    j += step
                return found
            i += step
        return None

    tl, tr, br, bl = outer
    top_pts, bottom_pts, left_inner_pts, right_inner_pts = [], [], [], []
    top_outer = lambda x: np.interp(x, [tl[0], tr[0]], [tl[1], tr[1]])
    bot_outer = lambda x: np.interp(x, [bl[0], br[0]], [bl[1], br[1]])
    left_outer = lambda y: np.interp(y, [tl[1], bl[1]], [tl[0], bl[0]])
    right_outer = lambda y: np.interp(y, [tr[1], br[1]], [tr[0], br[0]])

    for x in np.linspace(tl[0] + 0.10 * (tr[0] - tl[0]),
                         tr[0] - 0.10 * (tr[0] - tl[0]), 180).astype(int):
        y0 = top_outer(x)
        y = last_blue_run(mask[:, x] > 0, y0 - 3, min(H - 1, y0 + 0.08 * H), +1)
        if y is not None:
            top_pts.append((float(x), float(y)))
        y1 = bot_outer(x)
        y = last_blue_run(mask[:, x] > 0, y1 + 3, max(0, y1 - 0.08 * H), -1)
        if y is not None:
            bottom_pts.append((float(x), float(y)))

    for y in np.linspace(tl[1] + 0.10 * (bl[1] - tl[1]),
                         bl[1] - 0.10 * (bl[1] - tl[1]), 180).astype(int):
        x0 = left_outer(y)
        x = last_blue_run(mask[y, :] > 0, x0 - 3, min(W - 1, x0 + 0.10 * W), +1)
        if x is not None:
            left_inner_pts.append((float(x), float(y)))
        x1 = right_outer(y)
        x = last_blue_run(mask[y, :] > 0, x1 + 3, max(0, x1 - 0.10 * W), -1)
        if x is not None:
            right_inner_pts.append((float(x), float(y)))

    inner_lines = [
        _fit_line_loose(top_pts),
        _fit_line_loose(right_inner_pts),
        _fit_line_loose(bottom_pts),
        _fit_line_loose(left_inner_pts),
    ]
    counts = np.array([len(top_pts), len(right_inner_pts), len(bottom_pts), len(left_inner_pts)], float)
    if any(x is None for x in inner_lines):
        return None, float(np.mean(counts > 20) * 0.55)

    inner = []
    for i in range(4):
        q = _line_intersection(
            inner_lines[(i - 1) % 4][0], inner_lines[(i - 1) % 4][1],
            inner_lines[i][0], inner_lines[i][1]
        )
        if q is None:
            return None, 0.0
        inner.append(q)
    inner = order_pts(np.asarray(inner, np.float32))

    ow = (np.linalg.norm(outer[1] - outer[0]) + np.linalg.norm(outer[2] - outer[3])) / 2
    oh = (np.linalg.norm(outer[3] - outer[0]) + np.linalg.norm(outer[2] - outer[1])) / 2
    iw = (np.linalg.norm(inner[1] - inner[0]) + np.linalg.norm(inner[2] - inner[3])) / 2
    ih = (np.linalg.norm(inner[3] - inner[0]) + np.linalg.norm(inner[2] - inner[1])) / 2
    if min(ow, oh, iw, ih) <= 0:
        return None, 0.0
    ratio_ok = 0.62 < iw / ow < 0.995 and 0.58 < ih / oh < 0.995
    area = abs(cv2.contourArea(inner))
    area_ok = area > 0.03 * W * H
    in_bounds = np.all(inner[:, 0] > -0.08 * W) and np.all(inner[:, 0] < 1.08 * W) and \
                np.all(inner[:, 1] > -0.08 * H) and np.all(inner[:, 1] < 1.08 * H)
    support = float(np.mean(np.minimum(counts / 180.0, 1.0)))
    quality = 0.45 * support + 0.30 * float(ratio_ok) + 0.15 * float(area_ok) + 0.10 * float(in_bounds)
    if not (ratio_ok and area_ok and in_bounds):
        return None, quality
    return inner, quality


def _fallback_inner_from_holes(img, full_aspect=3.1):
    """舊方法作最後備援，但不再把「貼近照片邊緣」直接判成異常。"""
    p, _ = _detect_holes(img, full_aspect)
    if p is None:
        return None
    q = order_pts(p)
    # 備援時用很小的外擴，寧可多留藍邊，也不要把岩心切掉。
    c = q.mean(axis=0)
    return c + (q - c) * 1.008


def detect_inner(img, full_aspect=3.1):
    """自動抓岩心箱內四角。

    重點：歪斜是正常情況，不會因為斜就標成異常；只有幾何線索不足才要求檢查。
    """
    h, w = img.shape[:2]
    # 先縮小再做 Hough，速度更穩定；最後角點再放回原圖座標。
    sc = min(1.0, 1400.0 / max(h, w))
    work = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA) if sc < 0.999 else img
    p, quality = _fit_outer_and_inner_blue(work)
    if p is not None:
        if sc < 0.999:
            p = p / sc
        return order_pts(p), bool(quality < 0.28)

    # 備援：原有槽洞法通常能處理標準角度照片。
    p = _fallback_inner_from_holes(work, full_aspect)
    if p is not None:
        if sc < 0.999:
            p = p / sc
        # 有效四邊形就先視為自動成功；不要再用「靠近邊界」把正常照片判異常。
        p = order_pts(p)
        area = abs(cv2.contourArea(p.astype(np.float32)))
        geom_ok = area > 0.02 * w * h and np.all(np.isfinite(p))
        return p, not geom_ok

    return None, True

def _expand_source_quad(pts, pad=0.006):
    """把來源四角向箱外微擴，避免內角剛好壓在岩心上造成切心。

    pad=0.006 約為箱框尺寸的 0.6%；只補回極薄的安全區，不會明顯增加外部土面。
    """
    q = order_pts(pts).astype(np.float32)
    if pad <= 0:
        return q
    c = q.mean(axis=0)
    return c + (q - c) * (1.0 + float(pad))


def warp_points(img, pts, out_w=2400, aspect=3.12, margin=MARGIN, source_pad=0.006):
    """把四角拉成成果圖；來源角點先向箱外微擴，避免切到岩心。"""
    mx, my = margin
    out_h = int(out_w / aspect)
    x0, x1 = out_w * mx, out_w * (1 - mx)
    y0, y1 = out_h * my, out_h * (1 - my)
    dst = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    src = _expand_source_quad(pts, source_pad)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)



def detect_occupied_rows(im, n=4):
    """自動判斷最後一箱實際有幾槽岩心。

    以每槽中央區域的「非藍色 / 灰岩比例 + 紋理 + 邊緣」判斷；
    只取從第一槽開始的連續前綴。若所有槽都像有岩心，就回傳滿箱。
    """
    if n <= 1:
        return 1

    arr = np.asarray(im)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    H, W = arr.shape[:2]

    scores = []
    for k in range(int(n)):
        # 避開上下隔板與左右箱壁。
        y0 = int(H * (k / n + 0.12 / n))
        y1 = int(H * ((k + 1) / n - 0.12 / n))
        x0, x1 = int(W * 0.07), int(W * 0.93)
        if y1 <= y0 or x1 <= x0:
            return n

        s = hsv[y0:y1, x0:x1, 1].astype(np.float32)
        v = hsv[y0:y1, x0:x1, 2].astype(np.float32)
        g = gray[y0:y1, x0:x1]

        # 岩心大多是低飽和灰/褐色；空槽主要是高飽和藍色。
        grayish = ((s < 105) & (v > 35)).mean()
        dark_core = (v < 150).mean()
        edge = cv2.Canny(g, 35, 110).mean() / 255.0
        texture = min(float(np.std(g)) / 55.0, 1.0)

        score = 0.58 * float(grayish) + 0.12 * float(dark_core) \
              + 0.18 * min(texture, 1.0) + 0.12 * min(edge / 0.16, 1.0)
        scores.append(score)

    scores = np.asarray(scores, dtype=float)
    if not np.all(np.isfinite(scores)):
        return n

    # 前兩槽通常都是有岩心的，用它們當「有岩心」基準。
    head = float(np.median(scores[:min(2, n)]))
    if head < 0.16:
        return n

    # 後面的槽若明顯低於前兩槽，就視為空槽；避免箱壁/碎石把它判成有岩心。
    threshold = max(0.20, head * 0.48)
    k = 0
    for score in scores:
        if score >= threshold:
            k += 1
        else:
            break

    # 若四槽都和前兩槽同一量級，視為滿箱。
    if k == n or np.all(scores >= head * 0.82):
        return n
    return max(1, min(int(n), k))

def crop_partial(im, k, n=4, margin=MARGIN):
    """最後一箱只有 k 列有岩心：保留前 k 槽（含其下方隔板），刪掉後面的空槽"""
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
    """用告示牌照片當底，把孔號/深度/日期/工程名稱蓋上去（同範例 Word）"""
    im = Image.open(board if board else io.BytesIO(base64.b64decode(BOARD_B64))).convert("RGB")
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
    bpp = per_page // rows_per_box
    for p in range(0, len(boxes), bpp):
        chunk = boxes[p:p + bpp]
        r0 = p * rows_per_box
        n = min(len(chunk) * rows_per_box, total_rows - r0)
        d0 = start_depth + r0 * row_m
        yield chunk, r0, n, f"{d0}~{d0 + n * row_m}m"


# ------------------------------------------------------------------ PDF
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
            h_cm = BOX_H * (im.height * FULL_ASPECT / im.width)   # 滿箱 = BOX_H；只畫一半的箱子 = 一半高
            c.drawImage(ImageReader(_jpeg(im)), x0, y - h_cm * cm, BOX_W * cm, h_cm * cm)
            c.setFont("Helvetica", 16)
            mx_, my_ = MARGIN
            for k in range(rows_per_box):
                num = r0 + i * rows_per_box + k + 1
                if total_rows and num > total_rows:
                    break
                frac = my_ + (1 - 2 * my_) * (k + 0.5) / rows_per_box   # 該列中心在滿箱圖中的位置
                c.drawString(x0 + (HEADER_W + 0.3) * cm, y - BOX_H * frac * cm - 5.5, str(num))
            y -= h_cm * cm
        if r0 + n >= (total_rows or len(boxes) * rows_per_box) and end_mark:
            ti = _text_img("鑽探結束", 60)
            if ti is not None:  # 轉成圖片，任何閱讀器都不缺字
                tw = 2.6 * cm
                c.drawImage(ImageReader(ti), W / 2 - tw / 2, y - 1.3 * cm, tw, tw * ti.height / ti.width)
            else:
                c.setFont("MSung-Light", 10.5)
                c.drawCentredString(W / 2, y - 0.8 * cm, "鑽探結束")
        c.showPage()
    c.save()
    return buf.getvalue()


# ------------------------------------------------------------------ Word
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


# 範例 Word 的告示牌照片（內嵌，不需要 assets 資料夾）
BOARD_B64 = (
    "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"
    "BOFajScc1SuTztLRLzDRQgSjg2CTs6JTvAFuMWOX0n2riT5RGLfUyvZGY2EaFQmaggbFPkm3nD9+86EIT5gSfII2Qoq3EHqhwCo2"
    "tCEVfwW7W3RMYnqBqDaDtIkWUluWR1p3rPWfJEYptJwrT6E7SapKzMy93Q66Fy0g8Vt3VoEqCARYW6ItB1CtggFSTz23RAsv8T1e"
    "o1KvSdaqKHUU9/km1LCWydTzWvpGm/pzPknrY0KRjOkSjYw/SHMUzz7ZV9D5JIeHRdyxFo36ZTlTNSD8xgedcUsbKpyqTjbriQeZ"
    "JJsOgWiLutsrzlnX2KvLU/6X5QTJDa0glABA2ja5iYNTUjIzKZidzH5UJIJaLsuhB6CAIxPSTiirVXZjqOWr0s+ZrCtamaG6o3Uy"
    "klTKv1eEdWh0vF8otPfTEEhONjeBJWUR+dcfJAczIwgysJViGRJP4qyoecCO7T6pIVWX7okJxiaZP37SwoQbT33CujM8lamlJQoJ"
    "WQQFEXsYhFOwc7hOj1efdm2JurPlcwZpxkKCSBcAAxOuERbMKpKlMPPykskuz0+O5pdpOqlKVpfyDWOtPVpPYzL2IfRMZ1moUFuo"
    "1iZrzDLpKUuUqSb5MAG2/VQ8whuAnn8RYtqbU9OTFTp8iNuWFRb2lgqOirKAsd/COtIYdquF6PKyk9idqn0phsbfc7Gy7tHUguG9"
    "hc745WW8zKLx3iNUi9ysmpKSh5SyvbsRrtHU311jlO+Rv3/+Ta3O/jSfw/LTzDU2imCohGy13dJOPJKeZISNdY4yXqiobUnLyqAN"
    "xlMLr+IuKEdzH9LYqTMpUGJ1tmcpzvLNlDyErULagFVxfmuI48tiakVChrqkzjGsS+xtbUuuaZbd2hvACU3PWI6S3il6mI7MyYZx"
    "hS362aXVS65U217DfLUtDCkK/UKiPLFjpVoDFO5ZVaj0yZnqvV6lT5Z2aJKFPzocfUm/3w1sbcd/RFn0rEtErhUmm1OUm1gapaXc"
    "jyb45XtJ8G2tDoOoDyFIJUAoWJSog+QiK0zRwlIy2GZyeTMVF15vZKQ/OuOIFz+KTaJ/Vmqk5JrFKel2Zq3iKfbK0eUAiIC7gjG+"
    "KNuWxLiBliQJ8ZiTQPog8wt5Y7xko6vU5tX0MmHMM0CRwtT6k1KUWUqDjKV91TyNsbXPYqHxRHqzRapVMf0eXq1a5YTDRUh2no5D"
    "YQLmwNz54sKh5fYfw+EKYkw++gWD8yeUWOq+g8gjmYzw/V11en4gojTUzMyN0mXWrZ5RJ4A88c6cXkcfWxqTWZMjGNcDUunVWgty"
    "DDzrsxNhLrj7ynVrAsdSonpixa+87TqM43TZUrmVp5NhppNhtHQHoA3xGXMwm5dTaqzhCty77RulQlw6EnnSRDjnBQgPtCtbXN3E"
    "e2KnaSlbZCzasdTAeDmsI0xQeUlyemDtzDo5/wAUdAiQSNUlKkl1Uq8HAy4WlkcFDeIrWr5r1Gel1y+H8M1Vx5YIS68woBPTYXuY"
    "2Mo6ZiempnBWJBbEvMK5VKnjZe3x8XfrGZOMVfdtlSb3LKJ6RaI9V8Rpl6xKUeR5NyemFXXpfkmxvUfmjaxS5UmKJNuUgJM4lsls"
    "KF7nq54heUlFmltzlfqpdXPTSygKdHjADf8AHHSX0wvzoZWr/glb+GGJnErFdemHVuS7RQ0ybbKCfvh0xEa5V8RsY9bpEpVZzuJ9"
    "ovck0GytNhqElYsPLFjuuIZbUtakpCRcknQCK2l6VUMWYwm67T5tdPlpZrkJaZ5ILDp4kBWhG/WNf7G3vaxF5tDdk3ZCuTyqVOu4"
    "uRMkHxZpxTSD+s3ZMc7LFuZVXK5KOz0w9IyrhbRLvOFxOpOpvfmjuGlytPNsS4rmJx1QuGnnhLsn9VNr+UxxMo3WnqviFUslPIKe"
    "BQU+1tc2tHKon09eUajqyzG0IaFm0pQBuCQB8kYptMy4ypMq4024dynElQT5BvjYA/JgKBCbhN+i8dFuZIq7QJSkumovmXqdUc8V"
    "LtQfDYA/FQLEJHQBECxpSql4SURzvfR6c467stmVJWFG41X4qb+TfEsxHiRU6pyjTmCqtOFZsnYKShXMQobo03cKYkrgpdpWSojF"
    "NVtsh59Uy6fzrWHxxZtJNPdlV7pnZk5tU7UHqDV6wxOTAQCtmUlVMgDmKto+bSOBiQLmq01Qm5IKkpJCX2USsot5Q4WKQtIt0xLa"
    "Xhh+nu8tMViYeUVbSm2UJYbWekAXV5TFeY7S/I4oemG2qdVX5hCG2ZFSFrcSBxsgi3WYzSi1F+yD8yO0/PVJDiFCWq7KmxsIWinS"
    "rGwOYco4bRxKV3ViLGxpdXcdmmG2i4FP7CXkjm22SB8cbbeDqi5TUz79GoMu6RdUsmlB11PvnLGNbASaYxjB2Ycqkky5yZaTK9yd"
    "yK2ubY3fGbxym2qbcTStmLIo+FaPQ1qdp8khp1YspxSlLWf1lEmG1/FtGw0gKqc4GSr2qNkqUrqAEdwW2dI1Z2nyk+jYm5ViYSNd"
    "l1AUPjjtTyrcw2yu65juoYgo06aHQn1SQbUFzk2eTQE8SkbyY6GXdKdRhmTnFzUxMubBLbC3NlpHQAPlN41sysUIo9FmKaKVNtcs"
    "jk2nUpTyNusHTqtG9giZn5bBdPDUg4++UGySoIAF9CVHcIlVP6Yy52KrWbRln6piiULr6KFRmG0i6n3p87uc2TeKjqmKKrXsXys6"
    "1JyvLsHZbXLtrcbcsb3sbFQ6ott3CtRxA+HcRziFSyTdNPlSQ2fz1HVXyRH6x3PL5lUSUYShtLLWyEITYJ32Fo1PWEsvBI+ZHQpq"
    "sX19gPyeLqQhP3yWZA7SDzEKNxEvo8vU5eX2apOsTjv9Y0zydx0i5iPYlwQ5OPd86HNKptUTryjZsl3oUOMcql17MaSd7mqOG0VB"
    "CTbl2nEt7Q573t8Uc8jjFS3X+ehq99CwzGKYCy0oI1VbQHnjHITE1My6XJmTXKOHe2paVkeVOkF6flZdey9NS7SrblupSfjMdIyv"
    "qjDXoV7QJbEFGcnGpii0cuPTC3Uvzc2lFwTwFiSIwqpVbkq+/W01bCcnMOoCAhbqlpQOjdGXNcUuoUtmaTOSTr0s4DsB5BJTx0ve"
    "FVnMETmFFtMv0KXmFsBSUoLaVhdt2mt4s7OSV9JFW1+DsUfFwlg4nENfw45b2q5N4gnrEaFRey1eWucmmpSZUdVLDTrg+IWjhz2I"
    "KA9gGXpTdTkhNLShtaAfGRrqTpwjereNcP03Bq6XS6m3NPhjkUpQFG9953dcZp+kefgsuUSPDHgRUDylClqWXG9fobIS4npsReJV"
    "5YpzL7EeEcI03lJmYdcqL32VSJZaigfig2izKHiml4jbUqmzYe2dVIKSlSesGOcJq7v/AEK0dYiIxig4epTzNTnJFh6ohQEsAPoj"
    "i+FufriQTiJhyWdTLr5N0pIQoi9jbSIHg3BlWRWpitYmWt+aQopY5RQUAPxgOHRHdyywct/Ywldm3j7vo9glSe5i5MPFAdbZSTsA"
    "m+7eYj8hPYcqFpCXwxWapOMoAcaceKdkga6Kc+aJpWpysVEqkKIzyRPiuTrySlDY/JG9R+KDhvBFNw6wvYRy827flppweOsnf1Dq"
    "jM39MUt0VaXfoQnANFnWMd1GaNGepkqlopDSxoi5FhfcY7+H6bMtZgV6bXLutsrQgIcUmwV1GM7lHxPSK4w7SZ8zVLcWOVlplza5"
    "IX1KSdY7NbxRIYfQFVJbzKCNHQypSPOLxHBuUlzb4DlsyHyJ5XOCctb6HK26dwixToYqDBVbarmaM9Oyt1MPNObCiCPFFrGLevZa"
    "QSN3NCUk60relha0UKFe3XDj5NYam1o0BOqHIr5wDGs57RPVGw8foSxbhGs4fESOiONbZGomo9vis83E7VNkxcD6Y48dIsx656Ir"
    "PNzaTSpNYudiYv8AEYzR8xpjMtyBmGxbcVPfwqi9E7xFE5aHazBlzax+i/wqi908I1X8sP4OcRKgEjn1g7zYQtnTdHlNjU3Jhx88"
    "IbumETFBQOZCr4wqVxf6IP4RFkZRkeCpsLfTK/kEVnmSfrzqmn89r70RZeUNvBH/AFLnzR71/wDinL/cThJsIINzaAN0IdUeA6Bv"
    "Btxgb+YQleW8QpycU4jlsM0d2ffspY8VpsnVxfAdXE9Eec6hU3qlPTE3MK5R59RWtR5zEix/WarV6483PtGX5Alttg/zYv8AGTz8"
    "Y0K1hOZw/TKbNzZUl+eStfJEfY0i1r9Jvuj6OEpxgupL12ONRt6HBS4426lxskKSQQobwYt3C+OVYmoM7QZ9YNRclltsrJty/i6J"
    "P5Xyxw8q6NKV01mQnWwtp2WQOlCtvRQ5iI7tZoVPyzprFSk5VmfnVvbHKzQJCBsk+KkWsdN8dcTUh9uaJCL3RDTllixSQRSXB1rQ"
    "PnjUqOAsSUiSenZ2nqal2wNtZWk7OttwMSc501vZB7gp3kSvtjl17NWq16lzFOfkpNpp6wUtsK2hY30ubRmLrrSyH0kUpdJmaxUm"
    "JGWCC+8rYQFGwv1xNmcl8RuC7j0gyOl0n5BELo1Xeo9Wl6iylK3ZdYcSlz2pI57RO0521si3e2n26l9sd6vUVumSNvU6NOyUn2HW"
    "n3K3LtLQoLBaaUogg3G8iLXaStLaQs7agACQLXMVDL5zVp1xtBkaeApQBsF31PXFwDUR8/EdWy6h1ja+hA6pm/Rac+thMpPPONqK"
    "SClKLEGx3mI/PZ4TKr9xUdlvmLzpV8QtEhr+VMnXq09UFTy5ZDx21NttAnatqbk8YBykw3JyzrjiJuaWhtShyjthcA8EgRKboqKc"
    "9w819Cv6hmxieeSoCcblUnhLNhJ85uYiM3UZmpTC35t9195W9bqipR8pgrbAUoBNxvtFkU3JNc5JszKqyhBebS5siXJ2bi+/aj3x"
    "dKmlK1jk80nYg1Gl6A4dqsVCbYF/aS8vtk/rE2iw6HifLnDoSuVlJtTw/nnpfbX8Z08kD2Cjf3bTb/Ln1oJyNUrfXEn/AE59aOVS"
    "tSnvJlUWvQkHsxYXCR4876D/AHhJziwqSbuTvwc9sR05FK4V1Pllj60R7GWWbmEKczPGoomkuO8lshooINib7zzRxUMO3bMbbkWv"
    "QcwaDiOeTJST73LqBUlDrRTtAb7RJVbtNIoHKYpGNpO412Hbe9MXzNcsqWcEupCXyk7Cli6Qq2l7cI54qjGk1lLCV9yB5rYxTSqY"
    "qkSjo7smk2c2d7bfHyq+SKNUraUok6xJKjRa5UsTLkJtpxypPPWVtcSfvuq2t91oOOsOS2Gay3TpbxuTlmytZ+/WQdox68Ko00r7"
    "sxPU4LLM9KttVFtDjbfKbKHk7gsa2B590S3FGPFYnwrJyU6hXd8s/dawnxXE7JAV0K5xEvyto8jW8EzcjUGEvMLnCopvYghI1BGo"
    "Mc/NGZw4jDslIUR+Q+gzJKmpYgkDZIubdPEwq1oueRrVMkYtakIwfVqPSagp+t07vhKlopDQSDZVxY6+WJBX8V4MqFKmJel4ZEpO"
    "LSA2/wAmgbGtydDHBwXSaLWKipit1DuGVDSlBzbCLrBFhcg9MSLEGEsE06jzM1S8RGanG0gts8uhW3qBuAvujpWcMyzX/oSN/QhV"
    "HnJWTrErMT8t3TKNuhTrNgeUTxGsWG5jbLsoI8EVXIP80jm/OiAUKTkp2tSkvUJgS0o44EuvFQGwnnudIsReA8vC0SMV62Nj3S3z"
    "dUWu4aZr/wBBG5WCXPHUWxYG9o9U0rWQlf0Df8Ijyr4iHFBGo4R6ioE/K1ClyrspMNTDYbSjbbNxtBIuI4Yz7cbGqe50lQyalJee"
    "ZUzMsNvtLHjIcSFJPkMOMadXrdPocr3VUZpuXa3Aq3k8wG8mPmHY1JLBWHJJ/umXo0k28DcKDd7HnAOgjtnxRe9oiMhmlhadfDCZ"
    "9bRUqwU8ypCT5eHliUuLDjQUhQIOoI4iOrjPTMRNeg9ClKULLSLneoaCFMBDuyHJdL2yraSVbJAPPrxjA2lSiQFa8Lxtp2EqCCRt"
    "kXt0RqxQtqJTcoKeaxvHIbwbQma8a8inNpqB/nAdATvVs7trpjpPz8pL3D03Ls23hbqU285iPVbEMs4pSGcYUaQRuukIdcHnXb4o"
    "l7alO8+1TpRxc+81KMOW2VzCwlJtzFR7YjDeXtBmsRyuIpV1ZSyrlW2GnNpkr/GGpt1DS8VpOOU+sZhIkapX52vUVKOULg21lStn"
    "2oS2NLK5hFh0es4Zwu0tmi0Ku7K7FQakHztHrXCPcS3oTOZlJedYVLzUu2+yvRSHEhSVdYMOlJViRl25aWYQyy2NlDbYslI5gI5F"
    "ExUitvFjvPWZJVidqblChB/WvaO8nduiu4EFfkmET+SYdA1/FPmiFOFW8HUiv1SRqc9LKVMSRBbKTYLANwFc4B1Edh5hqaAExLtP"
    "AG45RAUAfLEbx7iqpYQkmalKSDE5KJVszKVrKVt33KFuHDdppG7RsY0iryEpNd2yss7MtpWJd19AcQTwIvvirVEOwhsNJ2W2kNp5"
    "kAJHmEONzpsfHGObnZWQllzM2+0wwjVTjiglI8piuKtm5NTs+mTwdSHKuEqs4+W1lKuhIHynzRG/QpIqnlhhWqzS5qZozQdWbrLL"
    "imwo85CSBGWnZc4VpjqHZehy4cSbpWsqWQfKY7FCnJ6oUxmYqVPNPmVg7cuVhZT5RHR0EW1iDQLcI1KpSJCssJYqEmzNNIWHEpdF"
    "wFDcR0xu6QCQNREKDkxe+wAT0WhwBA0uPLFPpXWZXOlyjyVXm2pN10TTjanCtPJlG2pNjpv0HNeLfCue0Vaq5B11HdtHqJhi0hxC"
    "kLAWlQspKtQRzERCM42VjBj1QZefYfknm1oW04pGilBJBsdd8bOWs0+jBNPm6tU1TDsykvcpMujxEk6JueAA488F6lsdLDuCqHhZ"
    "+ZfpUnyLkyfHUVFRAvfZF9yb8I7luiMErUpGe2hKTktMFO8NOpWR12MbELAQtzWhCwVcXB5xvgxXuNseVNquSmH8INonamle1NDY"
    "C0Np4JJ3DnJvp1xAWAoBV7i4PPEHxHlHQcQTaJhLk1IeNtONy6voa+kJOiT0iJlIKmzJMmeSyma2AXQySUBXHZJ1tFdYeqdVlMzc"
    "R096pT8zS5BpT4l1EukbRTYJG/S50EGtbMFgUWjSNAprNOpzHIyzIslI1ueJJ4knjG7p+VHDpONqBXFvNU+oJffZQVuMBtQdSBv8"
    "Qi58kaT+YUkFqak6PX59xO9LNPWkedVooJVpwBhX6DHIoVfmaxt90UOp0spFx3WlNldRB39EdYq6IhAg2voRffC8hjjYuq07Q8PT"
    "tTp7LL70ojli29tAKQPbWtxtr5IjuE8X4jxrSO+MjJ0iRa5RTW2+466SpNrkJAGmvEwFiSYewxS8LomkUthbSZp4vOBSyrxuYcwH"
    "AR1SocxisaNjrFCcxnMI1FMhOhKztvNNFrk0BAXtDU30I0PnjYxpjLE1AxtSaNIGmqlKtsoZMwyolpe1squQQTwPlhcpI8Z4GpeN"
    "2JVqorfR3M4VpUyQCUn2yT0G0SBlttlptpsbLbaQhIHAAWA80R40nE8wi0xidthR0tJU9CfMVlRiH5c4gxFVsb16nO1xdTo9MUpH"
    "LPNJ2nFbWymxSABqFc40gEWmdnpgac5gXPRC80CBuOcxwMS4HouK5qQmqowtxyQXttlKrbQuDsq503ANo6VTXUESpNMRJuTIOiZp"
    "SkoI60gmOAudxMW3VzlXw1TQyCp3k21vFsDUk7a0geaARLb3uTck6xjW4llCnFK2EpBUVHgBreKgpNczIxBXHX6BPd0URKglE3UZ"
    "RDDTw4lKQNojmtraO9jzEtXnpF7C2GpU1CtzDOxOOMaNSaCLKuomwUdQATcA6xStGzgvDWDxiSo4nw9UW516aCroacSpDG2brsN4"
    "2jz7uEToab4qCkuP5SYYmJ5ODnwUJT3XOzU8ylTpuAEhKSo2udEiO7hDGmLcbSqKlT6fQWJDlC2tLz7ynARvGibX6tIhbFiX6DDO"
    "UJv4htuvpDhe2sAJ038YGQbZ/EPngcob25MnywSkQdkQA3bJ/mz54Y82iYbLbzIdQd6FgKB8hjKE9cKwgCH1XKvB9ZWVvUBppxW9"
    "csosn93T4o0JLJLBUo5tqpUxMdD8ypQ8wIif26TAKd8HqDlUvDFEooHe6iyMqR982yna89rx0wo/iHzw61tIRTzwBrOysq88l92T"
    "acdR7RxaEqUnqJGnkjMXFf1aj5Yda454OybaiLcGlUpZNSp8zIvtr5KZaWyvZVY7KhY2PljlYMwtL4KoqaVKcu8jlVOqcdKbqUbD"
    "cNBoBEh2CRwhbCt0AMK1fiK8hjXmJOXmXmnnpBp11k7Ta3EJKmzzpJ1Hkjc2CYIQRAGLaX/VnziAVr3hpR8ojKdOaAee0QDApZ/m"
    "yP1hGJ+WafFnpVp3n5RKVfKI2L9UI3MXUHNVQqYvU0annrYb7Ib3gpiNU0WnA9DDY+aOmSRBv0QBpIp8u1o3TpVH5raB80bCQtGi"
    "GgnqVb5Iy7Rva0AkmAGXX/Vj30C7n9WPfxl1gG9t8AY/ogHtB76ES4fvB76HwDugApKwPaD30c2rYepVcSUVKlyk0Lb3Egq8+8ee"
    "OolNxxgEEa2ip2IV3OZH4Um3S423OyovfZamAU+TaBtG3T8m8HyFlKpSptQ4zL5WPMLCJzt9ELb6IjSeppNnOp9Ep1JRydOpcnJp"
    "P9S0lHxgXjfuTwh1zzCEVbjaBBvjDckQRtcAAeuDtQuMAac/R6fVUFE/T5WaSf65tKvjIiIVXJjCdRJU1JuyKzxl3bD3qriJ3cjh"
    "A2uiLcFUt/yfqSHrqrNQ5P8AFShAPn/2iR0zKLC1MsVSb82ocZiYUR70WETLaJVBUoxMqfoW7MMnJsSLCWJWXbYaTuQ2LAQ99hEw"
    "ytl1AUhYsoXtcQQqCVXGkXbYhF3MtsNreLi5KZVfXYM47s+bajMzl9hVj2mHpAnnWnbP7xMSAqPPpDbn8aHuDlownh5oeJQKYB/l"
    "kdkZ00CkgDZpFPT1SyB80bwJ54Oo4wBoih0wfgqQ+Do7I2JeTYlE7EvLNMIOpS0gJHxCM1zzw0qJ4/FAB4breWNU0+UM4Jwy6FTA"
    "TsBxWpSOYc3kjYJPPDPLGiCcbQ6goWhKkqFiDqCI5DODaBLy8xLt0mVDMwrbdRs6KPzeSOwNeMKw54t/QhwGcA4Vl9UYfp1+dTQU"
    "f3rxvN4aojXtKPTk9UsjT4o6OhgHSJoW5gTSqegWTISg6mU9kPakpVhe21KS7a920htKT5wIy3vxgEm28wA6xF90Aqhu1AI6YtgA"
    "k80Iq03awrW6IadYpA7RGg+WFtK/+mGjzQ4RSB2ljS/xwto8bQCk8/ngWMQois80C54ACEU3MLYMUhimJVqab5N9pt1B3pWLg+SM"
    "iEJbSEpQlIGgA0AhwTwvC2bwBo1aiyNclFSk/LNvsqG5Q3dXMY18PYZpmF5RUrTGOSQpW0oqVtKUekx1bW4wCbcTB6qzC0Dc3g35"
    "9IxFR6YRJ5z54AyFYF9R54aXBwtDAATBteANedlhOtFpbr7aTv5JewT0X3xgp1Fp1JB7hkmWVK9ssC61dajqY39kQdLbtIAw7F9N"
    "kRyarg2iVt9uZnZJKn2yClxCihXnG+O2T1wLkRbu1gOTsoQEpAsBYQFK03DzwCqGqV0xEraIM15uWYnUFuZl2n0HXZcAUPMYemza"
    "QlCAkDQAcBBJvxgeW0aIg7Sre1HnjQdw9TX6m1VHJNtU40LIdvqBHRSLQ++m6F9LFsN8aw8UW64abj70X64yXhp1G+CYGXV+IPPH"
    "OqeHaVWbGoUuUmiNxcQCfPvjqaA74JWkcIt2SxwmMGYdlrcjQKYkjceQST8YjebpEg1o3TJNH5rKB80bvKCEXE9USxTWEqynRMq0"
    "OpKeyDyCR/MIt1CM/KgQC7pFIYgym32BHmEOSjZ9oylPVYQ7lPLALnliNFuHxuKPjhi1FP3nxwdu++GLUCdLxLMALij94fPC21f1"
    "Z88K4EG45oagaFK/qz1XjHMsNTjK2H5dLrSxZSFi4I6oyki8K/NFV0Q4lCwlR8OPvP02R5Fx4+MoqvYcw5hHYKipQukiHXgGFtbg"
    "XNAsYVoVuiBROj6CvUbo1Xfap6o2XR9AXY2NueNZw+InqjjW2RqJqPm94rTNoqNJlvGLf0fz6RZT0VtmqQJCSubfR99r8OaM0X9R"
    "pmLLEEZgta/evfIYvUHQWii8t/EzDYTxs8PiMXnvtxjdfyw/g5xOdWMSUqg8n3ynES3K32NoE7Vt+4GOd7I2FN3fqW96rsjm5iYO"
    "qOKzJGQXLp5ALCuWUU77WtoeaIQcm8Rby/Ia8zp7IzRhTkrzdg2/QsoZiYVNx36lvMrsheyHhXat37lPLtdkVocm8RDVL8if2p7I"
    "Ycm8SqULuyHpv9o69Gj3EvLg4eO6jKVPE9QmpN5LzDjoKHE7lCw1ET7LTF9Co+GUytQqbEs/y61bC73sbWOgiOKyZxJb28hf9N/t"
    "BGTOJEiwXIk/pv8AaO96PT6eYx9V7lnjMTCpBArkoffdkIZiYVB925XzK7IrFOTeJAD40h6f/aCnJvEgV7eR9MeyOHSo9xvNLgs0"
    "5i4V0tWpc9QUfmiRIcS82hxBulaQpJ5wRcGKTTk5iMK+ySBH6Y9kXPJNLYkpdly2220hBtzhIEc6sKcVeDuVN+qOfUcK0uqVWVqk"
    "ywFvy40/FXbdtDjbhEAzuJCqTb8V2/nTFrcIr3NTDNVxCqnmmyS5nkUubeyR4pJFt56I505WlG+yZWtDhZID6pVQ2OsugfvxaFbp"
    "1LqUqG6u1Luy6FbQ5ZWykK3b7iILlXhasYenZ5dTkXJdDrSUoKlA3IVfgYkmYVGna9hp2Sp7IdfU62oIKgnQHXUx2xclKomiQ2MJ"
    "wrgPZAMpRwP049aI/jbD2D5LDU9MU6XpyJpCU7BZeBV7YXsNo8IhasqcWKHuYgftm+2AMpsWA+5qNOZ9HbHRU4/8hm/scrCkvJzW"
    "JqezOIaVKrmEpcDirJKeNzwi6vBTAitTKUk2/wDIHrRVYyoxbe/e5Pp0dsBWVGLh+DEn9s32x3quE7WnYkbr0LZZwngZJSpuTpW0"
    "DpZ8b/fRKkgAWEULK5X4saebUqmJCUqBJDyNNeuL5b0Gu+0eKtFJK0rm4/wHjGGesZR+/wDVr/hMZuI1jFNIU5LuoSLqUhQA6SCI"
    "8z2No8tvk7ZF4uSkZvYcladLS7ongtllDarMAi4AB49EQR7K7Fi1m1KUb8eVQB8sSOUyNdXLtOO1kNOqSCtsy99gkai+1raPqOVK"
    "VNKbONmnoST2ZMLbXtp/4P8A7wTnJhbiqe9B/vEeORSida6n4N/ygqyKJTZNcTfplz60cMuH5Zq8iQezJhb8ee+D/wC8RTMbMGi4"
    "nozEnTlTJdQ+HFco1sC2yRvv0xs+wS4B7ut/Bz60c2tZMVOQl0uyEx3ydK9ktNt7BSOe5VFUcPdO4bkc3KcFWNpG2viuXtw8Qx6A"
    "O6Kfy6wRiCiYolpyfprrEuhCwpalJNrpIG4xcHC3GGNmpNZWKa5MJkpYzImyw13QE7Ad2RtBPNfmikc3k3xe6QDpLtfIYvWx2Yp3"
    "M/D1YqeKFvyVMnJlrkG0hbTRULgajSPPQdqquaktDv5Mi2F5n8Yzaxp+aIrrGeAaphpkT047KraedKEhpZJvqdQRzRaGVNMnaVh1"
    "5ielXpV0zSlBDqCk22RrY9UYc3qdNVKiybUpLPzCkzBUUtIKyBsngI7zquNe8WRK8bFP4XwtO4qnjIyKmkOpbU4S6opTYWvuB54l"
    "SMkMR7Ws1Tt39ar1Y6WUVCqVNxE69O0+al2zLLQFOtFIvcaXPVFwp3+SO2JxMoytAxGCKLXkfiPf3TTfSq9WA5kriRDekxTSACfs"
    "yvViZZvVCq0tqmTNLmpqXUVuIWWCRfQEXtFaT2PMVTbCpZ6rzZbWNlQFkkjmuBeFGrWqq6aK4pEbDRQVC9yDbrj0XltQpvD2GWpS"
    "cLRcW4p4cmq42VAEeWKNw9hWsYifSzJyTqkqNlOqSQhA5yTHpWUa7nZaavfk0JRfnsAIuMqfQo3JTWpmUea8VbnTS6jMLkp5ltxy"
    "UaaKFbIuG1bV7nrFteiLRUeaAsggjRVxaxj5sJZZKS9DqzytLMuuqDbSVLWpVglIuT5I9HYHp07TMJU+UqG0JhCCSlRuUAkkJPUI"
    "6rEhJsLLrUnLNOcVoaSD5wI22xtG0euriuostjMYWGsgBQ1jbtfhDWmgVgc2scmsVGvSb1qZRGJ9q2ijOBpV+kEfPHBs3Y1a4cNN"
    "zPKTdLYqU8NA2zJh93q3aeUiOJM07EldbUhiTksJUux23AhCppSOPtdEadN47TVSxi6PufpcuDv5Sok/woiN43xRiSQl0UctUUz1"
    "VCpdliXdcW6EqFirxgAAOcwKRag4UoruZwp1OS9N0tuT2lTLb5JUspvynKJOnjafNEyquEMX0m8xhfE846ga9x1BzlAegKVf4/PG"
    "nhPBNZwFTJl5up0JkLG3MTL7Di7JHAnaAsOjfHFwPO4wxZiSp1Kn1nueRS4QpxbKlS7itwSloq059+nli22j6k9bkrwjivFMxVU0"
    "jEuH3pZaknZnG0EN3Av428a84MTtNiIiztFxm4CBiyTav+JTE/OqMcth/GbEwlxeMWX0A6tuSCbEeQi0L30LYk87JonpZcut19oL"
    "Ht2XC2sdShqIiM9l9IttqfdmcTVU30YFRIJ+NPyxNE7QSNogm2tuJjkYrrEpQ6K/OzzE29KJAS73NopAPG4IIF+IiPQEMVhSXR4z"
    "OXSFW/narUU9qoktOwpJOU/Wg0elzR3KlmW5gJ6bqSBeI9K0mnV2UZqEhg12camEhxpyp1OyVA7jbaUfijVwVVJLHBnKK/hlqQk5"
    "FR2nJKZWhCV3tbSxJNj5oqXp6hjcz8K0+UwtO1Kdq09PVFktqZMzMCwO2AQloWTuvuEbWA8fUeQwlIprFZkxOqCipllvx0Jv4qSl"
    "A32Hxxp5m4Iw7QMDzs3IUxtE0lTSUzC1qW4LrF/GUeIid4Xp0tI4fpjTMuwjZlWrqSgAk7A1vbWJG+tiM45zVohdDcvIVuZTexW1"
    "Iq2fj1iVyM83UJVuaZDgbcTcBxBQodaTqIzbZAsCR5Y5bOI5OYxBNUNKx3XLMofUL70q7NPPFQOoeg2iPzrGLnXFCVqFDl2rmylS"
    "rq1W8qrRu1+uooEiZ12TnJpCTqmVa5RQ6SOA6Yh0njPF+MFqTh+hN0yUvYztRubdSRa588ZKRo0WqTWbE2zP4kMrNN09Lrk/KNpZ"
    "sjZA2AFGydDqYsuVpQqNGbal8T1V9IJtOtPoC19BITYgf/TFdUWhCYzgnZGvTArbrUiHFOPtgJK9lBHiDSyQbAGLXqVHkaxT1SE7"
    "LpdllW8S5Ta261t1oqX0j1K0zQwvJU3B05NLr1WnZpCm9hM3P7aVXWAfoYsDp0aRqYZayvk8M0x2sqprs+uXSX0rW46oL43SCQOq"
    "Op4P0vLqcLlRo8vUqK6u6aguXDr8monc5obo5lDdxjmYvm28yKzJYUwuGDT5dQmJycZbAbTppYgagA+UnoMLcFJ9hWm4NfSKnhqU"
    "ppKCUctLospJtqDfUG3CJMLARzqJRZHD1NZp1PZDTDSbdKjxUo8SeMbpJ1tGmZMNQku+EuWFTMwwhXt+QXsKUOba3jyaxXeLKlL4"
    "fQ3hDAssGq3NOJU4qUttMpBuS4s31PG50GsSmryWI60VSstONUWTOi32vosyscyfvW+vUxAn8LsYRzUwpJUN+al0TTDhnFKcK1TI"
    "BUTtk772HmjPqWxPUTdbkKdJSDlUokzWeTu93WtTfKHnSlOpHC/G0QGjSuJZrMTFakVeQpk2002Zt5qWLqCmwPiBRuLWuSY6mKpN"
    "WKM0sOSkqgOt0VJm59wC4buboQTzmwsOmNXBswqfzSx8lCweWb5NvXS6TsH44vqCQUednFy/fCiNUrE76/EcqKXG5Rd/xVAJJ5jv"
    "vHMxzmFi7BcjLzs7SKQhh9wtgofceKCBfX2o3Xt1Rq5cVqUwJhlqi1am1hmoomHVOobkHHdslWigpIsRa0Z8/SmYy75QJVfuplSb"
    "p1FwrhEewOxLy2OqnKsTSMRUmTQ8hLiUN00qICgCAdpW/WJVTW55qTQioTDUzMp9s600W0q/VubeeGUVXK0iQc/GlmlfuCN8btIu"
    "xDlYnKBhurKc9qJJ+/VsGKrypYn14Il3kYwTRZVUw7ZnkWdom4BO2s8ea0TbNmstUnBU+1tnuifT3GygHVRVvsOhN/OI1cEZbUOm"
    "4ZprdUosi/UEtbbzj7IWvaUSbG/MCB5IlrspEsOKU3n080aiakoySz3Wdj6LdlP4ni9GnNG/m1UpOk48wROzzwYlmHVuOuHchIWL"
    "k21jQkpWVp38oXkZJhqXZSyWg0ygIQm8vc2A0EdDHzbFTzgwbIvIQ620krW2pIUk7RVoQehMPQvqbeJK5RsWMlyhV3EzMwoW26ZK"
    "TDjTielNgnygiOPhfGuFstHThxNIryJp5aXH3JhlIecUoeKeTBva24Dnjp4pnzlCtudpkwy7SZpwBVDedIUhR+/l99hzp3c0R3KN"
    "EhirGNRxbW5+UNWLyjLyKlgKQSLbYB4JFkpA3WJgC80K20pVskbQBsoWIvz9MNeXyTSl7C1lIJ2Uak9A6YeLAWtaCTpFMke8J590"
    "kS2E62vpe5JhP7y4prO2pycxPy8tP0Nmm1ZxsLM0zOB1aG76cqhCfG4ka30i1MQY4Wp1yj4UlBWKufFKm9ZeVP4zi92nNeILjvA7"
    "GGct6vP1N8VGu1B5gzM45qSSu+yjmTv6+qwg9i+515urZjU+mUhNKVS6y46y0QhiTcuWtkWcccWoBN9NNDrujYcxHs0ruGpU51mu"
    "hZU3IYZmitwqP3zhT4qNd+2TExwuOXwbSEKShzap7AKVnxVfQxv6IiWK8X1vAdIXM95sPSjal8nLtNTK1LfWfxUJQm9t5vBAh8xR"
    "MY4tYkqjXlsYhp0mtRVSZGdbDwHArKBsqUNxtrppaJlSc38G05pumTDM1QVS4DYlZiVKQ0Obxbxx8sMtK7QpJdZXU2ZCfqDf0Rtc"
    "mHVtIJvbVQSCdCRY23RrpqWIZnNfwUmJun12XaSlx9ycp7f0JGwFKAsLgi4A13mJ6D1sWPIY+wtU1JRKV+nurVoEhyyj5DHeTpfh"
    "c3jDLU+UkxaWlJdgf4bSU/IIzbhAgd3PChpMK94AfA54F4V+mACOgQjuhAGFY88AGBdPRAVoL3hl4AyXAg7YjFcDfAKwIAy7ULa6"
    "IxbZ5oVyeaAMu1pCv1RiA6YdcDjADtrfDTrCBHPBB6YAYBDocOuAR5ItwCBBsOMICFwIwLwdkQtBC4ACRxMK4gmBv4iFwC45oXi2"
    "4QdICgBruEAOQBswiAeeEmwFoPDqiAYRbje0IKsd8G+m+FYGKAbXTBBhWHPCAH/9gA7QgXF98Iw1V7i0APtCEMuQPa2hXPTADtyo"
    "KoxbSgdYJWbcIoEQOG+GgkQDcwLXOl4EHhULXohloPGKB1jC14whug20iFBr0Whpv0QFKEN2+iKQOvMIFieAhb4KeoxQHZhEdEG8"
    "LaiXANkmG8nrD7m0NJi3ArW3Q0gkw83hu6AFsw20OuYG+AsC0NKSIda8ADSLcg2CB0wjYbt8C8UBPRCAvAvCvbjAgSLQjoIFzeDw"
    "0gUENJMP6YF/JFFhhSTx1gbJ3aRkvpDSqAsMKIFocVCATrAgNnqhEWhbRhbUUCAvCtaBe0K8AE6w0gmHE80C27WIBh0EA6w/ZgEd"
    "MUGO3QIUOIvDbRSCvaHC5ENg3MChIMM8YmHeWBcQuQabw03vGS0KwvAowJNoWyRvMOveFxhcDdnSBsHnEOvaFe8W5BhSRA2eMZAf"
    "NC0vC4MeyYBBEZPkgHQXJgBliRB2TxMPFt0I24kmAMexaFs3gk24w25GsLAcEDrhFIhA6awibQACkDWw64Zck2hxsYSRaAGuIKZd"
    "fjKVodTvjUc9qOqNx8/QF8PFMaalBSQQoHThHGtsbiaj3G8V3mZOmnyshMJbSsh/co24RYj+46RVucD3I06RVr9mJ04aRiirysaY"
    "/L5R9kmWAH3z3yKi9QDp4p80ecaNRn6/jQ06XnDJuLccs8L3TYk8CDE+GUtYtpit7zOetHacYuEbu2hyTLRsfxT5oJBt7U+aKvOV"
    "NbA+6t3T9J60PGVlfCdMXPf+p60cunDuNXZZgSTwPmg2ItofNFZJyuxCL2xa753fWhexfiO4ti53qu760TJDuF2WaQTwPmgkHiDF"
    "YnLHEl9MWuD9Z3th3sZYlsPrtc0/Kd7YZIdwuWYAbbjAAuePmitBlpiYJt4WuD9d3thexniYHTFrvv3e2GSHcLvgswg8xg2PMfNF"
    "Yqy0xOT91zl/z3e2Hexricj7rnPfu9sMkO4XfBZmttx80JIvfQ+aKz9jXFOzbwuc0/Ld7YCMtcVg6Yuc9I72w6cO4XLNt0HzQSDz"
    "RWXsb4sBuMXL9I72wVZdYu4YuX6R3th04dwuyzQnS1j5oQHQfNFZex1jEp+7Bd/0rsIZdYz1tjBevO67Dpw7hcs3je0K1ze3xRWY"
    "y8xmDc4wXYf4rsBWX2Nr+Li9XpnYdOPcLlnW0GkJOkVicvscW+65Xp3YCcAY4F/rvVr/AI7sOnHuF3wWfexhE3MUfi6VxfhASqpn"
    "Es0/3QVbPJPr0tbffrjqUHDON6/S2Kizih5pt4EhK5hwKFjbWNrDpxzKWhMxbukAEaxWpwHjwbsWKv0vOdkNTgbH6CT4V3v/AI7n"
    "ZE6C7i5izN5MEDSKzGCcwP8Auu37dzsgHA2YSt2K/wD9hzsh0F3f3FyzrgwhbcYrPwHzDCQBiodfdDnZAGCsxkX+ugHm+mF+rE6K"
    "7v7i/sWZcAwiEFQUQNoaA80VerB+ZSVeLiS45+6VdkIYRzLJ1xGB/qT2Q6K7l8i74LR2tISSDfWKv8EsywPFxGD/AKk+rDk4RzLA"
    "O1iRPQO6FerF6C7kTN7FmkgEawSoc8VcrCWZxP3SJGv9pPqw44TzPsB4Ro+En1YdBdyGb2LPCtPbQkqTtb4rHwUzPG7EKD/qf+MB"
    "GFc0Qda+gD/Mj1YdBdyGb2LOUQeMNUwydS00TzlIitDhfNG/u+j4SPVgnDOaf3teR1d0j1YdFdyLf2LNSUhNhaw4QgobWpEViMNZ"
    "pgG9cQf9Qn1YAwzmne/fxA/1KfVidD/shf2LPUQd5Hnhi3AN1vJFaHDeahv9XG/hCfVgnDmadvdtsG39en1YdBdyF/YsgTItu3m0"
    "ZJVz6YT06RWSMN5pi/1bav0vp9WF4N5plVu/bWz+nT6sVUf+yF3wW2q6VeL8l4bfXdFQP4NzMmPbVZo9U0B8iYKMB5iBn3YZChuT"
    "y97+XZjXTXKLctaeqLcg3tLYm3r8GGFOH4oiVbm3K5dLOAnqg5bZS7UkNspA6ySq0RVGDszE3+qcv5X0+rDk4RzPBFqsxpzvI9WH"
    "SXrJDN7GaXybm6u689WZ9umsODxZCmFRbSeF9skHzRlThbMDA8shrDlUl6tIM+1lHmUpWB0Dj5DGqcK5o/e1Vj0yPVhHCmaGzrV2"
    "Cf06PVh0ktpImZ8E0wPjKfxIl+WqlFmabOywG3tIUG1628Unj0axK9q8VCnC+aQNjVmVAbvo6PVg+DGad/dRj0yPViumuUXN7Fvb"
    "VuMaVXp7VZpc3Tnz9CmmVMqIG4Eb/Jvir14YzT3JqbHpm/VgJwzmqE2NTZJ/TN+rGXSXchm9iWUTA9UpdJlaccXVJtuXRsJTLNtI"
    "SBc7iUlXnMdXCmEZDB8m/LSLr7iX3S8tT5BUVEW3gCK/8Gs1b+6LNuh5v1YRw5mtbSos9XLN+rGlT/7IX9iysRUSWxJRpqlTl+Sm"
    "EbJKd6TwUOkGInJyWYGG5VuRlO9FalGUhDS3lqadCRuB544Hg7mqBbvi0P2zfqwhh3NO2tRY9Mj1Yz0/+yGb2JA9U8zX0FEvQaLL"
    "k6ba5rbt5LxzsKZZ4ilsUjE1brTZmisuOIYuou3FikkgAJtwA4Rzzh3NEkfVFr06PVgnDuaP3tSa9Oj1YKmk75kM3sW8EkWjnVwV"
    "oSxcoqpJT6QfoU2hWy50bQN0/GIrLwfzV2dag1f9Mj1Yb4PZqX+32url0dkOnf8A3IZvYyYBpmJp7M2o4grVOXJWbcQ6CkhG0pIS"
    "lKCfbCyb3ixp2tz7ai3I0Cfm1/juKQy375Rv5hFaLw9msd1Qa6Po7fZC8H81gLCfZ9O32RrpaJZkM3sS2o4cxRixtcvWarL0unOi"
    "zknTgVLcT+Kp1XDqFojFay3r2DJg1XAU3NBCgA9JFe2pVuIB0WOg6jhGFNBzXF71Bno+jt9kI4fzYJAFRZtf+vb7InRXpJEzexuU"
    "XMrGqVJZqeCZyaI0K2GVtnzEERYlEq79VZK5ij1CmKA9rNhI2urZJ+aKwVQs2D+EmfTt9kJNCzXA90Wdr9O32Rel/wBkM3sXFtA8"
    "YgFawrWpzNWjV9spVTZVsXVe3JWSq6bbyVFV79kRsUTNkLJ7vaI/Tt9kJVDzcUNpM83p/wCQ12Q6S7kXN7FtScjKyAV3LLts7a+U"
    "WUpttq5zznriOYXwDJ4Xr1Vq7M048uoE7KFC3JJKtoi/E348wiEGjZt20n0ena7IYKJm2QdqeR0fR2+yHSXchm9i4nVOcmoNKAXY"
    "7O1fZv024RUeacxjCtSjOHnMOBSXZgONzUktTqHrAgCxHiG5ubxr94s3Cq5qAA6JlvshGhZuEeLPjXfeZb7InRXchm9i26LKOyFI"
    "kZR6xcYl22lm/EJAMbh2uEUyihZthNu707XP3S32Q0YfzdJ8afHkmW410V3Imb2LCTg4T+IG69XX0zkxL3EnLpBDEqOcA+2Wfxj5"
    "BG/iCkVGpy9qbWpqlTCR4q2kpWgn8pJHyG8VccP5vW90Lf6puAcP5u8J/wD/AGkROj/2QzHYwRlnXqbjR/E2IqgzNTCQvYUhZUXl"
    "KTs7Z0FgE7hD61l7W6pmu3X+7XZemciCl+XcCXWFJbKQgAg7ySb24mOKmgZup17vBPD6abhd4M3iq5nx8Kbh0V3IuZ8FjUvAVBpU"
    "33eiS7pnjqZycWX3ifzlXt5LRH8aZM0bFM2ajKuGmT6jdbjSNpDh5ynTxukWiOd5c3A2UJmkan2xmG9oeWGih5vgaTqb/wCZbh0V"
    "3IZmSGg5ZYkoZSlvH1QSyn+aQztp8y1ERO0U8OU8Sc++qfuLOLcSEFzrCbCKlFFzh/tqbf5hqCKRnELATqObV9rsh0Uv9yGZ8FuM"
    "ybEnLdzyjbUs0BZKWkBIT0gAWiq8zsM49xOJahtGTnaW4+HO6WkBpaCAQOVF7WFybpGsYTR84Tb6oIuP/Ib7Ib3ozhtrUEE/5hvs"
    "h0V3IKXsWepqZpNJlpOmSyZpxhpDDYW4G0AJSE3Ud9tOAJjg0vARfrScQYmnU1aqN/YEBOzLyY5m0Hefyj1xDTR84iQe+KLf5hvs"
    "gmk5xgeLUG/K+12Reku5C5M8Z4qxDhdPKSWG+/DDvitOS7itptfAOIsSR0g+aOJlLg6r06cqmKMSIKKrU1EBtdtpCCdpRPMSQABw"
    "AjjilZyBPui3f/MNdkJNJzivc1Ju3+Yb7IjpLuQzexcWm+8AG8U8qk5w8Km2P27fZAVSM4BqmpN3/wAw32Q6S7kS/sXELc8HdutF"
    "Ooo+cRvept2/zDfZANEzjv7rNgf5lvsh0l3It/YuOELb4p7vHnGr2tWR5JlvsjSr7ObGG6Q/VZ2ufS0u3yjpafQpSBpw2emHSW2Z"
    "EuXiCAIaVAc3njzthjEWZeMHphuk1t1xUulK3OUdQiwJsN46IkJouce81a/VNN9kWVDK7OSCl7FyKJVx0hpBPGKdTRM4Rqqq6c3d"
    "TfZB7y5wkkiqJAvoO6Wz80Z6ce5C5cATrvEOsBxvFOij5xAW76pv/mW+yD3pziTp30Sf9S32Q6a7kMxcRMHZinDSM4zY980+WZb7"
    "IXenOMi3fQX/AMy32Q6a7kMxclhxMAgbgIp0UrOMb6qn4S32Qu9OcZOtUT1d0t9kOnHuQzFxAJ5xAIHOIp40rOPhVEj/AFDXZCNL"
    "ziG+qJv0TDfZF6ce5DMXEFflQRcxTgpWcWt6sB/qW+yEml5xX91hbpmG+yHTj3IZi4iL/wD9hWtx+OKdVS847i1XT8Ib9WHd684h"
    "+FkX6X2vVh0o9yGb2LfuOeFYRUApucQGtVR6dr1YQpucROtXR1cu16sOnHuQzexb1r8RA3DfFRKpmcXCrt+nb9WEaXnBb3Wa8r7X"
    "qxOnHuQzexbwKR0QdOiKjRSc3i2Sa0yCD7Xlm7n92AKZnD/fDNh/jN+rF6ce5DN7FvBQHGFtCKi725vn8MM+mb9WNWpjNqk09+fm"
    "au3yMuguOFDjZISN+mzrBUk9FJDNyXOFX4Q6w6I89YaxlmLimbcladWipxtO2eU5NIte2ni674kapXOBtRBqiVJHFCmzfq8WLKhl"
    "dnJBS9i4SBAsDFQhjNg/hJwDnUGyfkg9x5vqVcVBvZ/PaHxbMZUF3IXLe2RC2AeMVC5K5vNp2k1Db50hbRP8MZJdGa6EguTZJvvu"
    "2rTqtBwS9S3Lb2QIBAvvirHXc0kpsh4rVxJbaHzQmfZSXq5MpR+o2fmjNlyC0NkbVyYJtaK35DMcnWpJHUyg/NGXuLMMj3ZSOtlH"
    "qxNORcsC3TrBt0CIImSx0hHjVZpav0Y9WNZSMwkXCZkK6QhHZD+pLliEW6IGu+4itkrzJVcBdulQb7I1ptrNJQCZaaSFX1KuTCbe"
    "aKkm7XFy0Nq5Oohq1nniqO4c4FD3Rlk/rterDVU/N86d8pfrC2/Vjr013IZi1iemCBc74qcUzN8k/VRjq5Rr1YIpmb/CpMDrcb9W"
    "L013Ily2LHoha8Yqk0zOAj3Wlx1ON+rA715v292GD1Ot+rE6a7kW5bN+FxB0EVIaXm+bfVhkftW/VhvenN9Rt34bA5y6i38MR013"
    "IXLbJ6oG0BzRUxombv8AfbXpkerDHKPm+2Rarcpf8R5Bt+7F6ce5EzFulXOYbfpEVKKJm8pIUa0EX+9Lybj92CaDm1/f7fXy6fUh"
    "049yLctjaF+EC/TFTd4c2D/1A36cepC8Hs2N3hC2f9QPUi9NdyJmLaChAuOqKmGHM1z/ANRN/CP+EA4azX/7hSeqZ/4w6a7kMxbB"
    "PVDb23WiqThjNX/uJHwn/jC8F81LX8IUX/zO/wDdi5F3AtcXI3w62sVQjDOat/uia8sx/wAYccM5qqT90bI/bn1YmRdyFy1dOaCL"
    "dEVP4KZpnfiRv4QfVheCeaX/AHK38IPqwyLuFy1zAJvFVpwtmgNDiRjyvE//ABjKjD+aDf4ekV/nKJ/+MMq5FyzrwFW5orfvNmcU"
    "278Uu/l7I1lYdzPUr3clP1XCB8kFFci5Z8AjnisRhrMwJNq7KX6XVH5oAoGZ7agRWZJXQpZI+MQyrklyzt3XDTFaih5mr0VVackd"
    "BPZGs/hXM11ZvXmUAje2+pPzQUVyLlpnSATFTHBmZhv9cn/7SuyAMGZma/XIfLNK7I1kXIuW2AYNuuKk8C8y7fdMrX/yl9kNOCcy"
    "idcTKt0Ta+yGRcjMW4RpDTrFSHA2ZB34nUf9WvshDAeY5sTidQP+bc7IZI8i5bRv1Q3z+aKqTgTMT/ug9XdLkLwCzDGvhPb/AFLk"
    "Mq5Fy1bc4PXBt1xVIwHmGTricAf5hyHeAGP1b8UAft3ImVci5ahF+BhBFuEVWvL/AB6BpioHo5ZwQxOXmOj7bFhH7Zwxcq5Ba9j0"
    "jyQCDppFUKy5xwTriwj9s5AOW+NiB9dh9M5Fyx5JctgXHA+aAoq32Jip/Y2xpxxYT+1dhoy0xnwxWfSuRcseQmWvZXMfNDrK4A+a"
    "KmOWOMyfuq/9V2AcsMZW+6m/7VyJlXIuWzsq4380Ig8x80VH7F+MQfunB/bOwBlZi5X2TE9uYB10wyrkpbhJ5j5oAvfcfNFT+xTi"
    "g/8AVJH6znbA9inE534nJ/Xc7YlkvUFs+Q+aEb8AfNFSHKbE/wD3Pp+e52wDlJiQ78Tq9852xUlyGW3skjQHzQ21tdfNFTnKTER9"
    "tidfvnO2EMn6/bXE6tf0nbFyrkFsgWHHzQ7ZvrY+aKl9h+vcMTKv07fbC9iCvbvCY2/X7YmVckLZA5wfNBuBrFRHJqtq/wCpf4+2"
    "GnJasK9tiMHr2+2LljyW5bMwtKWlknSx3Rp7CUISEAAW4RXEplXVKQ81MGtJf2FpUUnaFwDFie1QlJ4CPJXdmkjpFaXML43xU2df"
    "uXJDd9GV8kWy8b33xV+bssZqTkWrkXdO7qiUX9RWDAoCczmhw5d4fEqL1TfSKGwapxOZzPJpSVd0PCyjYffReJXMJKdllKx0LtHS"
    "pG8Ifwcr2ZsqUlI2lFKRznSHcIjONcVqwpTmJruNEzyznJ7Cl2tpf5ohns4vE60Ru3+YPZHKFGU9YlvYtkboUVL7OTtyO8jfwg+r"
    "COei0m3eRB/1B7I6eFqcDMi2la7oURnCeOJLFkqssp5GbbTtOS5NyBzg8RFct5q4lcqDbPdMvsKdCCOQTu2rRmNCblk9S5luXZwg"
    "ixMMSq4hyTrHFqzsW4SBeFYXhRrVObVI0+amkpSpTLK3Ak7jspJ+aJsU2tOeCnQxThzsqwAPeyQ86+2GJzvq20fqZTyOhS+2PQsN"
    "NmMyLmtBVuiscJZrVDEOIJSmP0+UaQ+ogrbUq4sCeJ6Isza0vHOpRlT8xU0xyTZOtoIIsbRXWZ+M6vheakWqa+22h9pS1hbQVqFW"
    "49ERyiZ2Tsq24iqyndzhVdC21JaCRzWtr1xqnQlUjmiHJLQufS/GCfiiq/Z0l9PqI55JkerEvwZjNGMZaZfRKKleQWlBSV7V7i/M"
    "ISoTirtDMmSbgIA3wr3EIRxNFXZ3D6HSbc7v/wAYlmWCirBVN2jc2X/GYi2doHctKV+W6PiESTKw3wXIa6guD98x7aH2JfyYe6Jj"
    "5YYvdBAFt8EgcY4o2NTqYeBCCYd5IpAdEK0GENYgABfmEG19IIAhaQAiAIFh1QT0CBa8AAbIh2nCAEi8OtaAEBYQt8KFbSBQ26zD"
    "gLcIF7QNoRAONr7oBsOeAVC0IEc0AG46YVxCuIVxACAH/wDYMDavCuLQAb21vCvDbAw4JF4AXVBEHZEK4iAEIi5g6XhaRQLYAELd"
    "zQgqET0xAG8IawAYcLQAbQNYNwIBNzpAAMNIgnpjGpSTAB2TwhBBvqYYlYG4wVKv99CwuZdNLmAQIxJ6IzIQeMAG0K0OtA04mABY"
    "DhCIuIBcHCADtbzFAdiFsC+sLaAHCATvsYgDsg8bRksEpjEBeHqG6ABcWvaG6c0G3RC2bwA2wg2HRBt1QAOqACEjngEQBe/GDYmA"
    "CEwtm0GygIGyYAN+iFpCsYViOYwAOoQbEQheFcA2JgBDTjBBvDQbq3GHacx80AIi8A6cYNtYRTeAGk35oQHVBCOiFsi4vv6IoDbT"
    "S14aTzgQ4pSOeGkDTfCwG6HhDgkW0EIDmBh4AEZsDFbgeMBR2d9h0xmUocBrGMk3vraAHbgIi+aCeWy5xCkjXuNR06FJiT8pbdp1"
    "xHswSDgPEPja9wOaDyRYLVB7FZ/yfhsz1bA0HIM3H66oueKY/k/6VCuA7+QZ/jVFzndGsT9xmY7CO6ANbxinZ2Xp0m7NzTqWWGUF"
    "a3FbkgRUys/mmq04k0srpZNkLCrPW/GI3a83xxinRnUvlQcki3rWhEAxx8OYto2K2Uu0ucQ6bjbaPiuN9aT8u6KpqefNYk6jNSzd"
    "LpyktPLbSVFd7JURrr0Rqnh5zbSRHNIu4AFMICxihlfyhK4k271Uvzr9aHf/AJB1tIuaTTdOlztjr4Gr/jJ1EXwYQ3RQw/lB1xZO"
    "zS6b1eObfvQv/wAg66okd66WD+09aHgqo6iL4gGKIOf9eGve2meZfrRZGXGNJrGtNmpmblGpdcu6lu7ROyu6b7jutGKmGnTWaRYy"
    "TJgTAG6CdTvjhYnxjSsIMMO1NbwD6ilAab2ybAE/KI4Ri5OyNNpHc4wiIgHs3YRBupc+P9N/vAOeOErHxqh8G/3jr4er2sznRYBE"
    "Ab9Ir5WeWE7aGofB/wDlEEXnxiLllcmxTg3c7N2Te19Pvo3HCVZehM6L8O+AoRXErnnhnuVozXdwf2E8psy42du2tvG3XjMrPHCZ"
    "1+qPwcetGfD1e1lzRLBG43gjebRXac8sKn72o9fIDtiSYTxrS8YJmFU0TAEvs7fLI2d97W1PNGJ0pxV5KxU09jvW8aOJjg/WhWf8"
    "o58kdzcoxw8b28Eaxc/0R2/mjNLzos9ipcjkA4pfHNLKPxjti+UI2r3ih8jiPCmYG68qrd1pi/EaC949WI+6yR8qAUJA3QLCCtQI"
    "te8N8scSi2b74JbhQb6W3xSjOT8sDZPXGTatAPPAgy0K3RD4HCKUAGmkDZh0AwA3ZtC2eiHdAhQJYYU6aQtkkQ/XdAtxigYQeEC1"
    "zD9IVgDvMQg23RCt0Q4m0CKBuzDtnSDpCtAA2eiCbgc0EaQDzwAxRJ54bsHjGSxvuFuMK1ooMexz3gbGsZDvhAc8CmPYhbPGH6wr"
    "QINCYVhxvD93PDTbjAAJA54aVcwhEG/RCCbRbAG11QCSYfs88AjywA20I9UO2TAKYC4zdCvpwg7OmsC3XaKQQhHqgwreaBbgtbhC"
    "taDpCIHNAg2whEQd3C0C14oBaEQOMHhAtxtEQGE8whb4eUwAI0QaU336wvPDoVoiANLQ0mHEeWG236RUUapXCGbt8P2dbwtOaAGW"
    "J3Awdk8RDr2O6HAX3RSGLYhwSNNIfbXhA2dYAbs2hEQ7ogEQsDHs7V4WyIfaBYb7RQNI0hW13CHkdUAiJYDdYGzDzbogX88UDbc8"
    "CHG0KABAtrvh1+qAddTrCwGqAhCwF4R3Q22kQpqVC7jJ2dpNjvEYlDQacIzzo2WTGFXtRHmr+h0iaz3VFdZkoU6KeACdlxSt/RFi"
    "vgc0VtmakKVTtrTx1eXSJT9S21RrYJuc0GNP6S9/8ovZO4aRRGCzs5nsG2pm3R8SovdJ0EdKvkh/ByRq1GkyFXaSzPyjM02hW0Eu"
    "puAd1456sEYZKfcKQ6PoUdwxwMaM1maoTrVDeDcyT41jZSkW1CTwMedSa2KyGY1nMGYdQ5KydDps1UN1ti6WvztdT0RBMOYOqWMZ"
    "8iVaS0xtXceKbNt9AtvPQI5uyZWpbNQQtQS4OWRuVYHUa7jF+UrE2FpWjsOys/Iykns+IhSwgptwKd94+ik6MMy1bOe7MmGsG03C"
    "kmUSbRU8pNnX1+3X2DoEUC0NmrIHM+P449EUjE1JxCXhTJxEzyJAWUgi192+POxX9V9OExvv+XHPDOTrNz3saklbQ9PJ3XMOG8Rj"
    "Rqjf0w8aeSPJPzM2th3RHOxCCaHUufuV3+Ax0Lxp1qZTKUqdmFsh5LTC1ls7lgJNwevdGHsU854cqzVDrkpUJiXVMNsK21NCw2ha"
    "1tdOMTuq5t0ioUqck26E824+yppK1cnZJIsDoLxwF42w9uOBaX6RUNGM8OqJtgWmHT+sXH1HFzSzR+TjfUwZaKvjikjndI/dMdyt"
    "Zu4hk6nNS7Akg008ttIUzc2BI33jDQsdUdisySpPB1PlXi8hKXUOq2kEkC406Yh2JgW8Q1IAbpp3T9YxrLnmlJaWJey0OlivEdbx"
    "H3NMVeX5Pk0FDaksFtJBNzv3xhw3Wp6ntPNSlHkp8FQUovyXLqTpwPARlxRiiv1+RkmavLBpqX+wkMKbv4oHHfoBCwZiuu4cM13n"
    "lg9y2zyt2FOWte27dvMdIxtBqyM7s6SsUVgb8J0jy0mNmTzTrFGUWmaTSZQKN1tolS3frAIiU4Lx7ies15mUq0ohmUUhalr7nW3s"
    "2SSPGOltI0avl0MSVZ+eaxVTZl6ZcKigWv1CxO4fJHmzJSyzirHSxZOGas5XKBJVJ5kMrmG9soSbgand5o6g3xrU2SbptOlpJoeI"
    "w2ltPUBaNgbzHhqWzPLsdFsVpncSJCln/Fc8mgiQ5VHawbJ9C3PJ4xiP52+5tLuP55zf+aIkWVAAwVI7I3qcP7xj00PsS/ky/MTF"
    "KeMIiDuENUePCORofe0C/RGPb64IVzxbAcFGHC5ho54cNBpEFgwrDn3QLkiFtWgA7oVwYBOm8QtTxEAEHWHQ1PSRDvKIhRQiOMIe"
    "SATrvEAK8DUQvKIPlEUgAINrwbdULzeeAEBeFsnmha9Hng8YhQWg7PTBFwIKb34QAAm0HWDfo80K/VACtCtA2iOEK5MAGxhWPGFe"
    "FviAVrQgnqg3twgg9EUAItzQhcwYAPXEAiIViOaHeSBqeEANUDaMfI3JMZbQ4dRggYQyOF4eGQN4jIIJMUDQnZ3QYRVAueaILC4w"
    "FIBMEX5oPkgLGIptqBCA6IeRfeIIA5haAMeyTCAPGMmidALDohpXpuigATbWDAF+YwSeiIBdMA9MHW26G2J0tFuA7QvvhX6oGyYI"
    "B5oAB3wb80I7oG1b/YRAOHVDrRj2odtwAbQLK5hCuTwhX4QAdbQgjpMEmwhlzewHxwA8AAwbQ0EnhCvAo6BtQiRzw0kGBAlQEAAn"
    "UwNL7xCBFt8AOIEDSGlYtvhFRtvMAPvpCJ1jHe/GCCRvigffohXF4aFHyQCDfSIAlKbxwceoSrA1fJGgkHfkjuG/C0cXHSfrIr+0"
    "bDuB7W/5MaitUR7FX5Afb9btv5FnX9dUXKTFNZAC09W9deQZ/jMXITDFfcZI7CUErRsqAUCLWIveIFinKbD2KEuTEiESE5cguy9i"
    "2pXMpI081jEkxfLVucojzFAfYl5xYtyjpIITx2TwV0mPPLk1izL2qm65yQmCdogm6Hen8VUbw1Ny1jKzMzfJNMHZOVunYiD8/OKk"
    "paVWFB6Vdst7Xck8Om+6ODUMnMXO1GaeRINKacecUgmZRuKiQTc9MWJl3mkvFj6adPSC2521+Vl0ktqA4q/E+SIzmJjnHFGrLtJC"
    "2ZVpWrDkqz4zyDuIUbm/AgbjHeM63UcW0mZaViQ4xm8P4OwOmVmpGniqOySZdpgNILhcKAkrva9gbnaiocB4eexHiaQkm29tsOpc"
    "eURcJbSQVE+a3ljs0XK/FWMJsTtQ5WVac1XMzyjtqHQk+Mfii0GhhfJ2mS4fEwpU2ooVMJb23HFJF9eYa6CNZ404unTd5MJXeaRD"
    "8+ZCTkpmjqk5OXlg4h4qDLaUbVlJtew1jNkNT5SaarRmZWXfILOzyrYXs+23XEcnM7E8rmC5Tzh6XnpoSTbhf+l1Ao2im26/NEdw"
    "1WMY4TL4pUtNsh/Z5Takyrate29PSY2oSdHI3Z+5L/VcleYuXVdnq/MVKUkJBuTcUEstsvNtnZSN+ybeMd5tFjZY0QYfwsxKuKb7"
    "rdUp99CVglBOgBtzACKTxDU8aYzXKtVOVnJjkCotJRJlABNrnROu6LcyhwjN4bo8zMVFgsTc2pNm1e2Q2kaX5iST5o44lNU0pSRq"
    "nuywDFRZ/K+lKNqfsj3yJi3DoIqH+UBrK0X9I98iY8+E+6i1NirsNt4ddmnfCR+fZYCPE7kQFKKr8b7haFikYZbWz4Nu1J1Fjyxn"
    "EgWOltm3lju5X4Kp2MqlOS9SXMoQwyHEllYSb7Vtbg6RkzWwLTMFuU9FNXNLTMpcK+WUFW2SLWsBzx9VVY9XLd3+Dk07EewujDLi"
    "3/CN6otI2RyPcaQSTfXavwtEjTL5V7u7cSejT2RGcM+DW294RqqSUbI5LuIJJvfXav0RIEexcCTtYlN+huFVPN/u/oEzI4xlXb7d"
    "xGOptPZELqvcSahMCmqeVJBw8ip4WWUcNoc8S9xWV2t/CXT8yIfP9yLn3u96XRJlZ5HliCvY4bVtLxqjo/X+plslVCby+XTGVVaY"
    "riZ630VLCElsG/D4ot3KqTw8zIzszh52fcacWltzusAG4FxYDriL4Hyjw/XcMSNTm1zwffSSsNugJuFEaCx5osfC2E6fhGTdlKcp"
    "8tOrDiuVXtG9rcwj5mJrQmrRb/qeiCa3Oykm+up6o4mOCfBKsC39Dc+SO4Bcxw8cXGEax/k3fkjy0vOiy2KkyRCji1zZtbuZe1fm"
    "0i/EpMULkab4ue13yq9/WIvwAmPViPusR8qEUAAG8ICCrdCFrRyNA4wSOMK44QtIAB0EKDqdYOzC4B5YEE7oFumKQGghQdm+toG6"
    "AFbTdAtBuIF7QAuMAwbwL7opAHWF16QbwL3gUR64G6HQ219YARJvwggk8YEHrECBuRxEK5gX6IV4FFfpgX1gk2gbQMAK0CCSOBhX"
    "ikBCtpaD0w0mAERv0gdcG8C4OhioAtCGkHqgbumKBKgX5oRIhoFzAg438kC46oO4QxWkS5RXvugQQLQuuKQB3QgbwiYIgAb4PNCv"
    "C3iAGwIJhu8xUwG0K0ECAemABvhQoV4ICJ0tAgjngE2ikAYVoSrkeKQOsXha24QANBANoJ1gECAYLgXgXEA688KxgUO2PLC2tYAB"
    "HCDzxSCuOcQL84gixg2gBsGwtAsLwdIARENJ54RNtYYVRSDuqFeGAm/VBBhcoTAMAmEItwLXjCg3PnhAgiBAQko54BOzuhXNowzR"
    "rVFwNtAEDeBGBYFhqIyTyNpAKiTqDaMa7FIjhX3RuGxpvEEGxB6IrzMR0hUi2HUt7RVqpN+HCLEeAsbRW2ZK9l+QSpxLaVbdyRe0"
    "ZgX1NbDQ5LNJgXsRPuD41ReyeEUTTvoea7SSbWqSh+8YvZOkdKv24fwc0QTMrF9cwquVVTxLCXmEkbbjW0oLG8b7bvniuH8x8ZVI"
    "7DVRfH5Mu0E/ILxfs1JS06lKZmXZfSlW0kOoCgk8+sOZlmZcWZZabA/ESE/JGaVSMFrG4cTzbM0XFNT5eoTMhUnxslbkw60rcBvJ"
    "McuSlFzk6zL8qhourSjbXuTc2ueiPU7oSptQdI2CDtbW7Z4+S0eYKglpuovJljdtLqg2RxAUbfNHtw2I6kstrHOUbalzYEy6m8Iz"
    "y516qIeUtBbWy02QlQ4ak8D0RTrwKKw5Y+1mFb/z49KyJWZGVLnt+Sb2r8+yLx5rnPFrD9tLPq3/AJ5jFGcpVnmNNfSenGz4oPQI"
    "yAxiZP0JN+YfJFc5uYgqVGepgp06/KqWlxS+TXbasRa/PHjcc08vJu9kWWSIa80h5tTbiUrQsbKkqFwR0xB8q63VK5SpuYqU25NK"
    "Q+G0FdrgbNzu64msy0t+WdaQ4WVrQUpcAuUEi1x1QrU3Tdgnc8+5gTshMYlmm6fLS7EtLq5FIZQEhRT7ZWnTeJThDEuBKbQ5eVqK"
    "G3pqxU6pyT2/GPAG24RlOSC++TanKsHZNSruqCNl23RvF7xp44y0omFqC5UWZycU/wAohttDqk2USdeAOguY9calPpqDbMNO9yXU"
    "WuYBrFUl5SmyckZxavoYMlsm410NuiKcxSL4kqZJ3zTv8Rjv5Ry4VjFuaJAZlGXXlq5gE2+eI1OOmq1xxaDt90zJI6dpWnyx1p0l"
    "Cssr9CN3WpaObUm/NYWoy2WnHUNW2ikE7N2xa8V/hLGFWwc5MGQlml90hIWHUKPtb2tYjnizMw8a1PBr8hKU8SxSuXurlW9rUHZ5"
    "90Kh40qVTwJVq46iVRNyqlJa2Ghsm2zvB37zHKE3GEpWurlaTsiKTWcOJJmXdl1SMkjlEFBUELuLi1xc9MYMqsPzc9ieWnly6xKy"
    "l3FOKTYFQFki/PeO1gfMOt17FErTp5cophzb2giXSk6JJGsWVN4ko1Ome5Jypyks+AFcm6vZIB3QlJ0tFHVhK51E6JEERil5hibY"
    "Q9LPNvNLF0uNqulQ6DGUa3jwPfU6IrbOz3Lpp5n16/qiJDlST4GSV7e2c/iMcDOvSj04nd3Sr+CO9lTbwKkd3tnP4zHrw+tGX8mX"
    "uTHXohqlGHXGsMKrkRzSNAIJ4w4DS8C1uMZU32YrALkQRfnhHpgXiAcDaETeGjnJh1rRAKALwRrBigF+a0CCPJBiFACd8K94NhCt"
    "bjADRc9EOAuYMIxAEaQgLQLiHX0igFoVjBvfhCHVEsQUEAwgQOuEPJAoiT0Qjc74UK+m8QAtRzQNb88O80IQAOmDeFaFeADaBB1h"
    "b4AXCD1wIO4wArwbjmhsGIAX8kEK6oXVC0MUDgYBMDTntDSrXfEA4QTDQu++0G/QRAC2wnhBCweEA24wNkW0MUDtsQ0uaWEDZtAi"
    "ABVfUwt4trC64W1pvigcIRgAjnhC/XADr6QDfngmBa8AIdEHovCtAKbjfAC2b80DZhwSqFrxgABAOsIiw3Qd0G40iAakaaw4WPRA"
    "JCoFgBpADiIVr6EQ23TDgBxgBBJ5oQHRC0toYGt+MAHhrAJAgGAQOmKBAg80LZUTe4hAQ7aA6YgGKRaF18ISiTAteLYB80PSAdbQ"
    "xPVGUE20ESwDa43QtN0AlUNIUd94AICQdSI4WPQHsD19FtO97x8wv80dsAX1McrGI2sH122lqe+f3DFjug9ipsgSTUK0b3+l2f4z"
    "Fzk2il/5P1u+FY0/ozP8Zi5zFxP3GZjsFRNtLXjl4il6O9R5k11thcg2nadLybhA3X5x5I6mluuKvzzxMiQoTVCaV9MT6wtwDg0k"
    "3+NVvMYxSg5zSRJuyud7CM7gimIEjh6oyJW+r2vK3ddVwGup6orzHeLMayGMKlLyDk4JWXfKZfZlQoJTYe1UUnz3jUyRoa6lig1J"
    "xN2KcguX/wARWiR8p8kXpU6tJUaTVNT823KsIGq3F2Hk5+oR6JWpVGorMR6rU89KzAzITc911S3H6U/4RHMTYyr2IENS9bnXZgS6"
    "ipKHEBJQoix3Ac0WDjrO6Yng5TsM8rLsqBSqbVcOKH5A+9HTv6oq2epNTbkGqrNS7yZaZcKEPufzqgLm19T1x9KivWcUji3wbWH8"
    "UVfDbzj1HnHJNx5IQsoAO0Abgag8YlSMcZmr3TdZI/y3/GIfR6BVK03Mrp0quZ7lb5V1Leqgi9rgbzFgYEzkqGH226dWkuzsmjxU"
    "u7X0Voc2u8DmOvTErrdwimyx9zURjDM1SwDNVu3+XPqxf9JW87SpJx/a5ZbDanNoeNtFIvfpvGvQMUUnEjHLUuoIfFrqQFWWjrSd"
    "RHUMfJr1Mz1jY7R0QTFQfygTaUouv8498iYt4nnikv5QE8lc/SpNKtWmXHVD85QA/hjWDV6qM1HZGLIFYTV6sd5EoD+/DMX5mYSx"
    "Wltqq4fqLipcqDbjcwlCk3Ovk0jYyAlSqbq79js8i235Son5o4WbWG6DhidlpSlS0wl9SS6+txwqTY7ki+l+MeuKi8RJPczLyo5X"
    "fLLwpH1Cr4/1yPVgJqOXpV7hV7T/AM5Hqxv4Vey4nkNsVunz0g/axdTMqW0o8+66fjiyKdlRl/VGeXkULmmj9+1OFQ+KOlSrGm7S"
    "UvyFHgqhVSy9P4Brx/1yPVjIirZfNlJGH60eufT6sWRW8v8ALTDrZdqaixYaIVNqKldSRqYqHFc3huYmg3hymzEswg6uvvFanP1d"
    "yR8cap5amydv5Mt2LYw1nRhqXRK0pqlzVOk2wEJWVpWlA5yBqekxaUrMszjSH5d1DrTg2krQbhQ6DHmjAeXlTxZNBSWyxJJP0SZU"
    "NB0DnMejqHR5SgU1inySSllkWF9So8Sekx87FQpwlamdo3auzoDSOFjj7kKzf+xu/wAMd4eeODjkA4RrA3fSbnyRxo+dCexUeRht"
    "i525Gkqv5ov1KwYoHI4JOK3TcXEsvr3iL7T8UerEfdkWPlRl8UwtN2+GX1EO8kcTQRa/GDYdMAamEVQA4WhEgDSGk3gWMCB03/PA"
    "FoVoHHUGKUde8MNuaHWhW0gQaSeEKHWhpECivA3wtkmFskWgQGg57weO6FYcRBgAEQNwh1hA0G+KAX6IV+iDe/AwLawICETaDbhr"
    "CIvAAuDvgacwgwrQAOOsI2HNC46wvjigF+aFr0QbdEDSAAd8K4hEjngRQK8AwTugb4EQDBFh1wrawj1xSgvAJhEwN8SwuAmFtHjC"
    "IvuhHTqikBcEwjzXg3HlgbINjxgAXIg7WkA3gAbwbGACTeBe0G3NC+OKBE7uENvDj1QD0aQAtoDjAKubfCtffCtCwFc88NvDreSG"
    "lOsAC46YBcHG8PKbnfAKem8LAYHUnjALgvuPmh5IhdEUgzbA4QQroMO2R0QvJAXG313GASbw4+aGkwKAX4Xg69MK4hbXXAgCSOeF"
    "c+WESLQOEUAUbRjKjzQ9WmkMNuMUCB1g6mEBCv0aRAK0I6c8KFe3C0AC/XA8hh14RN9AIMDFaa6wAo8QYcRc6wdnoiA1J0jYTcHf"
    "GFe6NicSChJsCbxgUeiPPW3R0iaj50N4rfMVKXZynpXsW8b20WS/exissxllFSppQoAgL37vLGI+ppbmBshrNlN+FUtf9b/eL3tw"
    "ih5u6M2zw+qgP7wi+AdfLHaprSgcluRHM1FXdoLaKO3MrdD4W4WL7SUgHm1380VrK5o4so/0u+6h4p02Ztrxh5dDF8HXjGGYkJWc"
    "FpmVYfH+I2FfKIxSq5E01cSV9igq5mViKvy5lXphthleim5dGzt9BO89UdDAGXs7WqgzPVBhxmnNKC7rFi8Qb7IHNzmLml6FSpZY"
    "WxTJJpY3KQwkH5I3uaOyxSirQjYmTkRFh5Y81YnkJul1ybZmWVNL5VZFxvBVcEc4j0sYwTdOk59KRNykvMAbuVbCrecRxpVXCeYr"
    "jdWKbk86qzLy6GnJCSfUlISVqKklVhbWxiNYqxbP4xnWXpptpvkk7DbbINgCbnfqTF/eDFD/ALnp/wAHR2RkYw/SJV0OsUuSaWnc"
    "pLCQR5bR3VenmzZdTOR8nAywo71IwowiZaU08+4p9SFixANgLjqHxxJao66xTpp1gkOtsrUk8xCSRGwNeaHEg7xHmqzc5ORtKxSU"
    "pnPiCV8WalpOa/OQUK+I/NEfxfjmpYxdZEy22wwzfYZbJ2QTvUSd54RfVQwxRKtcztLlH1fjKbAPnGsasngTDMg+HmKNKhwahSgV"
    "WPlJj0QrU1ZuOplp8lSsIVg3BcwtxJRVK4nYQg6KalhvURw2juiOYamGZCrsVKaln35eTUHlIbTfaI9qCdwBPGPSFQpMhVGeRnpR"
    "iZbPBxAPx8I1aXhSkUWVmJWSk0pZmTd1KjtBXRrw6IscVbM2tWTIUDWqvU8c17liwpx90htphsXCU8Ejt8sWscNHDOWNQkFkKfMu"
    "p14p3FZIOnQN3kiU0fClGoK3HKdINMuOElS96uoE7h0R01soebU24hK0LSUqSoXBHMYxVr5oZIKyLGPqeaML13waxDLVRcuqYSwV"
    "EthWyTcW3+WM+McS+FVbcqKJYyyFJQgNlW0bJFtTF4OZd4VdcKlUSVudTbaHxXjJLYDwzIvJeYosolxJuCoFVvISY6+Kg2pOOqJl"
    "ZlwTKrk8J0phwbKkyySRbdfX547Y6ICQABbSCn/6Y8dSWaTkbSsVznZ7hyB/8o/wR2cp1lWCpQD71x0fvRyM6gDQJE800f4DHWyf"
    "FsGs33F5y3nj04f7Mv5I9yaJSVQ8NhJhBVt0DaJ1jmaMlgIBUBDLm14F9IWA46woZfheH2igIvfjDrdMMKuAhXPPaIB9oXlgXN4V"
    "zzwAYBvbQwrkmBfqgAg9MEHogWvB1EQobwoFzzwRpAgdIPyQ25vrB16IpQ3EHyw0kg8IO0OiIQO+CLQ24g7UQotkcYdYCGlX/wBE"
    "G/VABPkhAQL9AhAmADwhcYBMG/VABELpht1Qb6wAbQtYW1aFtEi5gBC8LUwgYW1pugA2MAAjjCCjCJPMYAVtO2AU80ILO60EqIF7"
    "QA1SFbuEOFxASo80AqMQDt++8CBcwrnmigPlhXsIG1A2oAJPRAFjwMKx32hXgAiwh8MBPNBv0QA6w54VoFxAKrcDADhC0/8Aphm3"
    "rC2uuAHkmBtQ2/XCvCwDe0K8C/QYR03boWArC8LphAwtrogAwYF4QMAHrg8IbtQLwATAOsC8K+loAWzC2RC8kHptAAtbjAseuD5I"
    "Q13wAkaaxkCxGO44QQYAftphpVfjA0vrC8Uag74AAJ3bzwjk44ujBtaF7EyL38BjrpICri0cfGhS7harIvcGTeBP6hix3RHsVRkB"
    "pUqwLf0Vr+OLpMUp/J/P1Vq9v7I3u/SRdVuMMSv3GSOxzcR4hkcMUp2oz7my2gWSke2cVwSkcSY8w4hrNQxliN2efQpx6YWENtI1"
    "2U7koT8nXHqioU2SqkvyE9KsTTJ12HkBQvz68Y4dKy6wzR6qmqSNOSzMIB2RtEoQTxAO4xuhXjSTdtSShmZwabhWvYMwGJbD6JVd"
    "ZWeXmQ6LkqI9qjgSkaC+m+KhcpmMsc1ZxEw1Pzs0lWyvlQQlk8Qb6Jj1DYXAMApSL2SBc3Om8xKeKlC7tqw4JlV4LyPkqWUTteWi"
    "dmBYiXT9iSek71fJGv8AygWkN0KitNpShCJlwJSkWAGxutFup3WjhYrwdTMYsS7FT5fYl1lxHJL2TcixvoYzCu+opz1DjpZFU/yf"
    "0EVipn/xE/xiLAxdlXQcVbb/ACXcU8rXuhhIG0fyk7ldeh6Y38L4Do2EHXnqal/beSEKU65teLe9hoIkelhCrXbqOcdBl0szzzNZ"
    "SYzoFQDtLSqYCDdExKO7JHkJBBi4sDyuJZalgYknG33iBsJ2RtoH5ShvMSQAG8K1olXESqK0iqKQFmwuTbpMeX8yK6MR4wnZtlW0"
    "whQYZPOhOl/KbmPT7qAtCkKAUlQsQRvEROdyvwvUFlx2lNMq/GlyWyfINPijeHrxpNyaMzg2QXL3EtGy9w6hVYE01MVJXdDYQwVb"
    "bQ8VJvu37UblezmwnUpZUu9RJmotfivIQkfGSRE0xdgKm4tpUvT3byvcpHIONJBLaQLFNuYiI3L5D4da1mJuff6ApKAfMI6QqUH9"
    "U73DUnoUjXZylVGcL1IpTlOZ4tKe5QeTQW6tY16dMVSWfvTXZpt06fS6lBR97HpKQypwhTgCikIeUOL61L+e0SGSp0lTVJakpBmX"
    "TYm7LSUhPmjs8ekrRj+TPT5Z5In2pwOqM4l/lTv5YHa8t9Y7WEJ/DlMmA9XqXNzxBulLbqUo8oOp88enp+lSFSTsTslLzKTwebC/"
    "liNVDKfB9QJJpSZdR4y7ikfFuh46M45ZL8Dp21RyqXnNg1LCJdCJuSbSNlLfc3ipH6piUUHG1BxFMGXpc8H3gkr2OTUkgDedREOm"
    "8g6G740pUZ6X6FBLg+aJNgfAspgmWfQ053Q+8obT5TsnZG5NuGuseSr0cv0XudI39SVhV+qI7jzTCFYN7fSjnyRIBfdEex+LYOrO"
    "molHPkjnR86JPYqbI1P10vn/AMdXyiL7R0xQ2RVjimY5xLKPxiL6GsenEfdkWPlQbEkWEZNk9UNSbWjIVjdvjiUaYBhE8wgAmFyh"
    "43g7QgCFx6IEFe+6EYGvNC15opRdEKASeaET0QIHqhjm1snYICuBULiDtdEC9zwggYZN2cUFpm2mkKSrxVNKJChz2OoPRGxtdEDT"
    "ohEQArmBr1QRaDFAL6wgYXCBtQAdIWkAm8LagAgQIRVAJvAgYXVpAhXgBHyQt2+BtQLxbAPDfDSRxEG/GBtCABswbeaASLQLkCKQ"
    "J64FoV+sQooFAtfjBMAdMQCt5obYdcOMC+kUAgX6IN4ROkADS8K0I9cAqtABMIWMDauIbtWgB9t2sC0AqMLa0igNhuhWAhoMAqgL"
    "h0ELSNdUsyXw+pJLgFgSTYeSMu3C5BxMDfA2oG3AWDcQjChXigAAhXAhXgXgA74EC/TCv1RQHS8AwDvgQAjrAtfjDgeEAnTQRBYa"
    "RY8IXCEbmAYoAT1wBCtBA03wArDmgcN0I2HHzQ0Kvx3RAO6LQoF+EG4HGKLCuN0EHfDNoE7/AIoO0k8TC4sE74G2TBuCIIAtoDEu"
    "U1p0WbSbffCNZcbVS2QwFXtZQjVXzxwrehuJqv6pMVjmKgOVWnpub7Cza9tIs2Y9qYrDMNezWJE79ltRt5YxHZmlujSxLPN0rNCZ"
    "nHgotsTwdUEi5IBBiwfZpw7v5Co+iT60QOttofzZdQ6gKQuooSpKhcEEp3xdXg9RwT9SZDQ2+10dkd7xVGGZXOOtyJjOfDZH2KoD"
    "p5JPrQ4ZzYaP3k/6IetEr8HaMPwTIfB0dkA4ZoitTR6ef9Ojsjnmp9vyWzIuM5cM7rT/AKAdsOGcWGAd8/6D/eJL4L0L+5qd8HT2"
    "QDhWgq0NGpxH+XT2RM1Pj5FmR32YcMWvtz3wf/eCM4MLm30WdHXLntiQKwnh86GiU4/6dPZAOEMOnfRKd8HTFzU+BqcIZvYXP89O"
    "Dp7nPbDhm3ha4u/NDrlzHaGDsOf3FTfQJgeBmHCTeh07Xf8AQBFcqfDGpxjm1hb+0TfwZUP9ljCv9qmfgyo6pwVhv+46fp/hCD4E"
    "YZI9wqfb9EIl6fDGpyk5s4UUPt18dcsuHDNfCXGfe+Dr7I6HgLhgjWhSOv8AhwBgPC97d4pH3n+8L0+GLM0fZWwnoe+LnwdfZDjm"
    "phMG3fJfoF9kbhwBhZX4DkventgHL3Cx/AUn709sL0uGLM1U5p4SI0qavKyvshwzRwkT7qedlfZGf2PMK21ocp5Arthvsc4T2ie8"
    "ktr0q7YXpcMamP2T8Jk+6qR1sr7Id7JuEyPddHol9kOOXGEz+BZf3y+2Gry3wpbSisno219sL0uGNQ+yZhPZ92GvRr7IKcy8JXt3"
    "5Y94vshnsaYV2Rais677uL9aEMssJ3J7ztXt/WL9aF6XDGpDc1MX0KuUSWl6bUG5l1EztlKAoWTskX1AjoZc45w9RcLy8nPVFDL6"
    "FrUpCkk2ub8BGhmfgqhUGhMzNNkBLuqmAgqC1KunZJtqTzR18B5c4aqmFJKfn5AvzMwgqUsuqFrKI0ANhoI9VFw6UrXtcy02yRIz"
    "QwkRc1hvyoV2QfZMwmpVhWWb9KVdka7+VmEHV3FMLQsPFQ8u3xkxg9ibChV9pPenVHO9PhmtTf8AZKwnbWtMeY9kO9kbCZNzWpb4"
    "+yOeMosKf2N8dUwqHjKHCWh7imPhCoXp+41OmjH+FlAKTWJex3HWHjH+FyrZ78yu0eFzHORlLhZtJCJWZSDvtMHWG+xDhUq2u55q"
    "/A90K0jN4e5dTpqx5hhvVVZlk301JgjMDCv9+Sfvo5buUWF3zdxmbJGn2wYxHJnCZ05GcH+oPZBOn63Gp2Rj3Cyt1ckj+vDhjvC/"
    "CuyXVykcP2F8K/iTvwj/AGgHJXCt9Ez4/wBR/tF/b9xqd4Y4wwT7uSXpIcMbYZIv38kfSiI8clMLc0+P247IByTwuRa9QH7cerD9"
    "v3GpI041w0RpXJH0oh/hhhwjSuSHphEYOSWGOCqh6YerDTkjhn+tqPpU+rD9v3GpKhi7Dx0FakPTCHeFmH/75kfTCIj7B+Gifs1R"
    "9Kn1YXsH4bIty9RH7VPqwtT5Y1JgMT0IkWq8j6dPbBGJ6GTpV5H0yYh3sHYcH9IqPpE+rCORmHD/AEmogfnoP/xh+3yxqTMYlopP"
    "utIn9snthHElG/vWR9OntiFqyKw6b/TdRB/OR6sD2CsP20nakPKjshanyyfUTcYhox3VWR9Mntgiv0c/hWSP7ZPbEEGRNCG6fqIv"
    "+Z2QjkPQ94qNQ8yOyFqfLL9RPhW6ST7pSfpk9sO780q9hUZP0ye2K9VkPRlfhSfH6qOyGqyFpH3tUnh+oiJanyxdliir0sjSoSnp"
    "U9sHvrTj/T5X0qe2K3GQ9JH4WnvRoh3sD0jf32n/AHiIuWnywrlj99Kd/bpX0qe2F30px3T0r6VPbFcnIikkW77z3o0QRkPSQNKv"
    "Pe8REy0+WNSxe+lP4Tssf2qe2F3zp/8AbJb0g7Yrf2B6Xe/fie9GiErIamn2tZnR+yRDLT5Y1LJFSkbaTkv6QdsHu6S0vNy/pBFa"
    "nIiQ2QO/U96NPbA9gaQ4Vud052k9sXLT5Y1LK74SX9rl/SDthwqEmdRNy/pBFYnIOSUoHv7O6a/Yk9sOOQsko616dsf8JPbDLS5G"
    "pZ/d8p/aWPSCGLnpVR0mmbfpB2xWgyGk0pCRX57T/CT2wE5Dyo3YhnfK0O2GWlyLyLLE7K/2tjyOCHCbljvm2PSDtisFZByqjpiK"
    "dB/Qp7YK8hJZQA8IZwW/wR60MtPu+B9RaCZqWt9tM6/4ggGYlrfbLPvxFXDIVpNwMSTmvOyPWhoyGSlRIxNN+hHrQy0+74H1Fpd0"
    "SxOkwwepYhGYYG99odaxFWKyFSTpiWcHUz/yhHIULFlYmmj1sf8AKGWn3fA+otQPMndMM+/EO5ZjjMNe/EVSjIJKb2xNNWt/Uf8A"
    "KG+wGAbpxPNA/of+UMtPu+BqWvy7B/n2vfCAXmr6Otn9YRVS8hSrdiiaH7H/AJQBkIq9/CmZ9EfWhlp93wNS1g62oX5Zv3whxW3/"
    "AFqPfCKmGQawdMVTPoj60BWQj5P3VTPolevDLT7vgaltco3b7IgW/KEIONf1rZ/WipTkPMnTwrmvRq9eD7A0z/3VMn9RXrwy0+74"
    "H1FtBbd9HEe+ELlGj/OI99FSHIebv91czfoQr1oHsDTn3uK5kfqL9aGWn3fA+rgtzlmtPoqPfQuVbA+yI0/KioxkRUB/1ZMe8X60"
    "H2C6lrbF8zb81frRMtPu+BeXBbZdb4qT54XKNn74eeKj9g+qjdi+aA6A560OOSFUI0xfNX6eU9aLlp93wLy4LaDjR+/T76Ftt/jj"
    "zxUZyRrO8Yxmf/U9aAckq2f+sZj/ANT1oZKfd8DXgtwuNf1qB5YBfaFgHm/fCKiOSVc3HGL/AP6nrQDkhWT7bFzpPSHD/wDKGSn3"
    "fAu+C4QpFr8ojziEFtje4j3wink5JV1O7GDwHMOU9aB7CdfJ+65wdN3fWhkp93wLvguEuNJ+/Tb86By7O7bRf84RUYyUr4T91zpJ"
    "4ku+tCRkrXwhYOLlEqtYkO3HV40TLT7vgaluco3wWPPCLrZ3LT54qP2F8R7vDF23W760A5LYiv8Adg5brd9aLlp93wLvgt0ONn+c"
    "T54cFtJ3OJPlioPYWxGbfXi75S760H2F8QnfjJ3ycr60MlPu+BrwW2Xk39snzwC8j8dPnipF5K1//vBzyh31oPsKV633XL8od9aJ"
    "lh3fA1LaDyBvWjziDyiNPHT54qE5J4ivcYwV5ne2EcksTjROL9Ot3ti5afd8C8uC3DMMg2LrQUN42hHLxTMMnC9Y2nWwDIvgHaG/"
    "YNorFWR+ICvbOKklR3qs4ST545WKcqa7QsPTtQmsTGZl5ZvaWyOU8cXAtYm0WMIX83wRt2Nj+T45erVdOtzJNHd/iRdxB/FV5o8x"
    "ZdYRqeLKhNs06rqpq2GQ4tY2vHSVWt4pHGJ/7DuJ7WOMnPO760axFOm5u8vgkW7bFvkE7kq80IA79k+aKiGT2JkgWxk57531oKco"
    "MT30xm553fWjz9Ol3/Bby4LcsSfanzQSDzHzRURyfxMbfXk6PK760P8AYhxOAPrydvz3d9aL06Xf8C8uC2Qk29qefdCsb+1PmipU"
    "5QYn3nGbpHW760JWUWJri2MXR5Xe2J06ff8ABbvgtk3HA+aDY/inzRUqso8TW+7Jzzu+tATlFiZI+7Jzzu+tDp0+/wCGS74LbTfm"
    "PmhEE8D5oqT2IcTXJ8MnfO760PayhrynPpjGU1sW3t8oT8aoZKXf8MXfBbBvxSrzQNk7OoV5jFTuZP14KPI4xfKeBXygPxKhoyfx"
    "Hv8ADF0+V31odOn3/Au+C2EhW1ax80Ig8x80VMcnsRKP3Yuj0vrQvYfxCd+L3f8A1fWh06Xf8MXZbJBt7U+aEEnfsnzRUoydxDbX"
    "GDp9L60E5P4iUADjF6wFrfRNB76HTpd/wLyLZ1/FV5oSknmPmipTk5Xjp4YPC/6T1oQyZrgFvDB4+Rz1ouSl3/AvItoXtuPmhpBv"
    "uPmiphkzWzr4XveZz1od7DNZ44uevx8Vz1onTp93wLy4LaAt96rzRHsfqtgytKA3Si+EQhvJaqEgO4umdnjspXf+KORiPKOqUuiT"
    "s8vE78yiXaU4WlJX44HD21o6UoUs6+r4JJuxq5ELQnFL5UQE9zK1PWIvsvsj+cb98I8s4GkxNzE0lTqmtlvaCgTz7ok66aVgqSp5"
    "QSNxJ1i4pwjVd2ahdxRf/LMk/ZG/fCCHWP6xv3wjzU62oKUgOOJ6DeGJkis27pdT1Jv88cVUp8lsz0yXmf61v3wgB9j+sb98I80N"
    "09BUU92TG0ND9D/3jIKWViyZt5XWn/eKpU+SanpTuhng42f1hA5dm2jrfvhHmtdNKE6zj3Vs/wC8A01ZTbut/wAif940nT7hdnpX"
    "l2j/ADiPfCEZhob3Ee+EeZ0SL7bgUiYfJTqDa1vjjJOtzE9MqmJqYmFvOaqXsg7R5zrvgnT5JdnpPl2rfZG/fCG90M/1jfvhHmZ2"
    "XSkXD80enkxp8cYzLbQCuWmwOJ5O3zxf2+SOTPTgmGD/ADjfvhC5dng6374R5jVLDlNnl5q45kDth5kkm1pmZ95/vD9vuF2emC+x"
    "xdb98IBfZ/rm/fiPNSpFJbCe6prXjyd/nh7VMS2sKEzN6a6tf7wvT5F2ekTMNA/Z2/fiB3U0P55r34jzgqmoUSozU0EniGx2wW6S"
    "yR4s7NdZbHbEz0/Vluz0b3Uza/LNe/ELupj+va9+O2POopDAGyZ6av8Aoh2we88qB409NAjmaHbDqUuS6nonupgfz7PpBC7ql/69"
    "n0g7Y87d65AD7fm/I0O2MQpkmSfpyaI5+THbDqUuRqeje7Jf+0M+kT2wO7ZY/wBIZ9IntjzginSDi7d1zhtxS2ntjMqj08jWcm09"
    "bY7YdSlyNT0T3bKg/bLHpE9sDu2V/tLHpE9seczSJMG3LTak8+wmHN0mnEXU/PWPMhPbDqUuSanokz0qP6VL+kT2wu75T+1yw/ap"
    "7Y87LpNIN7TU/f8ARp7YwKo9HWqwenib2PiJ3xepS5LZno/vhJ/2yV9KnthvfCS/tkt6VPbHnJVBpezYTE9s8bpTDUYeo2l35zyh"
    "MOpS5ZLM9IGpSA3zsrf9Mntgd85C2k7K+mT2x5vVQKOATyk4RffdMBNCpK7JS7N7ri6k2h1KXLJqej1VanjfPynpk9sA1an8J+U3"
    "f16e2POYo9NR9DSt4kcSEn5ocrD9P3FbxPUnsidSBbHog1inD8ISnp09sA1qmcajJj9ujtjzt3ikALlT56PF7IxKokiskJS8SNxJ"
    "T2ROrAh6NNbpf94yXp0dsNNcpY31KSH7dHbHnVdEltlKeSSnzRgcoiL2SCPkidWJbHo81+kj8KSPwhHbA8IaQfwrI/CEdsedBh1v"
    "ZASvcN2nZGPwcJ1QtOnPF6sSano44ho4/Csh8IR2w3wjogGtXp/lmEdsechhtxaikqRca80O8GgBqUk894dWIPRRxLRP74p/whHb"
    "DDiah7zWKfb/ADCO2POrlALJCd5PltCRh9ezZWyb8L2i9WAPRPhTQf76p3whHbAOLMPj8NU34Qntjzv4Og38QAjphisO2I0Av0xt"
    "VaXuTU9FHFuHv77p3whPbAOL8O/35TvhCe2PPPeNsJveHoo7LZAUCq/NeMutT9Lmj0D4X4dI926af26YBxhhwb65Th+3TFFtUeUs"
    "FKaCrjioiMyabI2O1LJNuBWYz1oAus40w0Pw7TfTiG+GuGh+Hqbp/jCKUFNp/KW7jQQeBWrtgmQpyEm1PZPTtK7YvWp+4sXT4cYZ"
    "H4dp3phAOOMMj8PU70wijV0+SSqwlEk+XtjO3J06wKpBgk9Ku2L16fDFi6fDrC/9/wBO9LDVY9wuNDXpD0kU2mSptyO9kvYc5V2x"
    "jdkKeVgJp0ukHpV2xevT4Ysy5DmBhYfh6Rt+fCOYWFR+HpD3/wDtFMrpsgrUSLItzbXbB73yAGwaXLq43usH5YdenwyWZcnsiYUG"
    "vf6R98eyGnMbCY/D0n5z2RTRpFPWn7TbTfmKu2B3npqrgSiE247Sjf44vXpe4sy5DmThMfh2UHvuyB7JOEv79lf3uyKcNFkErRaU"
    "aIvxKrfLGU0qnAlRkGSAd11D54delwxZlu+yVhQi4rTBHOEqPzQPZJwofwwz7xfZFUdzyCCAKRLniLLX2xlblqVspUqlS+/Xx16/"
    "HE69MWZaCszcJJ31lvTjya+yGqzRwgke67fo19kVg9L0pYUhulMIP4wWskecxqJpcqlYOyDfXWDxEOC5S2fZNwq4CUVQH9kvshvs"
    "nYVB1qRv+hX2RVqJKTbWFllLiRvQomxhzzVPF0ppEsfytpR+eJ4iD9C2LRGaGFeFQUeplfZGNzNTCw3zzt+hhfZFXhiRCkkU6V6j"
    "ta/HGVHcIWVGlyIt0HtieIhwTKWQM0sMKOky/bpYUIyDM/DVriZfPUwqK25WV2wTTpHTW+ye2EuelEHa73SRvp7Q9sR148FsWV7K"
    "uGUGypl/mvyBh6s0cNBO13Q+RzhkxVzs+yqyBISgH5KN8dimmVmGwhdOlk9KRaMvELg2oE3Tjuj151MlIOvLeUQbKbIFgY7bigdI"
    "qDDLiJfG9kNpbQSUpSkaCLdXwPRGKs25WEVoaswo7JirsxXeTrEqNkEFk3598WjMe1OnCKtx+EKxBKXcDZSwVBWzc74sXo/4Ktxt"
    "aP8A/lhZt+EUfGpMXyTdR6zFC1yxzacJ/vBo/GmL5++PWY6zf7MDktwnUxqO1qmS6y09UZNtxOhQt5II6wTG0CdpPWI8x4iua1P3"
    "390O/wAZjFGn1J5RJ2PR4r9JO6qSPwhHbB7+0okWqcj5JhHbFMSeTlfnZVmYTM05KXUJcSCtVwCLi/ixl9hPEH9qpvv1erHZ0aSd"
    "nIl2XJ35pn95SR/bo7Yd34pxtaoSfp0dsUyclMQ/2mnH9ofVgHJXEYGj9OP7U+rDp0u8l3wXQKrIEaT8obf4yO2HIqMmtYSicl1E"
    "8A6kn5YpL2GMTC4C6cejlv8AaIYhDspObO0ErbcsSOBBjUMPTm7RkM7XoeqbiFpGCVVty7SjcktpPnSIzHQGPJJZXY6LU15+pylK"
    "lVTU9MNy7CSAXFnQE7o06Ziyh1ia7mp9Tl5l4pKuTbJvYbzuip8y8VVt2eqNFc2e97bwKbM2OliPGiHYdxHUcPVETtOCOX2FI8dG"
    "2LHfpHqo4bPHM2YlKzPT5MceZxlh6UfWw/WZJp1slK0Kc1SRwMVCc3cYDe3K+WU/3iGVOozFRn35yZsHn1qcXYWFzqbDhFhg239T"
    "0I5nqRE5LqlRNB5HIKTthzasnZPG/NGqK/SP71kPhCO2KcksXVSvYJrchOBkMSUoyGuTb2To4kam+ukRLDtCfxJWmaZLuNtuPbVl"
    "ue1AAub26oRwl5NN7DNoekFYgpAPurI35u6EdsZW6zTHiA3UZNZO4B9JJ8l4qFWR9WIAFUp5J/IX2RAHpdUjPOMLIKmXCgkc4Nvm"
    "iU8PTm7RkHJpbHqpKri4hAAmNenq2pJhXO0k/EI2BaPLJWdjaZA85B9azWv9LR8io7WW6vrGpRH4ih+8Y5GcKb4TSeaab+eOplmo"
    "HAlM6l/xmPTQ+zL+SPdEoSQqCALxrhRSbgxmbJcN90YKZUgxkt1QACBC2jfdADwdNbQBvgXPEQ4XiFFbWCYBNoBUbxLAN4Nyd0NF"
    "zDoAXl8kE6wLwCYWFw30gXIhEEjfBseEAIKPGEVdMKx54UALW8EE9ECw4wrDywA6554RJhCGk66RChubwgPNCGvCHW0gBX64Vue8"
    "LdCvfjAg20GFBtAooWyefSFB1IgACHDdugbMOBgAeWFaDpvhQAtBpC3wt0IXgA2hWBhawbQANLwj5IQB6BCKemAAIIGu6DswLQAu"
    "O6CbWgDQwbCABYcBDCNd0ZCPJAPkgBlr62hwTcQrkQQTbdADLWhEaw4k80DaN9RACCb8bwrEQ4Qb9EQGPjB1hx+KBbyRQDdCEIwt"
    "YARFzCIBFoUIm0ALcIWkK43QrgQADrC2QN/GCSIF9IAISOEK0Da54W3EAdm0NI6YMK3C8UAG+EYIA4GDAAtpeEBrBMLjADVQr6QT"
    "rBAuIACdTDlE7ockJENWrWIBhERvMRkOYHrgUbDuVRv1ERJdobojuYh+saum17Saz8YjUPMiPYqr+T/pXatw+lE6ftBF3RR+QemI"
    "arp/Qxf0gi8LxcV9xkjsEmwhIHi3jHNTDUsw488tLbbaSpa1GwSBqSY83YxzPqtQxRMztIqU3KSibNMJacKQUJ++I3XOpiUMPKs9"
    "DMpqJ6VJ1vBVFB0DPioUunhipSTlUmNsq7ocf2SQdybBPCOn/wDkVcaYfHwn/jHR4KqnsOoi6eA88Y1FYuQnhoYrjDGdMrW26k9P"
    "SAkGZCWMwVcrt7fjABI0GpJin5zMDEDtWmagxU5uXU+6pzYadUEpudBbdoLRYYOcm09LEdRI9TArIFxDlEhQsNIpCm5/PyUgxLzN"
    "IVMutICVvrmfGcPFR8WNpH8ogKH3Pj4SfVjLwNXgvURc43GGgEb4p9n+UFtuhHeCw5+6P+MW6y6HWW3LW20JVbmuAfnjlVoyp6SN"
    "KV9UZT0QBoIwzrymJR91HtkNqUL7rgExQAztxetYQ2qSUToAJYEnq1jVGhKrsRysehRvhGKAVmzmAn+iDySB7IxS+duLFPpbdXJ+"
    "2AI7nA4x18FO100Zz+h6E4QkmGIXtISSd4B+KI3mPU5ml4Lqc3Jvrl5hpCSh1s2Uk7YHzx5YRzSsbloiTEwiRHmKm5n4pkqi1MGp"
    "TU6GjfkHlqUhfQQIkyc98SqHuRJD9Rztj1ywM4vRowqiL34GG313RS1MzpxFOVCXl3aZJobdcSkkNrvYkA8Yui+pEeapTdN2Z0Tu"
    "ZAQSY4OODbCNYO+0m7p5I7ab7XttLboj+PlLGEavsgW7kcv5olJ/WjM9ikMtpZ2YnJsJSdnk9TbdFjzbGxIJOh03WiAZYOrTOTgQ"
    "FbBRchPEiLNrLTTFNS46FWV4unSImNd67OkF9KIY42pxZHibPQmMD7aUJ2EuISs202dY6aCwbDYuOkxozkw0hxQalA8ecH5485WY"
    "ZeQQsm6Qsc6hGZ2XbbTYbN079N8JmcdSNJYC/AcIxvOOrUT3OrxuqKkZMbkq2q6rg8wA3RiUhCE6DUDdGUmY/qVWPNaHCU2lXVvO"
    "m+KDTDZVfxdVQ9Mq7bQJTYc8bnIAHUkwgy3tbQFja1+iBk1DLO7HjL0MB6WAbsp65t7Uxu8iIwuyqXFAmxtuMVBnLS3tDa29/MN0"
    "ZQQld0ukKtoAI3hJpQmw0ENEmja2ikXEBc1EhakAl0lV77ocTrYvKBVuAEbYl0pSRuHNGMyiVag2MCGs2knZu45Yc0ZklCNCte0r"
    "cDaMiWNjQgac0OQwlSrqSCRqLwKjEkFwA+MSL2uYctABspdtrcDGyGwNwhFlCyLgG24mJYXNZDatCFbuiMRSm5QXALnm1vHck6FN"
    "zjC32GeUbSQFWOtz0RkOGZxM0ljkELeI2glKgSOvmhoXU4LTCiSra3EjRIhLZ3pLx33OgiQu4anpdsulpKkA2Km1hQB6bboyKwlU"
    "UFQLCFKSLlCVpKrc9r3gCMIllrJVyiubcIaqVCbBTtraxJJfD89MsOPsMbTbd9o3A69ITGGp2cYS+hgOIUrZGo39MS4I33LtEqLi"
    "tTwhdztosOVVcaWjsTdPVKPFlfJlQ37CrjziNtrC06+2h0oZSHBdAcWlJV1Xi7bkTI4iS2ivxjv3QFyjST4zmo0sbRIO8k6ha09z"
    "OkpNlbKb/JCdoE0Zdp/kQtLpITs6kkdG+BSNopyVrWCpW+NhUs21a7hBGmto6po86h3ZMo8FEXtsG5gqpzqnQwtohzdsqGt4uhDk"
    "JlQbgqVYHohhZZYUbuKBVqbnfEoXhecbuCZYOgXLXKjb80aa6JOcnyqpNzYG9RQbRQcQSySbBarJtaEZdtBJU4oE79I7veGeCdvu"
    "N4I/G2DaMj2GakyoJVKrUSkK8TxrCFwR4S20u20qw3boa8y22TtLIB33iQIoM+UlaZN3Z4nZ3QJ6iOySGVvBsh5O0m2unTEvcWZw"
    "Ey6iokLNgNIS20Njxlm2+JYzhGZdbaWHZdJeG0hJJufijXOG55W0Eyyl7OhItbSImtxZnB2G16NpWAALkkEk88MWlDdkqUseWJGr"
    "DNQSy26lm4c3BJF/KIczhidfWptTQbUE7Q2/vhFuWxGkNbSlW27HphrjSUEA7fRrEiVQJ9L3JGVXtEXAsPlgHD9RLmx3IvaAvY23"
    "dcLoliNhpSirfbrjC+UtqCSVA2juTkk9JuFDzfJq5jGmU7W9I64pDliXUskbSiOGsZ0tNtGyiskjhG8EdEODehAsYCxqNNJX+MQN"
    "2phrqWUKsUrJ6L6x0Eo6IRQLaxSGihgG5188YnENpVslKz59I6OwATDSgdMQtzQTLpuq+0fLGNYbbUE7Cz0jhHSLQhhbFt8UGm2y"
    "khV0nU85hpSkObIbUSOMbwRpzwig8YC5ppZSSbg790McUlK7BtajuuI3tjhA5PeIgNNDSSk3TaMdkhezyS7E2vG/yfkgBABveBbm"
    "BtgAaptzXhpACiOSNr743ANN14apOsCGoGgRcp4+WGWG0Uckesbo3Nm5PCG7OkUXNVDehOzqd8YyLL2eQNhxjcUkC5huxpzcYC5g"
    "S0kJBtrDCslVuR8sbQSLXtAKQeaBbmFlhTlkIRtLJsBbWN3wfqZTYU+YP7Mxt4eZC6xKJ/xAY6eIZqbcxW80iem0MMhP0Blwpvpe"
    "5tw57XjMU237GkcFGHKoUW73TO7+rMMdw9U0NrWqQfCQL3KIfW8Q1epO8mK7KSrKBZKGnlJPlO/zx1MGzEwJKqCYqPdqQ0NlQcKg"
    "NDffxiy+m1xYiyRyTounja0S6n01RlEuNEJXYmyt0Rh3V/dclWkTCSfWGUDYWoBva2rXHVGJq00jUXoRjDrjScXsoUhPKF8jbCtN"
    "260W+vUCKRwu6l/HMqpCVAcsTZRuRvi7SdI7VY2qf0Cd0ar5uk62irceO2xFLpTdSks7gNQL88WlMaJPVFZYzSTiHbUpKEplxckg"
    "E67oq8rKtzFiEFvNtYOl51lQ47ymL4++PHUxRmMQ03mu2WjqqYlyvXcrxb/NF5m+0bc5jtL7MDitw6XHWI8zYlSBiCpD/wAl3+Ix"
    "6Y3W46iPNGLTbEdT0sO6ndP1jGsH90zPY9E0Q7VHkSDvlmv4RG/Hn+Ux/jJiXaaYnHuSQgIQBLJIAA0+95ozDMjGYVZU6sf6RJ/+"
    "MdJ4ZuTd0FIvvSDwihFZnYyQftvTnMokfNB9lHGKR9ttnrlE9kY8LLlDOX0NCDbjHl6sG1VmxcaPOfxGJOnNXGA17pY8sqmIe667"
    "MTC3nTdayVHhqY9GGoOE7tkk7o9RU1QVT5VY4soOv5ojaMaVKVemSh/wG/4RG5w0jw1fMzoiI5ppCsFT1xchTZHXtCKRw3iJ7C9Y"
    "RUpdhp5aEqQEu32fGFuEXlmcL4JqXRyZ/fEUbhqtS9Arbc/MyDc82gKSWV2sq4tfUHdHqwO0kc5kyOeFU3GkU8/rL7YgdXqS61VZ"
    "moONobXMuFwoTuTfgOiLBObFDVocGyt/2fqRAq3UmqtWJmdYlUyjTy9tLKbWbHMLaR6aULS8ljL23LiqzaTlANlAF5Fkmw36jfFT"
    "YXxB4NV5iqdz90chtfQ9rZ2rgjf5YtqbO1k8CN/e9B8yhFYYBkZapYxp8rOsImGXFqCm1puFeKbXjnh/PULLZE39nVAAPeBdx/5P"
    "/GKwnJ3u+efmtgJ5ZxTmzzXJNvjj0QMB4XNr0CSOo/mz2xQFfl2pWuT7LKEobamHEJQPvQFEARMPOm52jGzDvbc9L0slVNlSCNWW"
    "z+6I2hpfn6Y0aIrbo8koaXl2ju/IEbv30eCr52dVsQrN0Xwgs80y188beWBvgaQFxYF0fvmNfNoXwY+eZ9k/GYwZXPrdwfLMIVs7"
    "LroJt+VHoofal/JJbomZXY2BBMbkognXZN4ZLyrbCLixVzqFzGcvOEbPKlKeZIAjH8FsZCLCABbfDU2AHjEmHWuN5EQogReHAgmG"
    "Ec1zDknTWACYWza1hCvzwrxAIaQQLm0A74IilEdIQgmFfpiEsI80Ai8GFe3NAA1gkXheWETFIIDoh3k0hoJ6IcDrEKheSFpzQiRC"
    "80AKBC8sKACRB3g6QOeDbngBWsIW/hBsLb4IsIhQbMEQrwvLABteBaFv5oMAAiAOmHHWFbWAEBzwYEK/kgAwoGsI6jWAHQIF7QeM"
    "ALWFbjCJ6RAvbjABtfmg/LDVdcLW3toAcd27zQLdMAXJ3wrHngBG0AEiERzQki++AET0QDY88OI6Ybs21HxQAioAwtq3GERfhCAM"
    "AOJ54b80I6QCbQAYMACCN0CCtAPXB8kM2tdYFDs9cKxPGDCgAWtrCI5yYdC0gBoEIJuYNxzQLwAbEc8BUK9jqIV78IAaNDuh0Aq1"
    "1+SFfogB0Led0DagbQgB14N4ZfS9zBveADc2hpudYcmx3wYjYMYTxsY42OWuUwZW0nQGScHxR3du2kcHHhUvBddSN5kXbeaLHckt"
    "iosgjfEFU/yY/wDcEXgYo7II/XFUxb+hj/3BF4kxrFfcZmOxEMycM1vFVJEjSagzLNm5dacBHL8w2huHRaK/wHkrNCrLmMTSyBLM"
    "GyZfbCg+rhcj735YsXMuuVHDuFXahSiEzKHm0AlG34qiQdIppecuOmySXWkgcTJp7I7YdVJQyxkkjMrJ3sbmYzmXz1KcYw8yiUqk"
    "u+AUoYWnlEgkKSSdNN46o6WAMnZ1uclalW2abM091vbMutRUpSVJukiw0IuOMVBOTjky+5MOG63VKWu3OTc/LE7ks3cbMS7Eu282"
    "GmkJbSTKC+yAANba6Wj2VKM4wUYS/m5iMk3exOMcZPS8xaao0zTKSwlISphd20rPOVkkdVxGlgPLjD8hy0xiidpE05qhuX7qQpCR"
    "+MSDqeYRjzKq09V8qMP1CfIE1MzRLtkbINgsDTyRW+EcIT+NKkqRkHJdpxDReJeJA2QQOAOuscaUHKl9U7Irdnoi1cwMA4XnqQhe"
    "Gl0iVn23AShE0hIdQdCDdXDf545eCMs6dT55MziGq4fmpcpIVK90pWb8CFAgAiNA5B4hI+3qX79Xqwx7IjEUsw473bTF7CSqwcUL"
    "2F+KYJwy5FUf4Gq1sWlLYQy9edShmSozjh0CUPAqJ99EyQhLaEoTolICQOYDSPH9KcUioyxSoghxJuNLaiPYAttbXHnjy4qj02le"
    "5uErq5gql+981+hc/hMeSadPqptTlpwJDhYcQ6Ek6GxB3x6zq0wZSmTcwlKXC0y44EKGirJJseuPPq80ZYE/Wbh0kgG/ImO2DzWa"
    "SuYnZMky/wCUQtJ2vB8XJvbuo+rFTJmTM1FT5GzyjpXbmuq/zxMVZpS1rjBmGx1smHN5ot7YKcH4bSRrcMHSPSoOCeWCV/cl7u5J"
    "c1sf4hw5iNqQpc+ZZjuRpzZShJupQN946IiM/iPMDE+HZtx5c5N0ggiYcDKdgBJBNyAN2kNzanl1SvyE6tIQuYpcq6Up3AkE2Eak"
    "jUcas4QmZWRZmjQFhwPLSwCgA227qtccLxqlFKnFq1/cy3du5xMPTFTlqtLO0guCocoAxyYBUVnQAA6XiwROZwK9sxVB/p2+yK8o"
    "S6gzVZZylpWZ5DqSwEAFW3fSw4mLHarebij48tUk/wCkR2RcQ7O+n9SxXoaNTxJmfRJfumorqEtL3CeUdYbCbncL2iTZV47xFiGv"
    "dw1GbE1L8itaiptIUmw0sQBxtHWxVL0WuUSjyWLK+5TJ0MJfW0bJKlkWJULdYjpZc4Mo+H236hS6j3ybmQEJeKQAkA6gW6beaPDV"
    "qxdLWOr9jrCP1E1bIN+eOBj+4wbWTuHcixHfZCXLqSVb7RH8xARgus67pVceOl50ansUzlSpaK9NBIJT3IskcN41i1cRILlJCU2v"
    "p5NIrDKRCVVioFRsEySzr1piz6vc05saFNgT5o3jPvMsPKiFPIJTslQ6ekxiSwm+up4RsPArJ2dNeaAlAt09UeWxdwIRYW4dUPOz"
    "bUDrjG8yp5lSEL2FEWB10jXEk6hV+VtqDYa20IjQM6gbeKqwhbN7Rrrl5ogfRQDpuvrYa/NGw0gpbAWSo7ibcYjQFYfHAVoL2h1u"
    "g26oaUKO4kDqi2AwgmBaxjOJdwj2i7fmmD3M5b7Gv3pi2ZlyRrG2vPDbaanTojO9IPOIKQhxJ/MOsYV0qbsOTQpu51CEG1u2FiXQ"
    "hrC2bfPCTSZ65KlOlB4bCueNhqQmEpsW3Vam10HQc0LMXRg5O+sBIsbWjc7if/qHfRnsgGSfSCpTDoHOWz2QsxdGsk3hw1tGcSb/"
    "APZ3veHsgmRmN/c71v0Z7IzmRbnRpeI3KPKPNMoAUtQUHL+1twtxjMjFKW58zbcmhKlpKXUhXirvvI5jHBepUy6QrkXtL6FokG8Y"
    "hRZwI2SmZGlhZtQtEzRJckScQsS8u6zJyimw8QVla9o25hpGdWK2jN92okdmZ2dkKLlwNLbrRFRRZ/xftoAcNhUZWqTONna5OZVv"
    "BuhWsM8S3RJW8ZOMBpDMo0Etm5CtSonefLDBi51hstybCWUlwrsTcEHek9EcUyE0P6K+f2ZhGRmgLmWeGu8oMM6JcdPTbU3Mqfal"
    "wxtalCTcA9EdHwhYmGWETsil5bKdlKg4U3HSI0U0aoEX7imD+oYXeOoKP2jMejMZdaF7tm8r2Oozi1LbbaO5EJDKtpGw4UgdB54c"
    "nGTiCgmXZulSidSL7XDojirwvPuOBzuSYHQEb4xJwjU9xYmzrxRDr016oZWdwYvcRsJZlkhCNqwUsqNzxvHITOuomUzAUdtKtq51"
    "1hqcJVIKF5ea5tE2jIxhapsq2ixNudChx54PEQatdDKzoTGIGn3FTDlOYMwresqNr8+zDlYsdN19zth0t8mVhRtb83dGqaBUv7C/"
    "72MD1EqLQK1yT4A3nZgq0PRoZWbisVTCnFq5JuymeRIud3PBcxVMONqTyLSVLbDZUCb2HHrjnd6p0gfSrvvYIpM7u7ldHki9aPJt"
    "Up8HRZxW+y23tMNKcaTsJcJIIHSN0aE9XzU+SQotbTSdkBJ3wzvRPm47kdseYRrKwxMkaS74Ouotxgq0F6joz4Oi9iScMsyylwtI"
    "bTs/Q1EX64zymKpqVZQhLTJKfviDc9euscTwWnNjZLL567dsbSaJPJAHc69Bxt2xOrDa5ejPg6TOKpppKUBtkhJNrg634b4aMROo"
    "KuSl2G9pJSQkHjGiKLPnUS6vOIXeqeTvl1i5tvHbDqxfqToz4OgjE841sbKWrITs2tvHTGJ7Fky24HVBlItaxvYfHGIUGpEfaqvf"
    "DtjE/heoTSNlcqbb/bjtjomOlPg1alWFVR/bdLQWBuTpYRzHZdta78ooHmCo6hwTPnQMKHTyggeA9RKgNi6Bw2xfzxbjoyOZ3K3x"
    "fdt0uQA20lduXc03/RI7DeCpxJIDAPW4IKsCzq97Vhe9uUECdGfBy1SyFXIW4CddFGAmTSk7W24T+cYkDeEqkhNg23YafZBDhhOp"
    "LcSjZburd9EES5HSmvQj3cjavvl+RREMMg2Ve2d6tsxL04EqttUsek/2g+AlUKrWl/Sf7QzGMrIaQzLE3cUOgkwwiUWv2wv1xMHc"
    "uZ91QUoSxIFvsh7IaMtJ0C30tYafZD2RbjKyLoeZbGyFpFuEZAQoXSRY80SU5aTqgATLekV2Rlby6qCE7CXZUAbvGPZC6GUilrm4"
    "1gWudNIl4y7qA15eW6rnsjnzuEZ2SeDZdYVcXuCeyCuMpH1Q32o6I7Rw3NbuVZ+OGHDczaxcY+OJmRcjOEqbbF7qOm8Whd0o0JCg"
    "DuuN8dkYRdBuFs3txvA8EHOK2uj22kMyGVnFM21fRe7fD0uIc1QQoDmjseCJ3bbHmOsZE4VdQmyHmUj806RcyGRnDI0haeWO4rDL"
    "19Jhu35phjGHHHhtcuhJuR7Q6/HC5nKziFMBSg2gqUNBrEgOFnOMyn3h7YPgopQ1mknh9j/3gDi0Wts0+oNTLjayEKBsLbo6s5Xs"
    "L1Cqd81SlQTNgjxmn9nUQ4YQQBbl0jqb/wB4Xga1axmEj9mO2C0uy3NarVfCtcf2nqO9yw3rQsIKuu2+BKVOj0uRmJenU95rl02J"
    "W7tRtHBqd/dR6LNiAcIgD7aWf1BCTuW5GWwpcyjZG9XmiXuCZcpjbcqko2kWU5b2ojTGGEsuJWJlStk3sUgXjsy7ZYkToAoNKO/o"
    "MY1c0zUdis8v9pzGkuVH+cUfiMXsfa3ijMuUlWMmVG3tlHqi8zuj14pWqv8AgkPKasyfFMUzmbNBrEexyIcJYTqeG/dFyzHtVW+W"
    "KRzPS2MTlS17NmUgW0+SMJ2jJ+xfU7ON17OaZO/6aaP8MX4o3Wo9JihMw0BnMTlkgkF1pRvzi17eaL6Cr66a66R13oROfqE899Y8"
    "3YzFsU1Yf+W58sekVcBHnHHSQ3i6rp5ppZjWD+6jMti+sLEHDdK3/ajX8Ijq7KYrnDeaWGqfQqfKTM0+l9iXQ2sBhRFwLaGOmM2s"
    "J30nX/g6oVaM87aQTRMy2nikeaByaSPaJPkERL2WMJ/3i58HX2Q4ZrYSI901eVhfZGOjU4NXRKgy3cHkm9CPvRHmWvoDdbn0AWs+"
    "7p+sYvEZqYRJ1qtutlfZFGV2banazOzDCtpp19xaFWtcFRIPRHowlOcal5IzJo9JUNW1RpA88s1/AI3ybC8c3DitqgU1XPKtfwCO"
    "kRpujy1vOzSKvzMx5Kdy1PDXckx3QClHK3Tsbwrr3RWGHp2myNYamatJGdk07W2xp41xp5oufG1LwZTkv1WtyiXZp7UIDigt5QFr"
    "AA6bt8VRh9yRm67ySqA3PNzCyG5ZC1hSOhJv8se3CNZXZGJ7kkOLcuyLHCDovzAetEHr01ITdWmJimSplJJartMnehNt2/ri465l"
    "7h+n0lyclcNqmphCdosJmVi2mvHW3RFdYcew0/WFMV2nJblniAhTbi0hg9OtyPkjVGcHJuKd17kaZ3Dj+nTOXb1CQ2+iaZlEMlS7"
    "bKztD2ut4g1EptRrFUalaUlaptdyjZXsEWFz43DSJBmJIYepVVRIUJrZLSBy6uUKwVHUAEngPliSZJ0MqmpysuJ8RtPc7R51HVR8"
    "gt546RcYxlVStcju9DjeAOYI3Cb8k7/yiHzsrMyU6/LzYImG1KQ5c3O0N+vHWPUoWh1N21hY50m8eb8YrbcxVVFNqSpPdTmoOh1j"
    "hh68pTyyRpxVj0Lh5QNDpx33lmv4BG+TrHMwudrDtM/yrRv+qI6dhHjrednSOxEc1Bt4Mm+hbZ/ejSybHKYbWN5RMrHxAx0czRbB"
    "VQOgtyZ/fEcvJElVAnBzTX/xEd8N5JkluiyAjS2+Fs2hwEMN777RzNDwBppDwi43RiTtA6mMiVeSAH7FhzwtjogbR54NyeJMQo1S"
    "YISOaCSTAgBWMK9tIUKACqALhNibwvLCgQKTCvrr8cDrg7zAguPGFx4wr2G+DeKUFzCueeFaFYwAjrxMI7t8ICCTaAAB0wbX42gX"
    "6IOnNEAjeCk9MDjACreSAHwLwAocxh2+ADC8ohXg8LQAoMAdcKIUV7QrwoUCBvChQrQKG5heaFeB0QAbmFc9EKFv4wAr2hpVroB5"
    "4daBugBXPRBueaATC1IgBDSDe/AQ0bzrrBN77xACJtC374SgTuIhWNtbQAgbcIBGukEeSASb6iADa46YIHPaFfTUGESOYwA0gE3E"
    "EjpgbYvaCVAwABrv08sIAmCLQbaQAxSb9cAJtzwVb+MOFrb4AGzbWFraDeFAAvvgb4O/hCIgAXhbRvCCYJF+iAATDbmHEQCCYAbc"
    "HnhXIO+ERaEIWA69xrC6YCVWh+0BEA0i/AwbK6YdtQr3hYAFxvEDpVe3RDtYaqwNiRaFgMKgLkgmORjHaVhGtADfIvb/AM0x2C6B"
    "uF45GK1LdwxWEgaGSeH7hjUVqR7FPZBfdBVFf+EPJ9EEXgdYo/IK3f8AqZ49xJ/jEXefPFxX3GSOwlWULKAI6YqfOzHMvTaavDcg"
    "pBm5pP0ypP8ANNfi9avkiR5m44msJ00tyEm87OPp8V7kyWmRzk7ieiKUwdgqp5h4gKn3XeQLgXNzatSATz8VHgI7YalBfu1HojnN"
    "t6IhTh3ERZuWebLlA5KlVm79N9qh0i65ftT0cOEV/VJJEnUpqWRtbDTzjab6mwUQPLpFtYpyc75UqmVXDbSUPPSzHdEtewUSgXcT"
    "za7x5Y+hiKlJpRnszEE/Q6ue03LzuDKXMSrrbrDs6FIcbN0qHJq1Bin8L4dq+JqgZSjfbIbLluV2PFG/XyiJljfA9SwZgWUanasu"
    "ZS7OhQlUi7bKihVyCdb8/CIdhWnYhn6gpOHO6jNobKldzObCwjjrcabo54e0aTyP8iXm1Jb7EWPrbh8NHbDHMpMeMtLcWBspSVH6"
    "cG4C/PB8H81bhWzXr8/dR9aGvUPNVDS+VNeLYSSq8ySLW1++iKq7/cX+f1Dj7EJp32+wQb+Ok/GI9hpsQOoR49poAnmOIK0+a4j2"
    "FbhHm/UH9SOlPYTiQpKkqAUCLEHW8eZc1KvKVLF8y3ItsNy0oBLo5JASFEe2Vpv108kelp1lUxLOtIdUypaCkOI3oJG8dIio15Ao"
    "VUmnVVgvSZXd4Kb2XSnjYi4vGMJVhTblMk4tvQ5OXeNMGYZoBYqbDr0466VukygcCRayQCfP5YlktmfgScmm2WpRQLiglJMgm1yd"
    "LxxscZUYWw3hefqbbs8l1lv6EFPAhSyQEgi2u+K5y+piqri6lyg1C5hBP5oO0fiEd3TpVIOorkTalYkOdyEt43DSAkBuUZSABYDf"
    "pExwfJzU7k5OSMmwp159uaCEjepVxYDzRXmaFXRWcfVN9lQU22tLCDw8QWPx3icTlZqGB8rKG/IPJYm5hfjXSFXSvaVuPkiVIvJT"
    "gtxG1mys5Gm4gotSam5amzrczLuBxBVLKIChqLi2sTFGZGY+u1Kr/wD+eeyJNlLjeu4pq05L1WdL7bcvyiBsBOyraA4DpiMYszLx"
    "TIYoqcnK1ZxlhmZW22hKU2SkHpEazSnNxaV0LJJHDqUpi/GlZRMzlNnXpgpS2CmXKEhI8lhHoDB1HVQcOyVOWLOMt+OeBWdT8Z+K"
    "NFnFcvS8JU2sViZWG3mGi44ElSitQ5hG1h/GdBxI8WaXPB91KCso2FJITz6iPHWnUqpNqyR0jaOhIG+MRvMf7iq0OeVUPkiRoNrx"
    "GMy3UtYIq6lm20xsDTiSAI5UfOiz2KlyjSe/NRUDYdxqHk2kxbK2wphg6gCx16oqvJ1O3WqhY/0M7/zhFtlv6WRfcBvjWL++zUPK"
    "aDoAUNLwvFO9MJZuvmtDhuvHdRM3ClIA0EMUlN7gQ8Wt80NUQN8bUSMxlF+EIIsNLCH3A4QFKEXKiXGbOsYJlI5Po2k384jP8UYp"
    "hILY/OT8oi20My2OuEE63hwTYQk9Bh2loJHzG9RhGv8AvDraf7wDswQoWipC4lJuIZYg/wC8ZLgw1eyItiXG6D/+xrzesq5bmh71"
    "iLAnfGKbUEyqk34RiSNJj02tvgnrhiXBbfC2xzx+bnuz2XHEX6oRHnhpWOeBtjdeM2IOIENtY6XgbYI0VDSvXfECMhBPOIxzAAaJ"
    "PRr5YPKab7wyYX9BJ6R8sSWxuO510zDdgEkqvzJMZgdoX114GGoJAAhXsY+O9z6aMgtDgBa8Yr9cEK8sEB5HMIWzYQEqhLWIoGKN"
    "uIjRqSh3G6fyY2zzxp1L7Tc14RuG5pbmgFCw04QbgWhg3C8Ekb9I9Z9NIJN+EN0N7wvLAuIagcNngDDCQYQUL6wSReF2LANxuGkM"
    "eOjen3wh+0BxEYphfjNgke3EdKfmRGtDokqABtD0m+tjAC0gAXEDbTwUPPH247HCxk4wQL63jGFtk6qT54dyrY120+eNpMjQjv3Q"
    "8DjGMvtj79Hnhd0NcXUecQsLMyjU7oc0B3W1pz3jAJlm/wBlbH6wjJKuIdnGwhxKiAdxBiS2OVRaHZNgIbuggG0LQ74weIWm8Qre"
    "WFYcINwAIoAoWGkAX6IeSIYSBugBE6RHK2fpwafeiJCVHfEUxHPy0rPbMw+01dAttqteKr2dgYFDUwy+msaJrtMB+35e/Nyghort"
    "L/t8v78RzyvgtzoDoG+F18I0O/8ASwdZ6W0/LgHEVK/t8uP14ZWS50COGsI7rDWOYcTUkf0+X99DVYnpH9vYHWqGR8C6OmrcY15H"
    "2hO8bRjRViej8Z9nzmMMrimjtN2VPMg7RNrxVGXBGzuq37rwNDoI5BxdRBr3wZ8l+yMSsX0Um5nmx5D2RuzMHaO/hBB0jhHGVFB+"
    "3Um35JhvhpRBvnL/AKh7IlmQkCvFFr9UMCjffeOCvHFDAAE2feGMXh1RQfthXkQYWYO87v1MZVtJEupdlG7RuQBpod8Ro4xpc59C"
    "ZeWXF6JugiJApT4lQG0goLJCtfyYi8yRtLQrbLVsKxalRUAU7RAPGLtV0CKSyzG1i5O6wC4u1XTHoxX3n/BIeU05nRJ0iosZ0/u3"
    "FD61ILiQ0m+o0EW7N6JVaKjxdN7OKJlohNglACrG40jm/I7GludfNd9h3Fcqy2lSXZdILqj98VKChbqEXQ2boT1D5IpHNmUelsWd"
    "1KSQ0802ptVtFbIsfji65Re3LtK18ZCTu5wI72XQjY5LczEXEVXijKmr1yvztQl35JDL7pWkLUoK1tvAEWrfzwgdI5Qk4PMg1cq+"
    "SyPkTKtGcqUymZKfogaSkoCvybi5EZlZG0vS1WnR+zRFlKVsi9iQIVxYCOniKj9SZUVmcjaadRV5sfskwDkZIcKzNX/Qp7Ys0Kv0"
    "WMOuBbW/GJ16nIyoq85FyXCszAP6BPbHAmcm681NOJlhLusBZ2FLeCVKTwJHDqi7xaxsYVx0RqOJqp7jKjSoUm7I0aSlX9kOssIb"
    "VY3AIFo3yARaAVDnEG4PNHGUnJ3ZorbG2V8zXas3PU6Ytyxs8H3CQ30jjboiS4QwLTcJM3ZHLTahZyYWPGPQBwESMHfrC2hcRpVZ"
    "KOS+gsgkdEQquZUUet1VyfLrstygutpkAJK/xv8AaJpeCbRhScXdMWKcqWSM+0vap0+xMIvudGwofKDFpYfobFApUtTpcXQymylc"
    "VqPtlHrjogiCDaOkq05RyyZLFIVHLXFwnJkybSgwtxRRszQTdJJtpeBS8nMQTMwnu8y8oyT46uUC1W6AOMXedRB0tG4YmpFWRHFN"
    "mKRlW5GTZlWvsbKEtpvvsBaM45oA5oW8xwbu7s0iM5kgHBdS3k7KT+8I5OSCSmhzqjexmbfux18xQTguqgb+SB8yhHHyWWo4fmkk"
    "6JmCR5QI9OH8kyS3RZRsBvvGO410tCvbSCOa8YNMPDSCkaXvCsLDSDYc0LgcDB0590NvYWhDntEKOKhzweqMalCCDffFJcfpeBpD"
    "RoOEIb7RAOsOeFYc94B3amFb/wCiFwO0AvAHlhCEDppABv1wtrjrDTCNvLAg7b6IW0YZu1g3JigdfphXgGFAo64hXsIAB6oXliMB"
    "ggwwnTfCC9YAyaDhCEM2tIN7CAHi0LQwwL5xCCt0AZBChm0eAEK5PCIUf1QrX1hu1C2jAD7dMK/NeGXvC1gB3XCEC9oIOmsAG3TB"
    "3CG7XRCJ6IAcYVhAChzwCrWAHWhHdDNuFtdEAOTxhHfDQrogE8bQBk3iETbfDLwb6cYAcLc8BRF4beASIAyXBG+FcHjGPavCBtAD"
    "79UCwMMvrBO7dABGyBYWgixgAC24Q0b+iBDIEg31PnhKFuMMJ2eNoJJtzxQG8Ak88BJvxgXtAXHXMLWG35hB3RCh14wdYCeYHzwi"
    "rZ0gQW11wCoW3QCoAXhqlX3QKIqAhBVtwMC0OA1vACTa+ukO2wDxhFVhYQBbjvgA7Q4mDtDiYaTYwtrTdAD9ra0EMUnhCCtYybFx"
    "rAGEAXNzHPxGnaoFTSOMo8P3FR1S0N+mscrEey3Qqj0Sj38Co1HdEexS2QKtqvVG39iSf30xeMUZ/J9P1dqH+QTp+umLzvDFfcZm"
    "OwloStJSoBSTvBFwfJGKSkpaSumVlmWEqc21BpASCefTjGXa3iCk2Hxx5zVjyLX0k16ojnmXv41R6pw6L4epf+TZ/gEa7mDsOOOq"
    "dcolPUtRKiosgkk8Y6zTSGGkNNJCEIASlKRokDcBHetX6iStsZjGxWP8oDXC9PH/AJo/9tUUxhqdxDT59buHVTaZktkK7mRtKKLi"
    "9xY6bo9Xz9LkqoylqflGJptJ2gl5AUAefWMMhQqZS3VOyNOlJVwp2SploJJHNccI6UsV04ZLXMund3uefPCvMzcX6zpw7nN/4YY7"
    "iTMstOcs/WuSUkhV2Da1tfveaPShJPG/lgLSFJKTqkixB4xrxr7UOmjx/TmyJ6WBGm2n5RHsC1jHJRhHD6CCmiU4EbrS6dPijrC1"
    "45Yiv1WnaxqKsrGpWph6WpM6/L35ZuXcWiwudoJJGnHWKKls8sUyVkTLElMm2vKNFJ/dIj0ARpwjmVHC1Dq/jT9IkplR++W0Nrzj"
    "WJRqxhpKNyON/U864zzErONWWmJpDcvLtq2gyzexVzkneY6OGWX8C0B7EkyyUzk8hUtTkKGqb+2d6gNBzxdcpl/heSeDrFDkkqBu"
    "CpO1bzkx2pmRlppgy78uy6yRYtrQFJt1R3ljE0oxjZGen7nkintOTdR2lsvvDbK3eTTtq2b3UbcY72MMVzuMp2XZblyxJyw5KWlk"
    "67I0HlOkeg6Lgqh4dnHpumSKZd14bKiCTYcwvujK1hOhsVJdTapksibXqXQjjzgbgemNvHLNmUf4HT0sRLKPBM1hmQenJ9PJzU2E"
    "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"
)
