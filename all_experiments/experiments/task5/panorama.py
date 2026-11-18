import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

BASE = Path(__file__).resolve().parent


def load(path):
    with Image.open(path) as photo:
        photo = ImageOps.exif_transpose(photo).convert("RGB")
        photo.thumbnail((960, 640), Image.Resampling.LANCZOS)
        return np.asarray(photo)


def demo():
    return load(BASE / "1.png"), load(BASE / "2.png")


def stitch(left, right, homography, right_mask):
    def corners(image):
        h, w = image.shape[:2]
        return np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)

    right_corners = cv2.perspectiveTransform(corners(right), homography)
    all_corners = np.concatenate([corners(left), right_corners]).reshape(-1,2)
    if not np.isfinite(all_corners).all() or np.abs(all_corners).max() > 5000:
        raise ValueError("投影范围过大 / Слишком большая проекция")
    x, y = np.floor(all_corners.min(axis=0)).astype(int)
    w, h = np.ceil(all_corners.max(axis=0)).astype(int) - (x,y)
    if w*h > 8_000_000:
        raise ValueError("拼接尺寸过大 / Слишком большая панорама")
    translation = np.float64([[1,0,-x],[0,1,-y],[0,0,1]])
    canvas = np.zeros((h,w,3), np.uint8)
    lh, lw = left.shape[:2]
    canvas[-y:-y+lh,-x:-x+lw] = left
    valid_left = np.zeros((h,w), np.uint8)
    valid_left[-y:-y+lh,-x:-x+lw] = 1
    warp = translation @ homography
    warped = cv2.warpPerspective(right, warp, (w,h))
    valid_right = cv2.warpPerspective(right_mask,
                                      warp, (w,h), flags=cv2.INTER_NEAREST)
    def weights(mask):
        return cv2.distanceTransform(np.pad(mask,1), cv2.DIST_L2, 3)[1:-1,1:-1]
    a, b = weights(valid_left), weights(valid_right)
    out = (canvas*a[...,None]+warped*b[...,None])/np.maximum(a+b,1e-6)[...,None]
    return np.uint8(np.clip(np.rint(out),0,255))


class Pair:
    def __init__(self, left, right, right_mask=None):
        self.left, self.right = left, right
        self.right_mask = np.ones(right.shape[:2], np.uint8) if right_mask is None else right_mask
        sift = cv2.SIFT_create(nfeatures=1800)
        self.ka, a = sift.detectAndCompute(cv2.cvtColor(left,cv2.COLOR_RGB2GRAY),None)
        self.kb, b = sift.detectAndCompute(cv2.cvtColor(right,cv2.COLOR_RGB2GRAY),None)
        self.neighbors = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a,b,k=2) if (
            a is not None and b is not None and len(b)>=2) else []

    def run(self, ratio, tolerance):
        matches = [items[0] for items in self.neighbors if
                   len(items)==2 and items[0].distance < ratio*items[1].distance]
        flags = np.zeros(len(matches), dtype=bool)
        panorama, error = None, None
        if len(matches) >= 4:
            target = np.float32([self.ka[m.queryIdx].pt for m in matches])
            source = np.float32([self.kb[m.trainIdx].pt for m in matches])
            matrix, mask = cv2.findHomography(source,target,cv2.RANSAC,tolerance)
            if matrix is not None and mask is not None:
                flags = mask.ravel().astype(bool)
                if flags.sum() >= 4:
                    projected = cv2.perspectiveTransform(source[flags,None,:], matrix)[:,0]
                    error = np.sqrt(np.mean(np.sum((projected-target[flags])**2,axis=1)))
                    try:
                        panorama = stitch(self.left,self.right,matrix,self.right_mask)
                    except ValueError:
                        pass
        view = cv2.drawMatches(self.left,self.ka,self.right,self.kb,[],None)
        # Limit the number of drawn lines, while computing scores over all matches.
        for accepted, color in [(False,(255,130,115)),(True,(94,235,190))]:
            ids = np.flatnonzero(flags==accepted)
            selected = sorted(ids,key=lambda i: matches[i].distance)[:50]
            for i in selected:
                m = matches[i]
                x,y = np.rint(self.ka[m.queryIdx].pt).astype(int)
                u,v = np.rint(self.kb[m.trainIdx].pt).astype(int)
                cv2.line(view,(x,y),(u+self.left.shape[1],v),color,2,cv2.LINE_AA)
        return view, panorama, len(matches), int(flags.sum()), error


def gui():
    root = tk.Tk()
    root.title("局部特征与图像拼接 / Локальные признаки и панорама")
    root.configure(bg="#101924")
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background="#101924")
    style.configure("TLabel", background="#101924", foreground="#eaf3fa",
                    font=("Microsoft YaHei UI",10))
    style.configure("Title.TLabel", font=("Microsoft YaHei UI",20,"bold"))
    body = ttk.Frame(root,padding=16)
    body.pack(fill="both",expand=True)
    ttk.Label(body,text="图像匹配与拼接",style="Title.TLabel").pack(anchor="w")
    ttk.Label(body,text="Сопоставление изображений и панорама · SIFT / RANSAC").pack(anchor="w")
    tools = ttk.Frame(body)
    tools.pack(fill="x",pady=10)
    status = ttk.Label(body)
    status.pack(anchor="w",pady=(0,8))
    tabs = ttk.Notebook(body)
    tabs.pack(fill="both",expand=True)
    views = []
    for title in ("原始照片 / Исходные фото","关键点 / Ключевые точки","匹配 / Соответствия","拼接 / Панорама"):
        tab = ttk.Frame(tabs,padding=8)
        tabs.add(tab,text=title)
        panel = ttk.Label(tab,anchor="center")
        panel.pack(fill="both",expand=True)
        views.append(panel)
    ratio = tk.DoubleVar(value=.75)
    tolerance = tk.DoubleVar(value=3)
    pair = Pair(*demo())
    panorama = pending = None

    def paint(panel, array):
        if array is None:
            panel.image = None
            panel.configure(image="",text="未获得可靠拼接 / Не удалось построить панораму")
            return
        image = Image.fromarray(array)
        image.thumbnail((min(1020,root.winfo_screenwidth()-100),420))
        panel.image = ImageTk.PhotoImage(image)
        panel.configure(image=panel.image,text="")

    def side_by_side(left, right):
        board = np.zeros((max(len(left),len(right)),left.shape[1]+right.shape[1]+8,3),np.uint8)
        board[:len(left),:left.shape[1]] = left
        board[:len(right),left.shape[1]+8:] = right
        return board

    def set_pair(new_pair):
        nonlocal pair
        pair = new_pair
        paint(views[0], side_by_side(pair.left, pair.right))
        left = cv2.drawKeypoints(pair.left,pair.ka,None,(94,235,190),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        right = cv2.drawKeypoints(pair.right,pair.kb,None,(94,235,190),
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        paint(views[1], side_by_side(left, right))
        refresh()

    def refresh():
        nonlocal pending, panorama
        pending = None
        match_view, panorama, matches, inliers, error = pair.run(ratio.get(),tolerance.get())
        paint(views[2],match_view)
        paint(views[3],panorama)
        rmse = f"{error:.2f} px" if error is not None else "—"
        rate = inliers/matches if matches else 0
        status.configure(text=f"关键点 / Точки {len(pair.ka)} + {len(pair.kb)}    "
                         f"匹配 / Пары {matches}    内点 / Инлайеры {inliers} ({rate:.1%})    RMSE {rmse}")

    def schedule(_):
        nonlocal pending
        if pending is not None:
            root.after_cancel(pending)
        pending = root.after(110,refresh)

    def open_pair():
        left = filedialog.askopenfilename(title="选择图像 A / Выберите изображение A")
        if not left:
            return
        right = filedialog.askopenfilename(title="选择图像 B / Выберите изображение B")
        if right:
            try:
                set_pair(Pair(load(left),load(right)))
            except (OSError,cv2.error) as exc:
                messagebox.showerror("读取失败 / Ошибка",str(exc))

    def save():
        if panorama is None:
            messagebox.showinfo("拼接 / Панорама","当前没有拼接结果 / Нет панорамы")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("PNG","*.png")])
        if path:
            Image.fromarray(panorama).save(path)

    for title, action in [("打开两张照片 / Открыть",open_pair),
                          ("默认照片 / Пример",lambda: set_pair(Pair(*demo()))),
                          ("保存拼接 / Сохранить",save)]:
        ttk.Button(tools,text=title,command=action).pack(side="left",padx=(0,6))
    for title,var,lo,hi,step in [("Ratio test",ratio,.40,.95,.01),
                                 ("RANSAC (px)",tolerance,.5,10,.5)]:
        tk.Scale(tools,label=title,variable=var,from_=lo,to=hi,resolution=step,
                 orient="horizontal",length=145,command=schedule,bg="#101924",
                 fg="#eaf3fa",troughcolor="#42637a",highlightthickness=0).pack(side="left",padx=8)
    ttk.Label(body,text="绿色：内点 / Зелёный: инлайеры     红色：外点 / Красный: выбросы").pack(anchor="w",pady=(8,0))
    set_pair(pair)
    tabs.select(0)
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export",type=Path,help="export the default panorama as PNG")
    args = parser.parse_args()
    if args.export:
        pair = Pair(*demo())
        _, panorama, matches, inliers, error = pair.run(.75,3)
        if panorama is None:
            raise SystemExit("Stitching failed")
        Image.fromarray(panorama).save(args.export)
        print(f"SIFT {len(pair.ka)} + {len(pair.kb)} | matches {matches} | inliers {inliers} | RMSE {error:.3f} px")
    else:
        gui()



