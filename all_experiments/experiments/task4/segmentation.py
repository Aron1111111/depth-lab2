from collections import deque
from pathlib import Path
from heapq import heappop, heappush
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageTk


PALETTE = [(28, 39, 56), (83, 187, 221)]


def load_dog():
    with Image.open(Path(__file__).with_name("dog.jpg")) as source:
        photo = ImageOps.exif_transpose(source).convert("RGB")
        photo.thumbnail((216, 144), Image.Resampling.LANCZOS)
    # Hand-traced visible silhouette. Coordinates refer to the 1920×1280 preview.
    outline = [
        (729,888),(740,849),(755,804),(769,779),(791,756),(816,740),
        (838,723),(847,697),(866,674),(878,641),(876,604),(863,607),
        (846,608),(837,592),(831,566),(839,526),(852,495),(876,476),
        (900,463),(916,443),(942,429),(976,422),(1008,423),(1040,428),
        (1071,441),(1095,460),(1127,472),(1147,491),(1155,520),
        (1167,544),(1171,574),(1163,600),(1145,617),(1113,631),
        (1111,649),(1132,673),(1158,695),(1184,717),(1207,741),
        (1219,775),(1220,810),(1207,824),(1170,829),(1142,841),
        (1121,871),(1108,914),(1100,940),(1086,956),(1082,966),
        (1063,980),(1036,980),(1024,969),(1018,946),(1019,897),
        (1025,855),(1033,832),(990,843),(954,840),(926,829),
        (894,830),(858,836),(831,839),(820,877),(817,908),(807,931),
        (790,936),(776,948),(750,944),(733,929),(726,911),
    ]
    reference = Image.new("L", photo.size)
    ImageDraw.Draw(reference).polygon(
        [(round(x*photo.width/1920), round(y*photo.height/1280)) for x,y in outline],
        fill=1,
    )
    return np.asarray(photo), np.asarray(reference)


def components(mask):
    padded = np.pad(mask, 1)
    result = np.zeros(padded.shape, np.int32)
    for number, (y, x) in enumerate(zip(*np.nonzero(mask)), 1):
        y, x = y+1, x+1
        if result[y, x]:
            continue
        result[y, x] = number
        queue = deque([(y, x)])
        while queue:
            j, i = queue.popleft()
            for r, c in ((j-1,i),(j+1,i),(j,i-1),(j,i+1)):
                if padded[r,c] and not result[r,c]:
                    result[r,c] = number
                    queue.append((r,c))
    return result[1:-1, 1:-1]


def watershed(surface, radius):
    # Approximate foreground and background markers, independent of the reference.
    y, x = np.ogrid[:surface.shape[0], :surface.shape[1]]
    labels = np.zeros(surface.shape, dtype=np.int32)
    labels[(x < 77) | (x > 144) | (y < 43) | (y > 113)] = 2
    labels[((x-112)/radius)**2 + ((y-68)/(radius*1.25))**2 <= 1] = 1
    fixed = labels > 0
    best = np.full(surface.shape, np.inf)
    queue = []
    for y, x in zip(*np.nonzero(fixed)):
        best[y,x] = surface[y,x]
        heappush(queue, (best[y,x], 0, y, x))
    hops = np.zeros(surface.shape, dtype=int)
    height, width = surface.shape
    while queue:
        level, steps, y, x = heappop(queue)
        if (level, steps) != (best[y,x], hops[y,x]):
            continue
        for j, i in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0 <= j < height and 0 <= i < width and not fixed[j,i]:
                cost = (max(level, surface[j,i]), steps+1)
                if cost < (best[j,i], hops[j,i]):
                    best[j,i], hops[j,i] = cost
                    labels[j,i] = labels[y,x]
                    heappush(queue, (*cost,j,i))
    return (labels == 1).astype(int)


def kmeans(image, k):
    pixels = image.reshape(-1, 3).astype(float)
    rng = np.random.default_rng(42)
    centers = [pixels[rng.integers(len(pixels))]]
    for _ in range(k-1):
        nearest = ((pixels[:,None]-centers)**2).sum(axis=2).min(axis=1)
        centers.append(pixels[rng.choice(len(pixels), p=nearest/nearest.sum())])
    centers = np.array(centers)
    for _ in range(20):
        labels = ((pixels[:,None]-centers)**2).sum(axis=2).argmin(axis=1)
        updated = np.array([pixels[labels==i].mean(axis=0) if (labels==i).any()
                            else centers[i] for i in range(k)])
        if np.allclose(updated, centers):
            break
        centers = updated
    labels = labels.reshape(image.shape[:2])
    # Select the cluster most frequent in a small patch of the forehead.
    patch = labels[51:55,108:114]
    foreground = np.bincount(patch.ravel(), minlength=k).argmax()
    return (labels == foreground).astype(int)


def boundary(labels):
    p = np.pad(labels, 1)
    return (labels>0) & ((labels!=p[:-2,1:-1]) | (labels!=p[2:,1:-1])
                         | (labels!=p[1:-1,:-2]) | (labels!=p[1:-1,2:]))


def metrics(truth, predicted, tolerance):
    g, p = truth>0, predicted>0
    tp, fp, fn = int((g&p).sum()), int((~g&p).sum()), int((g&~p).sum())
    tn = int((~g&~p).sum())
    iou = tp/(tp+fp+fn)
    dice = 2*tp/(2*tp+fp+fn)
    miou = (iou + tn/(tn+fp+fn))/2
    gb, pb = boundary(truth), boundary(predicted)
    kernel = ImageFilter.MaxFilter(2*tolerance+1)
    near_g = np.asarray(Image.fromarray(gb).convert("L").filter(kernel)) > 0
    near_p = np.asarray(Image.fromarray(pb).convert("L").filter(kernel)) > 0
    precision = (pb&near_g).sum()/max(1, pb.sum())
    recall = (gb&near_p).sum()/gb.sum()
    bf1 = 2*precision*recall/(precision+recall) if precision+recall else 0
    return iou, dice, miou, bf1, fp, fn


def errors(truth, predicted):
    g, p = truth>0, predicted>0
    rgb = np.full((*g.shape, 3), (28, 39, 56), np.uint8)
    rgb[g&p] = (91, 173, 144)
    rgb[p&~g] = (245, 137, 99)
    rgb[g&~p] = (121, 163, 251)
    return rgb


def instance_errors(truth, predicted):
    labels = components(predicted > 0)
    hits = np.bincount(labels[truth > 0], minlength=int(labels.max())+1)
    regions = np.unique(labels[labels > 0])
    return int((hits[regions] == 0).sum()), int((hits[regions] > 0).sum())


def gui(check=False):
    image, truth = load_dog()
    gray = np.asarray(Image.fromarray(image).convert("L"))
    smooth = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(1)), dtype=float)
    dy, dx = np.gradient(smooth, axis=(0,1))
    surface = np.sqrt((dx**2 + dy**2).sum(axis=2))
    root = tk.Tk()
    root.title("图像分割实验 / Сегментация изображений")
    page, card, ink, muted = "#101924", "#1b2938", "#f0f5f9", "#b1c3d3"
    accent = "#74ddcf"
    root.configure(background=page)
    root.resizable(False, False)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", font=("Microsoft YaHei UI", 10))
    style.configure("Page.TFrame", background=page)
    style.configure("Card.TFrame", background=card)
    style.configure("Page.TLabel", background=page, foreground=muted)
    style.configure("Title.TLabel", background=page, foreground=ink,
                    font=("Microsoft YaHei UI", 22, "bold"))
    style.configure("Card.TLabel", background=card, foreground=ink)
    style.configure("Heading.TLabel", background=card, foreground=ink,
                    font=("Microsoft YaHei UI", 12, "bold"))
    style.configure("Detail.TLabel", background=card, foreground=muted)
    style.configure("Value.TLabel", background=card, foreground=accent,
                    font=("Consolas", 13, "bold"))
    style.configure("Reset.TButton", background="#26394b", foreground=ink, padding=(10,5))
    style.map("Reset.TButton", background=[("active", "#36516a")])

    body = ttk.Frame(root, padding=(18,14), style="Page.TFrame")
    body.pack(fill="both", expand=True)
    ttk.Label(body, text="图像分割实验", style="Title.TLabel").grid(
        row=0, column=0, columnspan=5, sticky="w")
    ttk.Label(body, text="Сегментация изображений", style="Page.TLabel").grid(
        row=1, column=0, columnspan=5, sticky="w", pady=(0,12))
    controls = ttk.Frame(body, padding=(12,8), style="Card.TFrame")
    controls.grid(row=2, column=0, columnspan=5, sticky="ew", padx=4, pady=(0,12))
    legend = ttk.Frame(body, style="Page.TFrame")
    legend.grid(row=4, column=0, columnspan=5, sticky="w", padx=4, pady=(9,0))
    for color, text in [("#5bad90","重叠 / Совпадение"), ("#f58963","多分 / FP"),
                        ("#79a3fb","漏分 / FN")]:
        tk.Label(legend, text="●", fg=color, bg=page).pack(side="left", padx=(0,4))
        ttk.Label(legend, text=text, style="Page.TLabel").pack(side="left", padx=(0,14))
    ttk.Label(legend, text="mIoU = 前景与背景均值 / среднее по объекту и фону",
              style="Page.TLabel").pack(side="left")

    names = [("输入图像","Исходное изображение"), ("手描参考掩膜","Ручная разметка"),
             ("全局阈值","Глобальный порог"), ("标记分水岭","Водораздел по маркерам"),
             ("RGB K-means","K-средних в RGB")]
    panels, maps, stats = [], [], []
    for col, (zh, ru) in enumerate(names):
        frame = ttk.Frame(body, padding=(9,8), style="Card.TFrame")
        frame.grid(row=3, column=col, sticky="nsew", padx=4)
        ttk.Label(frame, text=zh, style="Heading.TLabel").pack(anchor="w")
        ttk.Label(frame, text=ru, style="Detail.TLabel").pack(anchor="w", pady=(0,7))
        panels.append(ttk.Label(frame, style="Card.TLabel"))
        panels[-1].pack()
        if col >= 2:
            ttk.Label(frame, text="误差图 / Карта ошибок", style="Detail.TLabel").pack(
                anchor="w", pady=(7,3))
            maps.append(ttk.Label(frame, style="Card.TLabel"))
            maps[-1].pack()
            stats.append(ttk.Label(frame, justify="left", style="Card.TLabel"))
            stats[-1].pack(anchor="w", pady=(6,0))
        else:
            maps.append(None)
            stats.append(None)
    pending = None
    cached = {}

    def paint(widget, array):
        widget.image = ImageTk.PhotoImage(Image.fromarray(array.astype("uint8")))
        widget.configure(image=widget.image)

    def refresh():
        nonlocal pending
        pending = None
        threshold, radius, k, tolerance = [int(v.get()) for v in values]
        paint(panels[0], image)
        paint(panels[1], np.array(PALETTE)[truth])
        if "k" not in cached or cached["k"] != k:
            cached["kmeans"] = kmeans(image,k)
            cached["k"] = k
        if cached.get("radius") != radius:
            cached["watershed"] = watershed(surface, radius)
            cached["radius"] = radius
        predictions = [(gray>threshold).astype(int),cached["watershed"],cached["kmeans"]]
        for col, pred in enumerate(predictions,2):
            palette_index = np.where(pred>0, 1+(pred-1)%(len(PALETTE)-1), 0)
            paint(panels[col], np.array(PALETTE)[palette_index])
            paint(maps[col], errors(truth,pred))
            iou,dice,miou,bf1,fp,fn = metrics(truth,pred,tolerance)
            islands, fragments = instance_errors(truth,pred)
            info = f"IoU {iou:.3f}    Dice {dice:.3f}\nmIoU {miou:.3f}    Boundary F1 {bf1:.3f}"
            info += f"\nFP {fp}   FN {fn}\n孤岛 / островки {islands}\n目标碎片 / фрагменты {fragments}"
            stats[col].configure(text=info)

    def schedule(_):
        nonlocal pending
        if pending is not None:
            root.after_cancel(pending)
        pending = root.after(100, refresh)

    specs = [("灰度阈值 / Порог",0,255,125),
             ("种子半径 / Радиус (px)",3,16,8),
             ("聚类数 / Кластеры K",2,5,3),
             ("边界容差 / Допуск (px)",0,8,2)]
    values = [tk.IntVar(value=default) for _,_,_,default in specs]
    sliders = []
    for col, ((label, low, high, _), value) in enumerate(zip(specs,values)):
        field = ttk.Frame(controls, style="Card.TFrame")
        field.grid(row=0, column=col, sticky="ew", padx=8)
        controls.columnconfigure(col, weight=1)
        ttk.Label(field, text=label, style="Card.TLabel").pack(side="top", anchor="w")
        ttk.Label(field, textvariable=value, style="Value.TLabel", width=3).pack(side="right")
        slider = tk.Scale(field, from_=low, to=high, variable=value, showvalue=False,
                          orient="horizontal", length=150, width=9, sliderlength=18,
                          background=accent, activebackground="#a3f5e9", troughcolor="#40556b",
                          borderwidth=0, highlightthickness=1, highlightbackground=card,
                          highlightcolor=accent, command=schedule)
        slider.pack(side="left", fill="x", expand=True, pady=(5,0))
        sliders.append(slider)

    def reset():
        for value, (_, _, _, default) in zip(values, specs):
            value.set(default)
        schedule(None)

    ttk.Button(controls, text="重置 / Сброс", style="Reset.TButton", command=reset).grid(
        row=0, column=4, padx=(12,4))
    refresh()
    if check:
        root.update_idletasks()
        initial = stats[2].cget("text")
        root.after(100, lambda: sliders[0].set(240))
        root.after(800, root.quit)
        root.mainloop()
        assert stats[2].cget("text") != initial
        assert "IoU 1.000" in "IoU %.3f" % metrics(truth,truth,0)[0]
        root.destroy()
    else:
        root.mainloop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify the interface and slider callback")
    gui(parser.parse_args().check)




