import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

BASE = Path(__file__).resolve().parent
DEFAULT = BASE.parent / "task1" / "car.jpg"
PARAMETERS = ("α", "β", "γ", "分块边长 / Блок (px)", "裁剪限值 / Предел", "截止频率 / Срез (cycles/px)")
VALUES = ("1.12", "-12", "0.75", "64", "2.5", "0.065")


def quantize(a):
    return Image.fromarray(np.rint(np.clip(a, 0, 255)).astype(np.uint8))


def clahe(a, tile, limit):
    h, w = a.shape
    padded = np.pad(a, ((0, -h % tile), (0, -w % tile)), mode="edge")
    ny, nx = padded.shape[0] // tile, padded.shape[1] // tile
    lut = np.empty((ny, nx, 256))
    cap = max(1, int(limit * tile**2 / 256))
    for j in range(ny):
        for i in range(nx):
            block = padded[j*tile:(j+1)*tile, i*tile:(i+1)*tile]
            hist = np.bincount(block.ravel(), minlength=256)
            clipped = np.minimum(hist, cap)
            excess = int((hist - clipped).sum())
            clipped += excess // 256
            remainder = excess % 256
            if remainder:
                clipped[np.arange(remainder)*256//remainder] += 1
            lut[j, i] = clipped.cumsum() * (255 / tile**2)
    y, x = np.ogrid[:h, :w]
    gx = np.clip((x+.5)/tile-.5, 0, nx-1)
    gy = np.clip((y+.5)/tile-.5, 0, ny-1)
    i, j = gx.astype(int), gy.astype(int)
    u, v = gx-i, gy-j
    k, l = np.minimum(i+1, nx-1), np.minimum(j+1, ny-1)
    return quantize((1-v)*((1-u)*lut[j,i,a]+u*lut[j,k,a])
                    + v*((1-u)*lut[l,i,a]+u*lut[l,k,a]))


def score(reference, im):
    mse = np.mean((reference-np.asarray(im))**2)
    psnr = 10*np.log10(255**2/mse) if mse else np.inf
    return f"MSE {mse:.1f}   PSNR {psnr:.2f} dB"


def render(path, values):
    alpha, beta, gamma, tile, limit, cutoff = map(float, values)
    if not (0 < alpha <= 10 and -255 <= beta <= 255 and 0 < gamma <= 10
            and tile.is_integer() and 8 <= tile <= 512 and 1 <= limit <= 20
            and 0 < cutoff <= .5):
        raise ValueError("参数超出范围 / Недопустимые параметры")
    with Image.open(path) as source:
        ref = ImageOps.exif_transpose(source).convert("L")
    ref.thumbnail((960, 960), Image.Resampling.LANCZOS)
    a = np.asarray(ref)
    f = a.astype(float)
    linear = quantize(alpha*f+beta)
    adjusted = quantize(255*(np.clip(alpha*f+beta,0,255)/255)**gamma)
    local1 = clahe(a, int(tile), limit)
    local2 = clahe(a, max(8, int(tile)//2), min(40, limit*2))
    rng = np.random.default_rng(42)
    uniform = rng.random(a.shape)
    noises = [quantize(f+rng.normal(0,20,a.shape)),
              quantize(np.where(uniform<.025,0,np.where(uniform>.975,255,a)))]
    transform = np.fft.fft2(f)
    radius = np.hypot(np.fft.fftfreq(a.shape[0])[:,None], np.fft.fftfreq(a.shape[1]))
    mask = np.exp(-.5*(radius/cutoff)**2)
    low = np.fft.ifft2(transform*mask).real
    high = np.fft.ifft2(transform*(1-mask)).real
    assert np.allclose(low+high, f)

    def spectrum(z):
        return quantize(255*np.log1p(np.abs(np.fft.fftshift(z)))/np.log1p(np.abs(transform).max()))

    canvas = Image.new("RGB", (1520, 2055), "#edf2f8")
    draw = ImageDraw.Draw(canvas)
    font_path = "C:/Windows/Fonts/msyh.ttc"

    def label(x, y, string, size):
        font = ImageFont.truetype(font_path, size)
        draw.text((x,y), string, font=font, fill="#172b46")

    def row(index, title, cards):
        y = 98+index*324
        label(24,y,title,24)
        for i,(im,zh,ru,detail) in enumerate(cards):
            x=24+i*370
            draw.rounded_rectangle((x,y+37,x+350,y+309), radius=12, fill="white")
            thumb=im.copy()
            thumb.thumbnail((330,181))
            canvas.paste(thumb,(x+(350-thumb.width)//2,y+91))
            label(x+10,y+43,zh,19)
            label(x+10,y+67,ru,16)
            label(x+10,y+282,detail,15)

    label(24,12,"图像增强与滤波实验",36)
    label(24,57,"Улучшение и фильтрация изображений",24)
    original=(ref,"灰度参考图","Исходное изображение",f"{ref.width} × {ref.height} px")
    row(0,"01  亮度、对比度与伽马 / Яркость, контраст и гамма",[
        original,(linear,"线性变换","Линейное преобразование",f"α={alpha:g}   β={beta:g}"),
        (adjusted,"线性变换后应用伽马","Затем гамма-коррекция",f"γ={gamma:g}")])
    row(1,"02  局部对比度 / Локальный контраст · CLAHE",[
        original,(local1,"参数组 A","Набор A",f"{tile:g} px   clip={limit:g}"),
        (local2,"参数组 B","Набор B",f"{max(8,int(tile)//2)} px   clip={min(40,limit*2):g}")])
    for index,(noisy,title) in enumerate(zip(noises,[
        "03a  高斯噪声 / Гауссов шум · σ=20","03b  椒盐噪声 / Импульсный шум · 5%"]),2):
        row(index,title,[original,(noisy,"加噪图","Изображение с шумом",score(f,noisy)),
            (g:=noisy.filter(ImageFilter.GaussianBlur(1.0)),"高斯滤波","Фильтр Гаусса · σ=1",score(f,g)),
            (m:=noisy.filter(ImageFilter.MedianFilter(3)),"中值滤波","Медианный фильтр · 3×3",score(f,m))])
    row(4,"04  傅里叶频谱与低通、高通 / Спектр Фурье, ФНЧ и ФВЧ",[
        (spectrum(transform),"原图频谱","Спектр оригинала","log(1 + |F|)"),
        (spectrum(transform*mask),"低通频谱","Спектр после ФНЧ",f"fc={cutoff:g} cycles/px"),
        (spectrum(transform*(1-mask)),"高通频谱","Спектр после ФВЧ",f"fc={cutoff:g} cycles/px")])
    row(5,"04  空间域重建 / Восстановление в пространственной области",[
        original,(quantize(low),"低通结果","Результат ФНЧ",""),
        (quantize(high+128),"高通结果（零值显示为 128）","Результат ФВЧ (ноль = 128)",f"fc={cutoff:g} cycles/px")])
    return canvas


def gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk

    root=tk.Tk()
    root.title("图像增强实验 / Улучшение изображений")
    path=tk.StringVar(value=str(DEFAULT))
    controls=ttk.Frame(root, padding=8)
    controls.pack(fill="x")
    entries=[]
    for i,(name,value) in enumerate(zip(PARAMETERS,VALUES)):
        ttk.Label(controls,text=name).grid(row=0,column=i)
        entry=ttk.Entry(controls,width=12)
        entry.insert(0,value)
        entry.grid(row=1,column=i,padx=4)
        entries.append(entry)
    view=tk.Canvas(root,width=980,height=640)
    scroll=ttk.Scrollbar(root,orient="vertical",command=view.yview)
    view.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right",fill="y")
    view.pack(fill="both",expand=True)
    result=None

    def refresh():
        nonlocal result
        try:
            result=render(path.get(),[e.get() for e in entries])
        except (OSError,ValueError) as exc:
            messagebox.showerror("错误 / Ошибка",str(exc))
            return
        preview=result.copy()
        preview.thumbnail((980,2000))
        view.photo=ImageTk.PhotoImage(preview)
        view.delete("all")
        view.create_image(0,0,image=view.photo,anchor="nw")
        view.configure(scrollregion=(0,0,*preview.size))

    def open_image():
        selected=filedialog.askopenfilename()
        if selected:
            path.set(selected)
            refresh()

    def save():
        if result is not None:
            selected=filedialog.asksaveasfilename(defaultextension=".png")
            if selected:
                result.save(selected)

    for i,(title,command) in enumerate([("打开 / Открыть",open_image),
        ("应用 / Применить",refresh),("保存 / Сохранить",save)]):
        ttk.Button(controls,text=title,command=command).grid(row=2,column=i,columnspan=1,pady=8)
    refresh()
    root.mainloop()


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--export",type=Path)
    parser.add_argument("--image",type=Path,default=DEFAULT)
    args=parser.parse_args()
    if args.export:
        render(args.image,VALUES).save(args.export)
    else:
        gui()

