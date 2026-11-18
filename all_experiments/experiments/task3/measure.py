from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


def gray(a):
    return Image.fromarray(np.rint(np.clip(a, 0, 255)).astype("uint8"))


def binary(mask):
    return gray(mask * 255)


def canny(image, sigma, low, high):
    a = np.asarray(image.filter(ImageFilter.GaussianBlur(sigma)), dtype=float)
    p = np.pad(a, 1, mode="edge")
    gx = (p[:-2,2:]+2*p[1:-1,2:]+p[2:,2:]
          -p[:-2,:-2]-2*p[1:-1,:-2]-p[2:,:-2])
    gy = (p[2:,:-2]+2*p[2:,1:-1]+p[2:,2:]
          -p[:-2,:-2]-2*p[:-2,1:-1]-p[:-2,2:])
    mag = np.hypot(gx, gy)
    angles = (np.floor((np.degrees(np.arctan2(gy,gx))+22.5)/45) % 4).astype(int)
    thin = np.zeros_like(mag)
    for direction,(dy,dx) in enumerate([(0,1),(1,1),(1,0),(1,-1)]):
        forward = np.roll(mag, (dy,dx), axis=(0,1))
        backward = np.roll(mag, (-dy,-dx), axis=(0,1))
        keep = (angles==direction)&(mag>=forward)&(mag>backward)
        thin[keep] = mag[keep]
    thin[[0,-1],:] = 0
    thin[:,[0,-1]] = 0
    edges = thin >= high
    weak = thin >= low
    queue = deque(zip(*np.nonzero(edges)))
    while queue:
        y,x = queue.popleft()
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                j,i = y+dy,x+dx
                if weak[j,i] and not edges[j,i]:
                    edges[j,i] = True
                    queue.append((j,i))
    return edges


def otsu(a):
    hist = np.bincount(a.ravel(), minlength=256)
    count = hist.cumsum()
    sums = (hist*np.arange(256)).cumsum()
    den = count*(int(count[-1])-count)
    between = np.divide((sums*float(count[-1])-sums[-1]*count)**2,
                        den, out=np.zeros(256), where=den>0)
    return int(between.argmax())


def measure(mask):
    p = np.pad(mask, 1)
    perimeter = np.count_nonzero(p[:,1:] != p[:,:-1])
    perimeter += np.count_nonzero(p[1:,:] != p[:-1,:])
    return int(mask.sum()), perimeter


def main():
    with Image.open(Path(__file__).with_name("cat.jpg")) as source:
        photo = ImageOps.exif_transpose(source).convert("RGB")
    gray_photo = photo.convert("L")
    edges = [canny(gray_photo, *params) for params in [(1,30,80),(3,80,150)]]
    a = np.asarray(gray_photo)
    t = otsu(a)
    global_mask = a > t
    local_mean = np.asarray(gray_photo.filter(ImageFilter.BoxBlur(55)), dtype=float)
    local_mask = a > local_mean + 12
    dirty = local_mask

    initial = binary(dirty)
    erode = ImageFilter.MinFilter(7)
    dilate = ImageFilter.MaxFilter(7)
    stages = [initial,initial.filter(erode),initial.filter(dilate),
              initial.filter(erode).filter(dilate),initial.filter(dilate).filter(erode),
              initial.filter(erode).filter(dilate).filter(dilate).filter(erode)]
    measured = [measure(np.asarray(im)>0) for im in stages]
    assert measured[1][0] <= measured[0][0] <= measured[2][0]
    assert all(e.any() for e in edges)

    board = Image.new("RGB",(1560,1480),"#eaf0f7")
    draw = ImageDraw.Draw(board)

    def label(x,y,s,size):
        draw.text((x,y),s,fill="#172b46",
                  font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",size))

    def row(index, title, cards):
        y = 123 + index*332
        label(24,y,title,25)
        for j,(im,caption,ru,note) in enumerate(cards):
            x = 24+j*378
            draw.rounded_rectangle((x,y+39,x+354,y+317),12,fill="white")
            label(x+12,y+45,caption,19)
            label(x+12,y+70,ru,16)
            board.paste(im,(x+(354-im.width)//2,y+95))
            label(x+12,y+279 if im.height<185 else y+296,note,14)

    # Reduce images for the cards only. Measurements use original pixels.
    def card(im, caption, ru, note=""):
        return (ImageOps.contain(im,(320,180), Image.Resampling.NEAREST),caption,ru,note)

    label(24,14,"边缘检测、二值化与形态学测量",36)
    label(24,61,"Границы, бинаризация и морфологические измерения",26)
    row(0,"01   Canny 边缘检测 / Границы Кэнни",[
        card(photo,"原始照片","Исходная фотография",f"{photo.width} × {photo.height} px"),
        *[card(binary(e),"较低阈值" if i==0 else "较高阈值",
               "Низкие пороги" if i==0 else "Высокие пороги",
               f"{p}   |   N={e.sum()}") for i,(e,p) in enumerate(zip(edges,["σ=1, low=30, high=80","σ=3, low=80, high=150"]))]])
    row(1,"02   灰度图二值化 / Бинаризация в оттенках серого",[
        card(gray_photo,"灰度输入","Вход в оттенках серого"),
        card(binary(global_mask),"全局 Otsu","Глобальный порог Оцу",f"T={t}"),
        card(binary(local_mask),"自适应局部均值","Адаптивный локальный порог","T=mean(111×111)+12")])

    names = [("自适应掩膜","Адаптивная маска"),("腐蚀","Эрозия"),("膨胀","Дилатация"),
             ("开运算","Открытие"),("闭运算","Закрытие"),("先开后闭","Открытие → закрытие")]
    cards = [card(im,*name,f"A={area} px²   P4={p} px")
             for im,name,(area,p) in zip(stages,names,measured)]
    row(2,"03   形态学操作 / Морфология · 7×7",cards[:4])
    row(3,"04   面积与周长 / Площадь и периметр · A: 前景像素 / пиксели объекта · P4: 暴露边 / открытые стороны",[
        cards[0], *cards[4:]])
    board.save(Path(__file__).with_name("results_zh_ru_p4.png"))
    print("Canny edge counts:",*[int(e.sum()) for e in edges])
    print("Otsu threshold:", t)
    print("Area and perimeter:",measured)


if __name__ == "__main__":
    main()






