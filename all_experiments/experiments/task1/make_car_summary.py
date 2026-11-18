import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

p = Path("car_results")
rows = list(csv.DictReader((p/"metrics.csv").open(encoding="utf-8")))
im = Image.new("RGB", (1600, 1700), "#eef2f7")
d = ImageDraw.Draw(im)
font_path = Path("C:/Windows/Fonts/msyh.ttc")
def txt(x, y, s, size=24, color="#172943"):
    d.text((x,y), s, font=ImageFont.truetype(str(font_path),size), fill=color)
def card(x,y,path,label,values=None, ru=""):
    d.rounded_rectangle((x,y,x+371,y+319), radius=12, fill="white")
    photo = Image.open(path).convert("RGB")
    photo.thumbnail((351,212))
    im.paste(photo,(x+(371-photo.width)//2,y+70))
    txt(x+10,y+7,label,21)
    txt(x+10,y+35,ru,17)
    if values:
        txt(x+10,y+288,f'MSE {float(values["mse"]):.2f}  |  PSNR {float(values["psnr_db"]):.2f} dB  |  SSIM {float(values["ssim"]):.4f}',16)
txt(32,20,"汽车照片｜图像失真与滤波实验",42)
txt(32,77,"Искажения и фильтрация изображения автомобиля",30)
txt(32,121,"灰度图计算：MSE 越低越好，PSNR 和 SSIM 越高越好。每项分数均与灰度参考图比较。",23)
txt(32,154,"Оценка по исходному изображению в оттенках серого: чем ниже MSE и выше PSNR и SSIM, тем лучше.",21)
names = ["高斯噪声（σ=20）","椒盐噪声（约 5%）","高斯模糊（半径 2）","高斯模糊（半径 5）"]
filters = {"median_3":"3×3 中值","median_5":"5×5 中值","gaussian_r2":"5×5 高斯","gaussian_r3":"7×7 高斯","mean_3":"3×3 均值","degraded":"保留失真图"}
for i,(case,title) in enumerate(zip(dict.fromkeys(r["distortion"] for r in rows),names)):
    y=198+i*372
    txt(28,y,title+"  /  "+["Гауссов шум (σ=20)","Импульсный шум (около 5%)","Гауссово размытие (радиус 2)","Гауссово размытие (радиус 5)"][i],24)
    group=[r for r in rows if r["distortion"]==case]
    card(28,y+34,Path("car.jpg"),"原始彩色照片",ru="Исходная цветная фотография")
    card(419,y+34,p/f"{case}_degraded.png","失真图",next(r for r in group if r["filter"]=="degraded"),"Искажённое изображение")
    for j,(metric,label) in enumerate([("psnr","PSNR 最高 / MSE 最低"),("ssim","SSIM 最高")]):
        r=next(r for r in group if r[f"best_{metric}"]=="1")
        f=filters.get(r["filter"],"反锐化 "+r["filter"].split("_")[-1])
        ru_label = "Макс. PSNR / мин. MSE" if metric == "psnr" else "Макс. SSIM"
        ru_filter = {"median_3":"медиана 3×3","median_5":"медиана 5×5","gaussian_r2":"Гаусс 5×5","gaussian_r3":"Гаусс 7×7","mean_3":"среднее 3×3","degraded":"без фильтра"}.get(r["filter"],"резкость "+r["filter"].split("_")[-1])
        card(810+j*391,y+34,p/f"{case}_best_{metric}.png",f"{label}：{f}",r,ru_label + ": " + ru_filter)
im.save(p/"all_results_zh_ru.png")

