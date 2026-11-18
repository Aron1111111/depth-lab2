"""Cat/dog CNN course experiment. The manifest fixes all data partitions."""
import argparse
import copy
import json
import os
from pathlib import Path
import queue
import random
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageTk
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "results"
SIZE = 64
LABELS = ["猫 / Кошка", "狗 / Собака"]
torch.set_num_threads(min(4, os.cpu_count() or 1))


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Dropout(.3), nn.Linear(64, 2))

    def forward(self, x):
        return self.classifier(self.features(x))


def read_manifest():
    manifest = json.loads((BASE / "split_manifest.json").read_text(encoding="utf-8"))
    sets = [{row["path"] for row in manifest[key]} for key in ("train", "validation", "test")]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("数据划分存在重叠 / Пересечение выборок")
    return manifest


def open_photo(row):
    with Image.open(BASE.parent / "task6" / row["path"]) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def preprocessing(augment=False):
    operations = [transforms.Resize((SIZE, SIZE))]
    if augment:
        operations += [transforms.RandomHorizontalFlip(),
                       transforms.RandomAffine(12, translate=(.08, .08), scale=(.9, 1.1))]
    return transforms.Compose(operations + [transforms.ToTensor(),
                              transforms.Normalize((.5,) * 3, (.5,) * 3)])


class Photos(Dataset):
    def __init__(self, rows, augment=False):
        self.images = [open_photo(row).resize((SIZE, SIZE), Image.Resampling.BILINEAR) for row in rows]
        self.targets = [int(row["label"] == "dog") for row in rows]
        self.transform = preprocessing(augment)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.targets[index]


def metrics(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predicted = probabilities.argmax(axis=1)
    matrix = np.zeros((2, 2), dtype=int)
    np.add.at(matrix, (labels, predicted), 1)
    tp = matrix.diagonal()
    precision = tp / np.maximum(1, matrix.sum(axis=0))
    recall = tp / np.maximum(1, matrix.sum(axis=1))
    f1 = 2 * precision * recall / np.maximum(1e-12, precision + recall)
    positive, negative = probabilities[labels == 1, 1], probabilities[labels == 0, 1]
    auc = float(((positive[:, None] > negative).sum()
                 + .5 * (positive[:, None] == negative).sum()) / (len(positive) * len(negative)))
    return dict(accuracy=float((predicted == labels).mean()), precision=float(precision.mean()),
                recall=float(recall.mean()), f1=float(f1.mean()), roc_auc=auc,
                confusion=matrix.tolist(), support=matrix.sum(axis=1).tolist())


def evaluate(model, loader, stop):
    model.eval()
    total_loss, labels, scores = 0., [], []
    with torch.inference_mode():
        for x, y in loader:
            if stop.is_set():
                raise InterruptedError("训练已停止 / Обучение остановлено")
            logits = model(x)
            total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            labels.extend(y.tolist())
            scores.extend(logits.softmax(dim=1).tolist())
    return total_loss / len(labels), metrics(labels, scores)


def train(epochs=10, learning_rate=.001, batch_size=64, augment=True, notify=None, stop=None):
    stop = stop or threading.Event()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    manifest = read_manifest()
    model = SmallCNN()
    if notify:
        notify("正在读取照片 / Загрузка фотографий…")
    loaders = {name: DataLoader(Photos(rows, augment and name == "train"),
                               batch_size=batch_size, shuffle=name == "train", num_workers=0)
               for name, rows in manifest.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history, best_loss, best_epoch = [], float("inf"), 0
    best = None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct, count = 0., 0, 0
        for x, y in loaders["train"]:
            if stop.is_set():
                raise InterruptedError("训练已停止 / Обучение остановлено")
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += int((logits.argmax(dim=1) == y).sum())
            count += len(y)
        val_loss, val = evaluate(model, loaders["validation"], stop)
        history.append(dict(epoch=epoch, train_loss=loss_sum/count, val_loss=val_loss,
                            train_accuracy=correct/count, val_accuracy=val["accuracy"]))
        if val_loss < best_loss:
            best_loss, best_epoch, best = val_loss, epoch, copy.deepcopy(model.state_dict())
        if notify:
            notify(f"Epoch {epoch}/{epochs} · loss {loss_sum/count:.4f} · "
                   f"val loss {val_loss:.4f} · val accuracy {val['accuracy']:.1%}")
    model.load_state_dict(best)
    _, validation = evaluate(model, loaders["validation"], stop)
    _, test = evaluate(model, loaders["test"], stop)
    report = dict(epochs=epochs, best_epoch=best_epoch, learning_rate=learning_rate,
                  batch_size=batch_size, augment=augment, image_size=SIZE,
                  parameters=sum(p.numel() for p in model.parameters()),
                  counts={name: len(rows) for name, rows in manifest.items()},
                  history=history, validation=validation, test=test,
                  seconds=time.perf_counter()-started)
    OUTPUT.mkdir(exist_ok=True)
    temporary = OUTPUT / "cnn.tmp"
    torch.save(dict(state=best, report=report), temporary)
    temporary.replace(OUTPUT / "cnn.pt")
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for page, name in [("架构与特征 / Архитектура", "architecture.png"),
                       ("训练过程 / Обучение", "training.png"),
                       ("识别与评价 / Оценка", "evaluation.png")]:
        render(page, report, model).save(OUTPUT / name)
    return model.eval(), report


def load_saved():
    path = OUTPUT / "cnn.pt"
    if not path.exists():
        return None, None
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SmallCNN()
    model.load_state_dict(checkpoint["state"])
    return model.eval(), checkpoint["report"]


def render(page, report=None, model=None, photo=None, channel=0):
    board = Image.new("RGB", (1200, 960), "#101924")
    draw = ImageDraw.Draw(board)

    def text(x, y, value, size=23, color="#eaf3fa"):
        draw.text((x,y), value, font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size), fill=color)

    text(26, 18, "猫狗 CNN 实验 / CNN для кошек и собак", 32)
    text(26, 72, page, 26, "#74ddcf")
    if page == "架构与特征 / Архитектура":
        text(26,126,"任务：整张照片判为猫或狗 / Классификация всего изображения",25)
        text(26,173,"选择 CNN：卷积学习局部特征，池化压缩特征图 / CNN: свёртки и pooling",23)
        layers = ["RGB\n3×64×64", "Conv + ReLU\n16×64×64", "Pool 2×2\n16×32×32",
                  "Conv + Pool\n32×16×16", "Conv + Pool\n64×8×8", "FC + Softmax\nCat / Dog"]
        for i, label in enumerate(layers):
            x = 26+i*195
            draw.rounded_rectangle((x,235,x+180,335),12,fill="#26394b")
            text(x+10,253,label,19)
        if photo is None:
            photo = open_photo(read_manifest()["test"][0])
        tile = ImageOps.fit(photo, (250,250))
        board.paste(tile,(26,405))
        text(26,365,"输入 / Вход",23)
        if model is not None:
            tensor = preprocessing()(photo).unsqueeze(0)
            with torch.inference_mode():
                activation = model.features[1](model.features[0](tensor))
                pooled = model.features[2](activation)
            for array, x, title in [(activation[0,channel].numpy(),345,"ReLU · 64×64"),
                                    (pooled[0,channel].numpy(),665,"MaxPool · 32×32")]:
                array = (array-array.min())/max(float(np.ptp(array)),1e-8)
                heatmap = Image.fromarray(np.uint8(array*255)).resize((250,250),Image.Resampling.NEAREST)
                board.paste(heatmap,(x,405))
                text(x,365,title,23)
            text(26,690,f"第一层通道 / Канал первого слоя  {channel}",24)
        else:
            text(320,440,"训练后显示卷积与池化特征图",26)
            text(320,490,"Карты признаков появятся после обучения",22)
        text(26,755,"训练增强示例 / Пример аугментации",25)
        for i, example in enumerate([photo, ImageOps.mirror(photo), photo.rotate(12)]):
            board.paste(ImageOps.fit(example,(160,140)),(26+i*195,802))
    elif report is None:
        text(26,160,"请先训练 / Сначала обучите модель",28)
    elif page == "训练过程 / Обучение":
        text(26,130,f"最佳轮次 / Лучшая эпоха {report['best_epoch']} · "
             f"{report['parameters']:,} parameters · {report['seconds']:.0f} s",24)
        for panel, (keys, title) in enumerate([
                (("train_loss","val_loss"),"交叉熵损失 / Cross-entropy"),
                (("train_accuracy","val_accuracy"),"准确率 / Accuracy")]):
            left, top, width, height = 70, 240+panel*340, 1060, 230
            text(26,top-60,title,25)
            maximum = max(1.,max(row[key] for row in report["history"] for key in keys))
            for tick in range(5):
                y = top+height-tick*height/4
                draw.line((left,y,left+width,y),fill="#34485b")
                text(10,y-12,f"{maximum*tick/4:.2f}",16)
            for key, color in zip(keys,["#74ddcf","#f3b66d"]):
                points=[(left+i*width/max(1,len(report['history'])-1),
                         top+height-row[key]/maximum*height) for i,row in enumerate(report["history"])]
                if len(points)>1:
                    draw.line(points,fill=color,width=4)
                for x,y in points:
                    draw.ellipse((x-4,y-4,x+4,y+4),fill=color)
            for i,row in enumerate(report["history"]):
                text(left+i*width/max(1,len(report['history'])-1)-8,top+height+10,str(row["epoch"]),17)
        text(26,910,"绿色 Train · 橙色 Validation · 按验证损失选择模型 / Выбор по validation loss",21)
    else:
        text(26,132,"测试集评价 / Оценка на test · Precision / Recall / F1 为宏平均",23)
        headers = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
        for col,key in enumerate(headers):
            text(260+col*180,195,key,21)
        comparisons = {"CNN":report["test"]}
        baseline_file = BASE/"baseline_metrics.json"
        if baseline_file.exists():
            baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
            comparisons.update({name:value["test"] for name,value in baseline["results"].items()})
        for row,(name,values) in enumerate(comparisons.items()):
            text(26,250+row*58,name,24,"#74ddcf" if row==0 else "#f3b66d")
            for col,key in enumerate(["accuracy","precision","recall","f1","roc_auc"]):
                text(260+col*180,250+row*58,f"{values[key]:.4f}",24)
        text(26,465,"CNN 混淆矩阵 / Матрица ошибок · 行为真实，列为预测",24)
        matrix = report["test"]["confusion"]
        for i,label in enumerate(LABELS):
            text(205+i*255,525,label,23)
            text(26,610+i*125,label,21)
            for j in range(2):
                x,y=190+j*255,570+i*125
                draw.rounded_rectangle((x,y,x+240,y+110),10,
                                       fill="#237c73" if i==j else "#9e5149")
                text(x+90,y+35,str(matrix[i][j]),32)
        text(770,550,"验证集 / Validation",25)
        text(770,600,f"Accuracy {report['validation']['accuracy']:.3f}",24)
        text(770,650,f"F1 {report['validation']['f1']:.3f}",24)
        text(26,875,"相同数据划分对比 / Сравнение на одинаковом разбиении",23)
    return board


def gui():
    root = tk.Tk()
    root.title("猫狗 CNN 实验 / CNN для кошек и собак")
    width = min(1260,root.winfo_screenwidth()-70)
    root.geometry(f"{width}x{min(940,root.winfo_screenheight()-80)}")
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".",font=("Microsoft YaHei UI",12))
    style.configure("TFrame",background="#101924")
    style.configure("TLabel",background="#101924",foreground="#eaf3fa")
    style.configure("TButton",padding=(14,10),font=("Microsoft YaHei UI",12,"bold"))
    body=ttk.Frame(root,padding=14)
    body.pack(fill="both",expand=True)
    ttk.Label(body,text="猫狗 CNN / Кошки и собаки",font=("Microsoft YaHei UI",21,"bold")).pack(anchor="w")
    controls=ttk.Frame(body)
    controls.pack(fill="x",pady=8)
    epochs=tk.IntVar(value=10)
    lr=tk.DoubleVar(value=-3)
    augment=tk.BooleanVar(value=True)
    channel=tk.IntVar(value=0)
    editable=[]
    for label,var,low,high,step in [("轮次 / Эпохи",epochs,2,30,1),
                                   ("log10(learning rate)",lr,-4,-2,.25)]:
        slider=tk.Scale(controls,label=label,variable=var,from_=low,to=high,resolution=step,
                         orient="horizontal",length=245,width=18,sliderlength=36,
                         bg="#101924",fg="white",highlightthickness=0,troughcolor="#36516a")
        slider.pack(side="left",padx=8)
        editable.append(slider)
    toggle=tk.Checkbutton(controls,text="数据增强 / Аугментация",variable=augment,
        indicatoron=False,padx=12,pady=12,font=("Microsoft YaHei UI",12,"bold"),
        bg="#26394b",fg="white",selectcolor="#247f74",activebackground="#36516a")
    toggle.pack(side="left",padx=10)
    editable.append(toggle)
    actions=ttk.Frame(body)
    actions.pack(fill="x",pady=5)
    nav=ttk.Frame(body)
    nav.pack(fill="x",pady=8)
    status=ttk.Label(body,text="CPU · 64×64 RGB · CNN")
    status.pack(anchor="w")
    area=ttk.Frame(body)
    area.pack(fill="both",expand=True,pady=6)
    canvas=tk.Canvas(area,bg="#101924",highlightthickness=0)
    scroll=ttk.Scrollbar(area,orient="vertical",command=canvas.yview)
    scroll.pack(side="right",fill="y")
    canvas.configure(yscrollcommand=scroll.set)
    canvas.pack(fill="both",expand=True)
    canvas.bind("<MouseWheel>",lambda event:canvas.yview_scroll(-int(event.delta/120),"units"))
    model,report=load_saved()
    photo=open_photo(read_manifest()["test"][0])
    page=tk.StringVar(value="架构与特征 / Архитектура")
    messages=queue.Queue()
    stop=threading.Event()
    worker=None
    closed=False

    def show():
        image=render(page.get(),report,model,photo,channel.get())
        image.thumbnail((max(500,width-65),2000))
        canvas.image=ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.create_image(0,0,image=canvas.image,anchor="nw")
        canvas.configure(scrollregion=(0,0,*image.size))
        canvas.yview_moveto(0)

    def launch():
        nonlocal worker
        stop.clear()
        for widget in editable:
            widget.configure(state="disabled")
        start_button.configure(state="disabled")
        # Snapshot Tk variables on the UI thread before starting computation.
        epoch_value,lr_value,augment_value=epochs.get(),10**lr.get(),augment.get()
        def work_snapshot():
            try:
                result=train(epoch_value,lr_value,augment=augment_value,
                             notify=lambda text:messages.put(("status",text)),stop=stop)
                messages.put(("done",result))
            except Exception as exc:
                messages.put(("error",str(exc)))
        worker=threading.Thread(target=work_snapshot,daemon=True)
        worker.start()

    def poll():
        nonlocal model,report
        if closed:
            return
        while not messages.empty():
            kind,payload=messages.get_nowait()
            if kind=="status":
                status.configure(text=payload)
            else:
                for widget in editable:
                    widget.configure(state="normal")
                start_button.configure(state="normal")
                if kind=="done":
                    model,report=payload
                    status.configure(text=f"完成 / Готово · best epoch {report['best_epoch']} · test accuracy {report['test']['accuracy']:.1%}")
                    page.set("识别与评价 / Оценка")
                    show()
                else:
                    status.configure(text=payload)
                    if not stop.is_set():
                        messagebox.showerror("错误 / Ошибка",payload)
        root.after(150,poll)

    def open_image():
        nonlocal photo
        path=filedialog.askopenfilename(filetypes=[("Images","*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            with Image.open(path) as source:
                photo=ImageOps.exif_transpose(source).convert("RGB")
            if model is not None:
                started=time.perf_counter()
                with torch.inference_mode():
                    probabilities=model(preprocessing()(photo).unsqueeze(0)).softmax(dim=1)[0]
                label=int(probabilities.argmax())
                status.configure(text=f"{LABELS[label]} · softmax {float(probabilities[label]):.1%} · {(time.perf_counter()-started)*1000:.1f} ms")
            page.set("架构与特征 / Архитектура")
            show()
        except Exception as exc:
            messagebox.showerror("照片 / Фото",str(exc))

    start_button=ttk.Button(actions,text="训练 CNN / Обучить",command=launch)
    start_button.pack(side="left",padx=(0,10))
    ttk.Button(actions,text="停止 / Стоп",command=stop.set).pack(side="left",padx=(0,10))
    ttk.Button(actions,text="打开照片并识别 / Открыть фото",command=open_image).pack(side="left")
    feature_slider=tk.Scale(actions,label="特征通道 / Канал",variable=channel,from_=0,to=15,
        orient="horizontal",length=200,width=16,sliderlength=30,bg="#101924",fg="white",
        highlightthickness=0,command=lambda _:show())
    feature_slider.pack(side="right")
    for column,title in enumerate(["架构与特征 / Архитектура","训练过程 / Обучение","识别与评价 / Оценка"]):
        tk.Radiobutton(nav,text=title,variable=page,value=title,indicatoron=False,
            font=("Microsoft YaHei UI",13,"bold"),padx=15,pady=12,bg="#26394b",fg="white",
            selectcolor="#247f74",activebackground="#36516a",command=show).grid(row=0,column=column,sticky="ew",padx=3)
        nav.columnconfigure(column,weight=1)
    def close():
        nonlocal closed
        closed=True
        stop.set()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW",close)
    show()
    poll()
    root.mainloop()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--train",action="store_true")
    parser.add_argument("--epochs",type=int,default=10)
    args=parser.parse_args()
    if args.train:
        if not 1<=args.epochs<=100:
            parser.error("epochs must be between 1 and 100")
        _,result=train(epochs=args.epochs,notify=lambda text:print(text,flush=True))
        print(json.dumps(result["test"],indent=2))
    else:
        gui()
