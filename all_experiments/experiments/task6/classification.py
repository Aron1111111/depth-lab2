"""Image features -> classification -> held-out evaluation."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk, ImageOps
from types import SimpleNamespace
from functools import lru_cache
import hashlib
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = Path(__file__).resolve().parent



@lru_cache(maxsize=1)
def load_dataset():
    paths, targets, images, seen = [], [], [], set()
    folder=BASE/"dataset"/"cats_and_dogs_filtered"
    for label, category in enumerate(["cats","dogs"]):
        for path in sorted(folder.glob(f"*/{category}/*.jpg")):
            with Image.open(path) as photo:
                photo=ImageOps.exif_transpose(photo).convert("RGB")
                digest=hashlib.sha256(photo.tobytes()+str(photo.size).encode()).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                images.append(np.asarray(photo.convert("L").resize((64,64)),dtype=float)*16/255)
            paths.append(path)
            targets.append(label)
    if len(set(targets)) != 2:
        raise ValueError("请先下载猫狗数据集 / Сначала загрузите данные")
    return SimpleNamespace(images=np.array(images),target=np.array(targets),paths=paths)


FEATURE_CACHE={}

def features(images, kind):
    if kind == "Pixels":
        return images.reshape(len(images), -1).astype(float) / 16
    vectors = []
    for source in images:
        image = np.asarray(Image.fromarray((source * 255 / 16).astype("uint8")).resize(
            (64, 64), Image.Resampling.BILINEAR), dtype=float) / 255
        gy, gx = np.gradient(image)
        magnitude = np.hypot(gx, gy)
        bins = (np.arctan2(gy, gx) % np.pi) * 9 / np.pi
        cells = np.zeros((8, 8, 9))
        for y in range(8):
            for x in range(8):
                directions = bins[y*8:y*8+8, x*8:x*8+8].ravel()
                weights = magnitude[y*8:y*8+8, x*8:x*8+8].ravel()
                lower = np.floor(directions).astype(int)
                fraction = directions - lower
                cells[y, x] = (
                    np.bincount(lower % 9, weights * (1-fraction), minlength=9)
                    + np.bincount((lower+1) % 9, weights * fraction, minlength=9))
        blocks = []
        for y in range(7):
            for x in range(7):
                block = cells[y:y+2, x:x+2].ravel()
                block = block / np.sqrt(np.dot(block, block) + 1e-6)
                block = np.minimum(block, .2)
                blocks.extend(block / np.sqrt(np.dot(block, block) + 1e-6))
        vectors.append(blocks)
    return np.array(vectors)


def evaluate(target, predicted, probabilities):
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, predicted, labels=np.arange(2), average="macro", zero_division=0)
    return dict(accuracy=float(accuracy_score(target, predicted)),
                precision=float(precision), recall=float(recall), f1=float(f1),
                roc_auc=float(roc_auc_score(target, probabilities)))


def experiment(kind="HOG", c=10, trees=120, balanced=False, folds=3,
               train_fraction=.6, minority_keep=100):
    data = load_dataset()
    indices = np.arange(len(data.target))
    train, rest = train_test_split(indices, test_size=1-train_fraction, random_state=42,
                                   stratify=data.target)
    validation, test = train_test_split(rest, test_size=.5, random_state=42,
                                        stratify=data.target[rest])
    original_train = train.copy()
    # Simulate rare dogs only in training; hold-out sets retain their natural distribution.
    rare = train[data.target[train] == 1]
    keep = max(folds, round(len(rare)*minority_keep/100))
    train = np.concatenate([train[data.target[train] != 1], rare[:keep]])
    if kind not in FEATURE_CACHE:
        FEATURE_CACHE[kind]=features(data.images,kind)
    x=FEATURE_CACHE[kind]
    weight = "balanced" if balanced else None
    models = {
        "SVM": make_pipeline(StandardScaler(), SVC(
            C=c, kernel="rbf", class_weight=weight, random_state=42)),
        "Random Forest": RandomForestClassifier(
            n_estimators=trees, class_weight=weight, random_state=42, n_jobs=2),
    }
    results = {}
    for name, model in models.items():
        cv = cross_val_score(model, x[train], data.target[train],
                             cv=StratifiedKFold(folds, shuffle=True, random_state=42),
                             scoring="f1_macro", n_jobs=1)
        model.fit(x[train], data.target[train])
        record = {"cv_f1_mean": float(cv.mean()), "cv_f1_std": float(cv.std()),
                  "cv_scores": cv.tolist()}
        for title, subset in [("train", train), ("validation", validation), ("test", test)]:
            predicted = model.predict(x[subset])
            record[title] = evaluate(data.target[subset], predicted,
                                     (model.decision_function(x[subset]) if name == "SVM" else model.predict_proba(x[subset])[:,1]))
            precision, recall, f1, support = precision_recall_fscore_support(
                data.target[subset], predicted, labels=np.arange(2), zero_division=0)
            record[title]["per_class"] = dict(precision=precision.tolist(),
                recall=recall.tolist(), f1=f1.tolist(), support=support.tolist())
            if title == "test":
                record["confusion"] = confusion_matrix(
                    data.target[subset], predicted, labels=np.arange(2)).tolist()
                record["predicted"] = predicted.tolist()
        results[name] = record
    output=BASE/"results"
    output.mkdir(exist_ok=True)
    joblib.dump(dict(models=models,feature=kind),output/"models.joblib")
    manifest={title:[dict(path=str(data.paths[i].relative_to(BASE)),
                          label="cat" if data.target[i]==0 else "dog") for i in group]
              for title,group in [("train",train),("validation",validation),("test",test)]}
    (output/"split_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return dict(feature=kind, dimensions=x.shape[1], C=c, trees=trees, balanced=balanced,
                counts=[len(train), len(validation), len(test)], results=results,
                folds=folds, train_fraction=train_fraction, minority_keep=minority_keep,
                class_counts=[np.bincount(data.target[group], minlength=2).tolist()
                              for group in (train, validation, test)],
                train_before_sampling=len(original_train),
                test_indices=test.tolist())


def report_image(report):
    return dashboard(report,"测试评价 / Тест")


def save_report(report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for page,name in [("数据概览 / Данные","data_preview.png"),
                      ("训练与验证 / Обучение","validation_preview.png"),
                      ("测试评价 / Тест","evaluation_preview.png")]:
        dashboard(report,page).save(directory/name)


def dashboard(report, page):
    """Each page has one role; datasets are determined by that role."""
    image = Image.new("RGB", (1200, 1420 if page == "数据概览 / Данные" else 1150), "#101924")
    draw = ImageDraw.Draw(image)
    colors = ["#74ddcf", "#f3b66d"]
    names = list(report["results"])

    def text(x, y, value, size=23, color="#eaf3fa"):
        draw.text((x,y),value,font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",size),
                  fill=color)

    text(28,20,page,32)
    text(28,72,f"{report['feature']} · C={report['C']:g} · trees={report['trees']} · "
         f"class_weight={'balanced' if report['balanced'] else 'None'}",21)
    for i,name in enumerate(names):
        text(735+i*230,22,"● "+name,21,colors[i])

    if page == "数据概览 / Данные":
        text(28,125,"数据划分 / Разбиение",27)
        x = 28
        for count,color,title in zip(report["counts"],["#74ddcf","#79a3fb","#f3b66d"],
                                     ["Train","Validation","Test"]):
            width = int(1120*count/sum(report["counts"]))
            draw.rounded_rectangle((x,175,x+width-4,265),12,fill=color)
            text(x+12,199,f"{title} {count}",23,"#101924")
            x += width
        text(28,300,"各类样本数 / Число изображений",27)
        counts=report["class_counts"]
        colors_data=["#74ddcf","#79a3fb","#f3b66d"]
        for group,(title,color) in enumerate(zip(["Train","Validation","Test"],colors_data)):
            text(90+group*370,350,"● "+title,24,color)
        maximum=max(map(max,counts))
        for cls,label in enumerate(["猫 / Кошки","狗 / Собаки"]):
            x=200+cls*520
            for group,color in enumerate(colors_data):
                value=counts[group][cls]
                h=value/maximum*230
                draw.rounded_rectangle((x+group*70,680-h,x+group*70+52,680),6,fill=color)
                text(x+group*70,680-h-35,str(value),22,color)
            text(x+20,700,label,27)
        data=load_dataset()
        for cls,label in enumerate(["猫 / Кошки","狗 / Собаки"]):
            text(28,780+cls*275,label,28)
            for col,index in enumerate(np.flatnonzero(data.target==cls)[:6]):
                with Image.open(data.paths[index]) as photo:
                    tile=ImageOps.fit(ImageOps.exif_transpose(photo).convert("RGB"),(174,174))
                image.paste(tile,(28+col*195,830+cls*275))
        text(28,1360,"狗的保留比例仅影响训练集 / Доля собак изменяется только в train",21)
    elif page == "训练与验证 / Обучение":
        text(28,125,"训练集交叉验证 / Кросс-валидация на train · Macro F1",26)
        scores = [report["results"][name].get("cv_scores",[]) for name in names]
        if scores[0]:
            spacing = 1080/len(scores[0])
            for fold in range(len(scores[0])):
                for i in range(2):
                    x = 65+fold*spacing+i*40
                    value = scores[i][fold]
                    draw.rectangle((x,460-value*250,x+30,460),fill=colors[i])
                    text(x-8,460-value*250-25,f"{value:.3f}",16)
                text(65+fold*spacing,477,f"Fold {fold+1}",19)
        for i,name in enumerate(names):
            result = report["results"][name]
            text(35+i*590,530,f"{name}  {result['cv_f1_mean']:.4f} ± {result['cv_f1_std']:.4f}",24,colors[i])
        text(28,597,"验证集指标 / Метрики validation",27)
        headers = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
        for col,key in enumerate(headers):
            text(260+col*180,655,key,21)
        for i,name in enumerate(names):
            y = 705+i*65
            draw.rounded_rectangle((28,y-7,1165,y+46),8,fill="#1b2938")
            text(42,y,name,23,colors[i])
            for col,key in enumerate(["accuracy","precision","recall","f1","roc_auc"]):
                text(260+col*180,y,f"{report['results'][name]['validation'][key]:.4f}",24,colors[i])
        text(28,880,"调参依据验证集；CV 只使用训练集 / Подбор по validation, CV только на train",22)
        text(28,930,"Precision / Recall / F1 / AUC 均为宏平均 / Macro",21)

    elif page == "测试评价 / Тест":
        text(28,125,"测试集指标 / Метрики test · Macro",27)
        for col,key in enumerate(["Accuracy","Precision","Recall","F1","ROC-AUC"]):
            text(260+col*180,180,key,21)
        for i,name in enumerate(names):
            y = 225+i*56
            text(35,y,name,23,colors[i])
            for col,key in enumerate(["accuracy","precision","recall","f1","roc_auc"]):
                text(260+col*180,y,f"{report['results'][name]['test'][key]:.4f}",24,colors[i])
        text(28,345,"混淆矩阵 / Матрица ошибок · 行为真实，列为预测 / строки — истина",22)
        for col,name in enumerate(names):
            origin = 48+col*590
            matrix = np.array(report["results"][name]["confusion"])
            for digit in range(2):
                text(origin+150+digit*180,392,["猫 / Cat","狗 / Dog"][digit],17)
                text(origin,470+digit*135,["猫","狗"][digit],17)
            for row in range(2):
                for column in range(2):
                    value = int(matrix[row,column])
                    tone = value/max(1,matrix.max())
                    color = tuple(int(a+(b-a)*tone) for a,b in zip((27,41,56),(35,131,124)))
                    if row != column and value:
                        color = "#9e5149"
                    x,y = origin+110+column*180,424+row*135
                    draw.rectangle((x,y,x+170,y+125),fill=color)
                    text(x+55,y+38,str(value),32)
        text(28,822,"狗的识别表现 / Распознавание собак",25)
        for i,name in enumerate(names):
            per_class = report["results"][name]["test"].get("per_class")
            if per_class:
                text(35+i*590,873,f"{name} · n={per_class['support'][1]}",22,colors[i])
                text(35+i*590,914,f"Precision {per_class['precision'][1]:.3f}  "
                     f"Recall {per_class['recall'][1]:.3f}  F1 {per_class['f1'][1]:.3f}",21)
    return image


def gui():
    root = tk.Tk()
    root.title("猫狗分类实验 / Кошки и собаки")
    width = min(1280, root.winfo_screenwidth()-70)
    height = min(960, root.winfo_screenheight()-90)
    root.geometry(f"{width}x{height}")
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".",font=("Microsoft YaHei UI",12))
    style.configure("TFrame",background="#101924")
    style.configure("TLabel",foreground="#eaf3fa",background="#101924")
    style.configure("TButton",padding=(15,12),font=("Microsoft YaHei UI",13,"bold"))
    body = ttk.Frame(root,padding=14)
    body.pack(fill="both",expand=True)
    ttk.Label(body,text="猫狗分类与评价 / Кошки и собаки",
              font=("Microsoft YaHei UI",20,"bold")).pack(anchor="w")
    settings = ttk.Frame(body)
    settings.pack(fill="x",pady=8)
    kind = tk.StringVar(value="HOG")
    folds = tk.IntVar(value=3)
    split = tk.StringVar(value="60 / 20 / 20")
    balanced = tk.BooleanVar()
    c = tk.DoubleVar(value=1)
    trees = tk.IntVar(value=120)
    rare = tk.IntVar(value=100)
    controls = []
    for column,(label,var,options) in enumerate([
        ("特征 / Признаки",kind,["HOG","Pixels"]),
        ("交叉验证折数 / Фолды",folds,["3","5"]),
        ("Train / Validation / Test",split,["60 / 20 / 20","70 / 15 / 15"])]):
        field = ttk.Frame(settings)
        field.grid(row=0,column=column,padx=(0,14),sticky="w")
        ttk.Label(field,text=label).pack(anchor="w")
        combo = ttk.Combobox(field,textvariable=var,values=options,state="readonly",
                             width=22,font=("Microsoft YaHei UI",13))
        combo.pack(ipady=8)
        controls.append(combo)
    weight = tk.Checkbutton(settings,text="类别加权 / Веса классов",variable=balanced,
        indicatoron=False,font=("Microsoft YaHei UI",13,"bold"),padx=16,pady=12,
        bg="#26394b",fg="white",selectcolor="#247f74",activebackground="#36516a",
        activeforeground="white")
    weight.grid(row=0,column=3,padx=6)
    controls.append(weight)
    sliders = ttk.Frame(body)
    sliders.pack(fill="x")
    for col,(label,var,low,high,step) in enumerate([
        ("SVM log10(C)",c,-2,2,.25),
        ("树的数量 / Число деревьев",trees,20,300,20),
        ("训练狗保留比例 / Доля собак (%)",rare,10,100,10)]):
        slider = tk.Scale(sliders,label=label,variable=var,from_=low,to=high,
            resolution=step,orient="horizontal",length=300,width=20,sliderlength=38,
            font=("Microsoft YaHei UI",12),bg="#101924",fg="#eaf3fa",
            troughcolor="#36516a",activebackground="#74ddcf",highlightthickness=0)
        slider.grid(row=0,column=col,padx=8,sticky="ew")
        sliders.columnconfigure(col,weight=1)
        controls.append(slider)
    status = ttk.Label(body,text="调参后点击训练 / После настройки нажмите «Обучить»")
    status.pack(anchor="w",pady=6)
    actions = ttk.Frame(body)
    actions.pack(fill="x",pady=6)
    navigation = ttk.Frame(body)
    navigation.pack(fill="x",pady=6)
    page = tk.StringVar(value="数据概览 / Данные")
    canvas = tk.Canvas(body,bg="#101924",highlightthickness=0)
    scrollbar = ttk.Scrollbar(body,orient="vertical",command=canvas.yview)
    scrollbar.pack(side="right",fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(fill="both",expand=True,pady=5)
    executor = ThreadPoolExecutor(max_workers=1)
    future = None
    report = None
    def show():
        if report is None:
            canvas.delete("all")
            canvas.create_text(30,40,anchor="nw",text="请先训练 / Сначала обучите модели",
                               fill="white",font=("Microsoft YaHei UI",18))
            return
        preview = dashboard(report,page.get())
        preview.thumbnail((max(500,width-70),2000))
        canvas.image = ImageTk.PhotoImage(preview)
        canvas.delete("all")
        canvas.create_image(0,0,image=canvas.image,anchor="nw")
        canvas.configure(scrollregion=(0,0,*preview.size))
        canvas.yview_moveto(0)

    for column,title in enumerate(["数据概览 / Данные","训练与验证 / Обучение","测试评价 / Тест"]):
        tk.Radiobutton(navigation,text=title,variable=page,value=title,
            indicatoron=False,font=("Microsoft YaHei UI",12,"bold"),padx=12,pady=12,
            bg="#26394b",fg="white",selectcolor="#247f74",
            activebackground="#36516a",activeforeground="white",command=show).grid(
                row=column//3,column=column%3,sticky="ew",padx=3,pady=3)
        navigation.columnconfigure(column%3,weight=1)

    def poll():
        nonlocal report
        if not future.done():
            root.after(100,poll)
            return
        train.configure(state="normal")
        for widget in controls:
            widget.configure(state="readonly" if isinstance(widget,ttk.Combobox) else "normal")
        try:
            report = future.result()
            save_report(report,BASE/"results")
            status.configure(text="已训练，图表显示本次参数 / Графики соответствуют последнему обучению")
            page.set("训练与验证 / Обучение")
            show()
        except Exception as exc:
            messagebox.showerror("错误 / Ошибка",str(exc))

    def start():
        nonlocal future
        train.configure(state="disabled")
        for widget in controls:
            widget.configure(state="disabled")
        status.configure(text="正在训练与交叉验证 / Обучение и кросс-валидация…")
        fraction = int(split.get().split("/")[0])/100
        future = executor.submit(experiment,kind.get(),10**c.get(),trees.get(),
                                 balanced.get(),folds.get(),fraction,rare.get())
        root.after(100,poll)

    def reset():
        if future is not None and not future.done():
            return
        kind.set("HOG")
        c.set(1)
        trees.set(120)
        balanced.set(False)
        rare.set(100)
        folds.set(3)
        split.set("60 / 20 / 20")
        status.configure(text="参数已重置，点击训练应用 / Параметры сброшены, нажмите «Обучить»")

    train = ttk.Button(actions,text="训练并比较两模型 / Обучить обе модели",command=start)
    train.pack(side="left",padx=(0,10))
    ttk.Button(actions,text="恢复默认参数 / Сброс",command=reset).pack(side="left")

    def predict_photo():
        stored=BASE/"results"/"models.joblib"
        if not stored.exists():
            messagebox.showinfo("模型 / Модель","请先训练 / Сначала обучите")
            return
        filename=filedialog.askopenfilename(filetypes=[("Images","*.jpg *.jpeg *.png *.bmp")])
        if not filename:
            return
        try:
            bundle=joblib.load(stored)
            with Image.open(filename) as photo:
                preview=ImageOps.exif_transpose(photo).convert("RGB")
                gray=np.asarray(preview.convert("L").resize((64,64)),dtype=float)*16/255
            vector=features(np.array([gray]),bundle["feature"])
            window=tk.Toplevel(root)
            window.title("猫狗识别 / Прогноз")
            preview.thumbnail((500,350))
            panel=ttk.Label(window)
            panel.image=ImageTk.PhotoImage(preview)
            panel.configure(image=panel.image)
            panel.pack(padx=16,pady=16)
            for name,model in bundle["models"].items():
                label=["猫 / Кошка","狗 / Собака"][int(model.predict(vector)[0])]
                ttk.Label(window,text=f"{name} → {label}",font=("Microsoft YaHei UI",18)).pack(pady=10)
        except Exception as exc:
            messagebox.showerror("错误 / Ошибка",str(exc))

    ttk.Button(actions,text="识别照片 / Открыть фото",command=predict_photo).pack(side="left",padx=10)
    saved = BASE/"results"/"metrics.json"
    if saved.exists():
        try:
            report = json.loads(saved.read_text(encoding="utf-8"))
            status.configure(text="已载入上次结果，修改参数后请重新训练 / Загружен предыдущий результат")
        except (ValueError,OSError):
            pass
    show()
    root.mainloop()
    executor.shutdown(wait=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    if args.export:
        report = experiment()
        save_report(report, BASE / "results")
        print(json.dumps({name: value["test"] for name,value in report["results"].items()},indent=2))
    else:
        gui()


