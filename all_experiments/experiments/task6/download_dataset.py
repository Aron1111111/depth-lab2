from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import shutil

base = Path(__file__).resolve().parent
data = base / "dataset"
data.mkdir(exist_ok=True)
archive = data / "cats_and_dogs_filtered.zip"
if not archive.exists():
    temporary = archive.with_suffix(".part")
    with urlopen("https://storage.googleapis.com/tensorflow-1-public/course2/cats_and_dogs_filtered.zip", timeout=60) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target)
    temporary.replace(archive)
with ZipFile(archive) as package:
    for item in package.infolist():
        path = (data/item.filename).resolve()
        if not path.is_relative_to(data.resolve()):
            raise ValueError("Invalid archive path")
    package.extractall(data)
print("Downloaded and extracted", archive.stat().st_size, "bytes")


