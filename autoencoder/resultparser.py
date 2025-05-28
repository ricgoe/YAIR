import re
from pathlib import Path

root = Path('autoencoder/res')
savepath = root / 'cleaned_res'
savepath.mkdir(exist_ok=True)

for file in root.iterdir():
    if file.suffix == '.out':
        filename = str(file.stem).split('_')[1]+'.txt'
        with open(savepath / filename, 'w') as rf:
            with open(file, 'r') as f:
                for line in f.readlines():
                    res = re.findall(r"\d+(?:\.\d+)?",line)
                    if res and len(res[0]) == 3:
                            rf.write(str(res)[-8:-2]+'\n')
                