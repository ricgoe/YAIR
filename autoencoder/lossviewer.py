import matplotlib.pyplot as plt
from pathlib import Path

root = Path('autoencoder/res/cleaned_res')


for i in range(1,9):
    with open(root / f'{i}.txt', 'r') as file:
        lines = file.readlines()
    numbers = [float(line.strip()) for line in lines if line.strip()]
    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_title(f'Loss Function Model ID#{i}')
    ax.set_ylabel('Loss')
    ax.plot(numbers)
    plt.show()
