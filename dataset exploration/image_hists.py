import os
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('dataset exploration/image_info/w_h_data.csv')
print(df.head(5))

print('Minimun height: '+str(df['height'].argmin()))
print('Minimun width: '+str(df['width'].argmin()))


mask = df['height']<=32
for m in df['path'][df[mask].index].to_dict().values():
    print(m)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,9))
ax1.hist(df['width'], bins=100)
ax1.set_title('Histogram of widths')
ax2.hist(df['height'], bins= 100)
ax2.set_title('Histogram of heights')


plt.show()