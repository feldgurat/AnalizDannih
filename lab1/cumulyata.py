import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
y = [0,
0.02,
0.05,
0.17,
0.36,
0.65,
0.83,
0.95,
1]
x = [4.98,
5.09,
5.20,
5.31,
5.42,
5.53,
5.64,
5.75,
5.86]
labels = list(map(str, y))
plt.bar(x, y, linewidth = 0.8, width = 0.4)
plt.plot(x, y, color='red', marker='s',markersize=7)
plt.xlim(4.75, 6)
plt.ylim(0, 1.2)
plt.xlabel('Число промахов $x_i$', fontsize = 12)
plt.ylabel('Накопленные \n относительные частоты', fontsize = 12)
plt.title('Кумулята \n накопленных относительных частот', fontsize = 14)
for i in range(9):
    plt.text(x[i] - 0.3, y[i] + 0.05,labels[i])
plt.tight_layout()
plt.show()