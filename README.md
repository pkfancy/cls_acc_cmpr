# cls_acc_cmpr
Comparison and boundary fitting for accuracies of various torchvision classification models

![Figure_1.png]

![Figure_2.png]


```python
plt.figure()
for k in dir(torchvision.models):
    if "Weights" in k and k not in ["Weights", "WeightsEnum"]:
        meta = eval(f"torchvision.models.{k}.DEFAULT.meta")
        acc = float(meta["_metrics"]["ImageNet-1K"]["acc@1"])
        size = float(meta["_file_size"])
        plt.semilogx(size, 1 / (1 - acc / 100), "+")
        plt.annotate(k[:-8], (size, 1 / (1 - acc / 100)), (4, 4), fontsize = 6, textcoords="offset points")
        # plt.semilogx(size, acc / 100, "+")
        # plt.annotate(k[:-8], (size, acc / 100), (4, 4), fontsize = 6, textcoords="offset points")
        if "MT" in k:
            print(k)

x = 10 ** np.linspace(0.5, 3.5, 100)
y = 1 + 1.1 * np.log(x)
plt.semilogx(x, y, "--", 
                label = r"$1 + 1.1\log \mathrm{size}$"
                )
# y = 1 - 1 / (1 + 1.1 * np.log(x))
# plt.semilogx(x, y, "--", 
#              label = r"$1 - \frac{1}{1 + 1.1\log \mathrm{size}}$"
#              )

plt.grid(True)
plt.xlabel("size / MB")
plt.ylabel("1 / (1 - acc)")
# plt.ylabel("acc")
plt.legend()
plt.title("torch models cls accs on imagenet")

plt.show()

```
