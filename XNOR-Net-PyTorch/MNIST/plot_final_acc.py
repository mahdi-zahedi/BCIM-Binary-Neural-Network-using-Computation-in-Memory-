import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
barWidth = 0.3
 
F1=[0, 3.06, 2.1, 5.16, 13, 4.01, 16, 5]
F2=[0, 2.43, 1.33, 3.22, 11, 4, 4.9, 4.4]	
 

r1 = np.arange(len(F1))
r2 = [x + barWidth for x in r1]

fig, ax = plt.subplots()
fig.set_size_inches(6, 4)
plt.bar(r1, F1, width = barWidth, color = 'navy', edgecolor = 'black', capsize=7, label='F1')
 
plt.bar(r2, F2, width = barWidth, color = 'brown', edgecolor = 'black', capsize=7, label='F2')
 
plt.xticks([r + barWidth for r in range(len(F1))], ['LeNet5_1Ref', 'MLP-S_1Ref', 'MLP-M_1Ref', 'MLP-L_1Ref', 'CNN1_1Ref', 'CNN1_3Ref', 'CNN2_1Ref', 'CNN2_3Ref'])
plt.ylabel('Accuracy reduction',fontsize=15)
plt.xticks(rotation=45)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
#ylim(0, 400)
plt.tight_layout() 
plt.legend()
 
plt.savefig("final_acc.pdf")
