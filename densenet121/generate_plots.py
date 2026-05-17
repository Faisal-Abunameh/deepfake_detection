import matplotlib.pyplot as plt
import numpy as np

# Data from log
epochs = [1, 2]
train_loss = [0.0698, 0.0399]
val_loss = [0.0719, 0.0606]
train_acc = [97.26, 98.43]
val_acc = [97.07, 97.77]

# Confusion Matrix
TN = 5289
FP = 202
FN = 705
TP = 4708
cm = np.array([[TN, FP], [FN, TP]])

# Plot Loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png', dpi=300)
plt.close()

# Plot Accuracy
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, label='Train Acc')
plt.plot(epochs, val_acc, label='Val Acc')
plt.title('Model Accuracy Over Epochs')
plt.ylabel('Accuracy (%)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_curve.png', dpi=300)
plt.close()

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(cm, cmap=plt.cm.Blues)
fig.colorbar(cax)

for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, f'{z}', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

plt.title('Confusion Matrix', pad=20)
plt.ylabel('Actual')
plt.xlabel('Predicted')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Fake (0)', 'Real (1)'])
ax.set_yticklabels(['Fake (0)', 'Real (1)'])
ax.xaxis.set_ticks_position('bottom')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()

# 1. Test Set Class Distribution (Pie Chart)
actual_fake_count = TN + FP
actual_real_count = FN + TP
plt.figure(figsize=(6, 6))
plt.pie([actual_fake_count, actual_real_count], labels=['Fake', 'Real'], autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'], startangle=140)
plt.title('Test Set Class Distribution (Actual)')
plt.savefig('test_distribution_pie.png', dpi=300)
plt.close()

# 2. Model Predictions Summary (Bar Chart)
pred_fake_count = TN + FN
pred_real_count = FP + TP
plt.figure(figsize=(7, 5))
plt.bar(['Predicted Fake', 'Predicted Real'], [pred_fake_count, pred_real_count], color=['lightcoral', 'lightskyblue'])
plt.title('Model Predictions Summary')
plt.ylabel('Number of Images')
for i, v in enumerate([pred_fake_count, pred_real_count]):
    plt.text(i, v + (max(pred_fake_count, pred_real_count)*0.01), str(v), ha='center', va='bottom')
plt.savefig('predictions_bar_chart.png', dpi=300)
plt.close()

print("Charts generated and saved successfully.")
