import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter
import numpy as np
import sys

sys.path.append('/home/aida/kangning/pFedHN/experiments/pfedhn')

file_name_without_pc = '/home/aida/kangning/pFedHN/experiments/pfedhn/pfedhn_hetro_res/results_50_inner_steps_seed_42.json'
file_name_with_pc = '/home/aida/kangning/pFedHN/experiments/pfedhn_pc/pfedhn_pc_cifar_res/results.json'

with open(file_name_without_pc) as f:
    data_dict_without_pc = json.load(f)

with open(file_name_with_pc) as f:
    data_dict_with_pc = json.load(f)

test_avg_loss_without_pc = data_dict_without_pc["test_avg_loss"] # 168
test_avg_acc_without_pc = data_dict_without_pc["test_avg_acc"]
test_avg_loss_with_pc = data_dict_with_pc["test_avg_loss"] # 168
test_avg_acc_with_pc = data_dict_with_pc["test_avg_acc"]

val_avg_loss_without_pc = data_dict_without_pc["val_avg_loss"]
val_avg_acc_without_pc = data_dict_without_pc["val_avg_acc"]
val_avg_loss_with_pc = data_dict_with_pc["val_avg_loss"]
val_avg_acc_with_pc = data_dict_with_pc["val_avg_acc"]

x = range(0, 5040, 30)
plt.figure()
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('iters')    
plt.ylabel('loss')     

plt.plot(x, val_avg_loss_without_pc, linewidth=1, linestyle="solid", label="val_avg_loss_without_pc", color='red')
plt.plot(x, test_avg_loss_without_pc, linewidth=1, linestyle="solid", label="test_avg_loss_without_pc", color='blue')
plt.legend()
plt.title('Loss_without_pc curve')
plt.savefig('./loss_without_pc_curve.png')
plt.show()

plt.figure()
x = range(0, 5040, 30)
plt.figure()
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('iters')    
plt.ylabel('accuracy')     

plt.plot(x, val_avg_acc_without_pc, linewidth=1, linestyle="solid", label="val_avg_acc_without_pc", color='red')
plt.plot(x, test_avg_acc_without_pc, linewidth=1, linestyle="solid", label="test_avg_acc_without_pc", color='blue')
plt.legend()
plt.title('Accuracy_without_pc curve')
plt.savefig('./accuracy_without_pc_curve.png')
plt.show()

x = range(0, 5040, 30)
plt.figure()
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('iters')    
plt.ylabel('loss')     

plt.plot(x, val_avg_loss_with_pc, linewidth=1, linestyle="solid", label="val_avg_loss_with_pc", color='red')
plt.plot(x, test_avg_loss_with_pc, linewidth=1, linestyle="solid", label="test_avg_loss_with_pc", color='blue')
plt.legend()
plt.title('Loss_with_pc curve')
plt.savefig('./loss_with_pc_curve.png')
plt.show()

plt.figure()
x = range(0, 5040, 30)
plt.figure()
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('iters')    
plt.ylabel('accuracy')     

plt.plot(x, val_avg_acc_with_pc, linewidth=1, linestyle="solid", label="val_avg_acc_with_pc", color='red')
plt.plot(x, test_avg_acc_with_pc, linewidth=1, linestyle="solid", label="test_avg_acc_with_pc", color='blue')
plt.legend()
plt.title('Accuracy_with_pc curve')
plt.savefig('./accuracy_with_pc_curve.png')
plt.show()

plt.figure()
x = range(0, 5040, 30)
plt.figure()
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('iters')    
plt.ylabel('accuracy')     

plt.plot(x, test_avg_acc_without_pc, linewidth=1, linestyle="solid", label="test_avg_acc_without_pc", color='red')
plt.plot(x, test_avg_acc_with_pc, linewidth=1, linestyle="solid", label="test_avg_acc_with_pc", color='blue')
plt.legend()
plt.title('Accuracy_Comparation curve')
plt.savefig('./accuracy_comparation_curve.png')
plt.show()

x = range(0, 5040, 30)
plt.figure()
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('iters')    
plt.ylabel('loss')     

plt.plot(x, test_avg_loss_without_pc, linewidth=1, linestyle="solid", label="test_avg_loss_without_pc", color='red')
plt.plot(x, test_avg_loss_with_pc, linewidth=1, linestyle="solid", label="test_avg_loss_with_pc", color='blue')
plt.legend()
plt.title('Loss_Comparation curve')
plt.savefig('./loss_Comparation_curve.png')
plt.show()