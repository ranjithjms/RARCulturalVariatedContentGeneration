import os

import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists("..\\Result\\"):
    os.makedirs("..\\Result\\")

def APRS():
    ProposedPCNN = [99.67,	98.62,	98.39,	97.49]
    ExistingCNN = [97.82,	95.11,	94.65,	94.29]
    ExistingGRU = [94.21,	93.28,	92.11,	90.67]
    ExistingLSTM = [91.02,	89.45,	89.01,	88.73]
    ExistingRNN = [89.38,	87.54,	86.72,	86.07]
    barWidth = 0.15
    br1 = np.arange(len(ProposedPCNN))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    plt.figure(figsize=(8, 7))
    plt.bar(br1, ProposedPCNN, color='#135D66', hatch='\\\\', width=barWidth, edgecolor='#458B74', label='Proposed DAM (TL)$^2$C2S-CGRU')
    plt.bar(br2, ExistingCNN, color='#EB3678', hatch='\\\\', width=barWidth, edgecolor='#458B74', label='Exisitng CNN')
    plt.bar(br3, ExistingGRU, color='#FB773C', hatch='\\\\', width=barWidth, edgecolor='#458B74', label='Exisitng GRU')
    plt.bar(br4, ExistingLSTM, color='#A367B1', hatch='\\\\', width=barWidth, edgecolor='#458B74', label='Existing LSTM')
    plt.bar(br5, ExistingRNN, color='#FFD1E3', hatch='\\\\', width=barWidth, edgecolor='#458B74', label='Existing RNN')
    plt.xlabel('Metrics', fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel('Values (%)', fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.xticks([0.35, 1.35, 2.35, 3.35], ['Content Generation \nAccuracy', 'Precision','Recall', 'Specificity', ])
    plt.yticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.xticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.ylim(0,130)
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.legend(loc=2, ncol=2)
    plt.savefig("..\\Result\\APRS.png")
    plt.close()
APRS()

def BLU_CIDEr_ROGUE():
    plt.figure(figsize=(8, 6))
    Iteration = ['BLEU', 'CIDEr', 'ROUGE']
    ProposedTMBWO = [0.942,	0.95,	0.9634]
    ExistingBWO = [0.901,	0.91,   0.9165]
    ProposedGWO = [0.86,	0.86,   0.8832]
    ExistingPSO = [0.792,	0.83,   0.8191]
    ProposedSSO = [0.721,	0.76,   0.7368]
    plt.plot(Iteration, ProposedTMBWO, 'H-.r', markerfacecolor = 'lime')
    plt.plot(Iteration, ExistingBWO, 'h-.b', markerfacecolor = 'yellow')
    plt.plot(Iteration, ProposedGWO, 'p-.m', markerfacecolor = 'red')
    plt.plot(Iteration, ExistingPSO, 'd-.g', markerfacecolor = 'orange')
    plt.plot(Iteration, ProposedSSO, '^-.c', markerfacecolor = 'magenta')
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.ylim(0.7, 1.05)
    plt.xlabel("Metrics", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel("Values ", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.legend(['Proposed DAM (TL)$^2$C2S-CGRU', 'Exisitng CNN', 'Exisitng GRU', 'Existing LSTM', 'Existing RNN'], loc=2, ncol=2)
    plt.yticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.xticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.savefig("..\\Result\\BLU_CIDEr_ROGUE.png")
    plt.close()
BLU_CIDEr_ROGUE()

def METEOR():
    courses = ['Proposed\n DAM (TL)$^2$C2S-CGRU', 'Exisitng \nCNN', 'Exisitng GRU', 'Existing LSTM', 'Existing RNN']
    values = [0.38, 0.312, 0.29, 0.278, 0.23]
    fig = plt.subplots(figsize=(9, 7))
    plt.bar(courses, values, color='violet', width=0.35 )
    plt.xlabel("Techniques", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel("Meteor", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.yticks(fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.xticks(fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.savefig("..\\Result\\METEOR.png")
    plt.close()
METEOR()

def RUL_FUZ_DEFUZ():
    plt.figure(figsize=(8, 6))
    Iteration = ['Rule Generation', 'Fuzzification', 'De-fuzzification']
    ProposedTMBWO = [1278,	1478,	1672]
    ExistingBWO = [1377,	1582,	1893]
    ProposedGWO = [1983,	2078,	2743]
    ExistingPSO = [2063,	2583,	3071]
    ProposedSSO = [2614,	2894,	3822]
    plt.plot(Iteration, ProposedTMBWO, 'H--r', markerfacecolor = 'lime')
    plt.plot(Iteration, ExistingBWO, 'h--b', markerfacecolor = 'yellow')
    plt.plot(Iteration, ProposedGWO, 'p--m', markerfacecolor = 'red')
    plt.plot(Iteration, ExistingPSO, 'd--g', markerfacecolor = 'orange')
    plt.plot(Iteration, ProposedSSO, '^--c', markerfacecolor = 'magenta')
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.ylim(1200, 4000)
    plt.xlabel("Metrics", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel("Time (ms) ", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.legend(['Proposed SCV-Fuzzy', 'Sigmoid Fuzzy', 'Trapezoidal Fuzzy', 'Singleton Fuzzy', 'Triangular Fuzzy'], loc=2, ncol=2)
    plt.yticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.xticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.savefig("..\\Result\\RUL_FUZ_DEFUZ.png")
    plt.close()
RUL_FUZ_DEFUZ()

def MAP_F1():
    ProposedPCNN = [99,	98.34]
    ExistingCNN = [94,	94.21]
    ExistingGRU = [89,	90.01]
    ExistingLSTM = [83,	87.51]
    ExistingRNN = [81,	83.91]
    barWidth = 0.15
    br1 = np.arange(len(ProposedPCNN))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    plt.figure(figsize=(8, 6))
    plt.bar(br1, ProposedPCNN, color='#D6BD98', hatch='o', width=barWidth, edgecolor='#458B74', label='Proposed EKT-RN')
    plt.bar(br2, ExistingCNN, color='#FFC55A', hatch='o', width=barWidth, edgecolor='#458B74', label='Existing RN')
    plt.bar(br3, ExistingGRU, color='#6EACDA', hatch='o', width=barWidth, edgecolor='#458B74', label='Existing YOLO')
    plt.bar(br4, ExistingLSTM, color='#E2E2B6', hatch='o', width=barWidth, edgecolor='#458B74', label='Existing VJ')
    plt.bar(br5, ExistingRNN, color='#E3A5C7', hatch='o', width=barWidth, edgecolor='#458B74', label='Existing SSD')
    plt.xlabel('Metrics', fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel('Values (%)', fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.xticks([0.35, 1.35], ['Mean Average Precision ', 'F1-Score '])
    plt.yticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.xticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.ylim(0,130)
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.legend(loc=1, ncol=2)
    plt.savefig("..\\Result\\MAP_F1.png")
    plt.close()
MAP_F1()







def MSE_RMSE():
    plt.figure(figsize=(8, 6))
    Iteration = ['MSE', 'RMSE']
    ProposedTMBWO = [0.0821,	0.1637]
    ExistingBWO = [0.2763,	1.4217]
    ProposedGWO = [0.7812,	1.9325]
    ExistingPSO = [1.6392,	2.6328]
    ProposedSSO = [1.9358,	2.8915]
    plt.plot(Iteration, ProposedTMBWO, 'H-r', markerfacecolor = 'lime')
    plt.plot(Iteration, ExistingBWO, 'h-b', markerfacecolor = 'yellow')
    plt.plot(Iteration, ProposedGWO, 'p-m', markerfacecolor = 'red')
    plt.plot(Iteration, ExistingPSO, 'd-g', markerfacecolor = 'orange')
    plt.plot(Iteration, ProposedSSO, '^-c', markerfacecolor = 'magenta')
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.ylim(0, 3)
    plt.xlabel("Metrics", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel("Errors ", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.legend(['Proposed CRC-MF', 'Existing MF', 'Existing GF', 'Existing BF', 'Existing LPF'], loc=2, ncol=2)
    plt.yticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.xticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.savefig("MSE_RMSE.png")
    plt.close()
MSE_RMSE()