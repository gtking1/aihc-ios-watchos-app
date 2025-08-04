import numpy as np
import tkinter as tk
import tkinter.filedialog
import torch
from torch import nn
import random
import torch.nn.functional as F
import coremltools as ct

consolidatedi = torch.zeros(2240)
consolidatedi[470:530] = 1.0
consolidatedi[1470:1530] = 1.0

swiftConsolidatedi = np.loadtxt('test_file.txt', delimiter=' ')
consolidatedi = torch.from_numpy(np.convolve(consolidatedi, np.ones(100)/100.0, mode='same')).to(torch.float32)
print(consolidatedi)

consolidatedi = consolidatedi.numpy()
consolidatedi = consolidatedi.round()

def savetxt_compact(fname, x, fmt="%.6g", delimiter=','):
    with open(fname, 'w') as fh:
        for row in x:
            line = delimiter.join("0.0" if value == 0 else fmt % value for value in row)
            fh.write(line + '\n')

savetxt_compact('consolidatedi.txt', torch.tensor(consolidatedi).unsqueeze(0), fmt='%1.8f', delimiter=' ')

print(np.all(consolidatedi == swiftConsolidatedi))
print(np.sum(consolidatedi == swiftConsolidatedi) / consolidatedi.size)
print(np.where(consolidatedi != swiftConsolidatedi))

#print("{:.8f}".format(consolidatedi[469]), "{:.8f}".format(swiftConsolidatedi[469]))