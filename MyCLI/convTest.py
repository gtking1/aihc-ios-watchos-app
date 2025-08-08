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

consolidatedi = torch.from_numpy(np.convolve(consolidatedi, np.ones(15).astype('float32')/15.0, mode='same')).to(torch.float32)

consolidatedi = consolidatedi.numpy()
#consolidatedi = consolidatedi.round()

swiftConsolidatedi = np.loadtxt('test_file.txt', delimiter=' ', dtype='float32')
swiftConsolidatedi = swiftConsolidatedi

def savetxt_compact(fname, x, fmt="%.6g", delimiter=','):
    with open(fname, 'w') as fh:
        for row in x:
            line = delimiter.join("0.0" if value == 0 else fmt % value for value in row)
            fh.write(line + '\n')

savetxt_compact('consolidatedi.txt', torch.tensor(consolidatedi).unsqueeze(0), fmt='%1.8f', delimiter=' ')

# print(np.all(consolidatedi == swiftConsolidatedi))
# print(np.sum(consolidatedi == swiftConsolidatedi))# / consolidatedi.size)
# print(np.where(consolidatedi != swiftConsolidatedi))
# print(consolidatedi[463:476])
# print(swiftConsolidatedi[463:476])

# print(f"Are Python outputs close? {torch.allclose(torch.tensor(consolidatedi), torch.tensor(swiftConsolidatedi), atol=1e-5, rtol=1e-4)}")
# # Adjust atol (absolute tolerance) and rtol (relative tolerance) as needed
# # You can also look at the mean absolute error (MAE) or max absolute difference
# print(f"Max absolute difference: {torch.max(torch.abs(torch.tensor(consolidatedi) - torch.tensor(swiftConsolidatedi)))}")

#print("{:.8f}".format(consolidatedi[469]), "{:.8f}".format(swiftConsolidatedi[469]))

testArr = np.ones((32, 6, 256))

for i in range(32):
    for j in range(6):
        testArr[i][j] = np.convolve(testArr[i][j], np.ones(15).astype('float32')/15.0, mode='same')

testArr = testArr.astype('float32')

print(swiftConsolidatedi.shape)
swiftConsolidatedi = np.loadtxt('test_file.txt', delimiter=' ', dtype='float32')
swiftConsolidatedi = swiftConsolidatedi.reshape((32, 6, 256))

print(f"Are Python outputs close? {torch.allclose(torch.tensor(testArr), torch.tensor(swiftConsolidatedi), atol=1e-5, rtol=1e-4)}")
# Adjust atol (absolute tolerance) and rtol (relative tolerance) as needed
# You can also look at the mean absolute error (MAE) or max absolute difference
print(f"Max absolute difference: {torch.max(torch.abs(torch.tensor(testArr) - torch.tensor(swiftConsolidatedi)))}")