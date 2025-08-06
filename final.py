import numpy as np
import tkinter as tk
import tkinter.filedialog

def moving_average(data, window_size=15):
    X_smooth = np.zeros(data.shape)
    for i,channel in enumerate(data):
        X_smooth[i] = np.convolve(channel, np.ones(window_size)/window_size, mode='same')
    return torch.from_numpy(X_smooth).to(torch.float32)

#tk.filedialog.askopenfilename()
# motionData = np.loadtxt('inputFiles/motionDataFix.txt', delimiter=' ')
# #motionDataTest = motionData.reshape((32, 6, 256))
# motionDataTest = motionData.reshape((32, 256, 6)).transpose((0, 2, 1))
# motionDataFix = motionData.reshape((32, 6, 256))

# for i, data in enumerate(motionData):
#     # if i >= 6:
#     #     break

#     if (i + 1) % 6 in (1, 2, 3):
#         if not (-2 <= motionData[i] <= 2):
#             print(f"Acceleration at {i} out of bounds")
    
#     if (i + 1) % 6 in (4, 5, 0):
#         if not (-250 <= motionData[i] <= 250):
#             print(f"Gyroscope at {i} out of bounds")

# for i, data in enumerate(motionDataFix):
#     # if i >= 6:
#     #   break

#     if (i + 1) % 256 in (1, 2, 3):
#         if not (-2 <= motionData[i] <= 2):
#             print(f"Fix: Acceleration at {i} out of bounds")
    
#     if (i + 1) % 256 in (4, 5, 0):
#         if not (-250 <= motionData[i] <= 250):
#             print(f"Fix: Gyroscope at {i} out of bounds")

#B = np.loadtxt(tk.filedialog.askopenfilename(), delimiter=' ')

#print(np.array_equal(A, B))

# motionDataReshaped = []
# window = []
# accelX = []
# accelY = []
# accelZ = []
# gyroX = []
# gyroY = []
# gyroZ = []

# for i, data in enumerate(motionData):
#     match ((i + 1) % 6):
#         case 1:
#             accelX.append(data)
#         case 2:
#             accelY.append(data)
#         case 3:
#             accelZ.append(data)
#         case 4:
#             gyroX.append(data)
#         case 5:
#             gyroY.append(data)
#         case 0:
#             gyroZ.append(data)
    
#     if len(gyroZ) == 256:
#         window = [accelX, accelY, accelZ, gyroX, gyroY, gyroZ]
#         motionDataReshaped.append(window)
#         accelX = []
#         accelY = []
#         accelZ = []
#         gyroX = []
#         gyroY = []
#         gyroZ = []

# motionDataReshaped = np.asarray(motionDataReshaped)

# print(np.sum(motionDataReshaped == motionDataTest) / motionDataReshaped.size)

# slice1 = motionDataReshaped[0, :, 64:]

# print(slice1.size)

# slice2 = motionDataReshaped[1, :, :192]

# print(slice2.size)

# print(np.all(np.equal(slice1, slice2)))

# for i in range(31):
#     slice1 = motionDataReshaped[i, :, 64:]
#     #slice1 = motionDataTest[i, :, 64:]

#     slice2 = motionDataReshaped[i + 1, :, :192]
#     #slice2 = motionDataTest[i + 1, :, :192]

#     if not np.all(np.equal(slice1, slice2)):
#         print("Sliding error")

# for i in range(31):
#     slice1 = motionDataFix[i, :, 64:]
#     #slice1 = motionDataTest[i, :, 64:]

#     slice2 = motionDataFix[i + 1, :, :192]
#     #slice2 = motionDataTest[i + 1, :, :192]

#     if not np.all(np.equal(slice1, slice2)):
#         print("Sliding error")

# motionData = motionData.reshape((32, 6, 256))

# print(np.equal(motionDataReshaped, motionData))

# motionData = np.reshape(motionData, ((32, 6, 256)))

# motionDataFirstRows = motionDataReshaped[0, :, :]

# motionDataAX = motionDataFirstRows[0, :]
# motionDataAY = motionDataFirstRows[1, :]
# motionDataAZ = motionDataFirstRows[2, :]
# motionDataGX = motionDataFirstRows[3, :]
# motionDataGY = motionDataFirstRows[4, :]
# motionDataGZ = motionDataFirstRows[5, :]

# print(motionDataAX.min(0), motionDataAX.max(0))
# print(motionDataAY.min(0), motionDataAY.max(0))
# print(motionDataAZ.min(0), motionDataAZ.max(0))
# print(motionDataGX.min(0), motionDataGX.max(0))
# print(motionDataGY.min(0), motionDataGY.max(0))
# print(motionDataGZ.min(0), motionDataGZ.max(0))

# test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
# test = np.array(test)
# test = np.reshape(test, (3, 2, 6))
# print(test)

import torch
from torch import nn
import random
import torch.nn.functional as F

in_channels = 6
winsize = 256
LSTM_stride = 64
n_LSTM_windows = 32
full_winsize = LSTM_stride * (n_LSTM_windows-1) + winsize

def savetxt_compact(fname, x, fmt="%.6g", delimiter=','):
    with open(fname, 'w') as fh:
        for row in x:
            line = delimiter.join("0.0" if value == 0 else fmt % value for value in row)
            fh.write(line + '\n')

def get_time_in_reps(preds, stride, winsize):
    # preds: N x T x C
    time_in_rep = []
    # consolidated = []
    end_reps = []
    for pred in preds: # each element in batch
        consolidatedi = []
        for i,predi in enumerate(pred): # each element in time
            upper = stride if i < len(pred) - 1 else len(predi)
            consolidatedi.append(predi[:upper])
        consolidatedi = torch.cat(consolidatedi, axis=0)
        consolidatedi = torch.from_numpy(np.convolve(consolidatedi, np.ones(100)/100, mode='same').round()).to(torch.float32)

        #savetxt_compact('consolidatediPython.txt', consolidatedi.unsqueeze(0), fmt='%1.8f', delimiter=' ')

        time_in_repi = torch.zeros_like(consolidatedi).to(torch.float32)
        end_repsi = []
        if consolidatedi.sum() > 0:
            diff = np.diff(consolidatedi)
            starts = np.where(diff > 0)[0]
            ends = np.where(diff < 0)[0]
            if len(starts) == 0 or len(ends) == 0:
                if len(starts) == 0:
                    # len(end) == 1 -> start is the beginning of session
                    starts = np.array([0])
                elif len(ends) == 0:
                    # len(start) == 1 -> end is the end of session
                    ends = np.array([len(pred[0])])
            else:
                if starts[0] > ends[0]:
                    # first end has no start -> first start is the beginning of session
                    starts = np.concatenate([[0], starts])
                if ends[-1] < starts[-1]:
                    # last start has no end -> last end is the end of session
                    ends = np.concatenate([ends, [len(pred[0])]])
            for start,end in zip(starts, ends):
                end_repsi.append((start + end) // 2)
            
            for i in range(1, len(end_repsi)):
                time_in_repi[end_repsi[i-1]:end_repsi[i]] = (end_repsi[i] - end_repsi[i-1]) / full_winsize
                # time_in_rep.append(end_repsi[i] - end_repsi[i-1])

        end_reps.append(end_repsi)
        # consolidated.append(consolidatedi)
        time_in_rep.append(time_in_repi)
    # preds = torch.stack(consolidated, axis=0)
    time_in_rep = torch.stack(time_in_rep, axis=0)

    # rewindow
    time_in_rep = time_in_rep.unfold(1, winsize, stride)

    print("END REPS:", end_reps)

    return time_in_rep

class ResBlock(nn.Module):
    # One layer of convolutional block with batchnorm, relu and dropout
    def __init__(
            self, in_channels, out_channels,
            kernel_size=3, stride=1, dropout=0.0,
        ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, 
                kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.skip = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, stride=stride
        ) if in_channels != out_channels or stride > 1 else nn.Identity()
    def forward(self, x):
        return self.block(x) + self.skip(x)
    
class DepthBlock(nn.Module):
    # "depth" number of ConvBlocks with downsample on the first block
    def __init__(
            self, depth, in_channels, out_channels,
            kernel_size=3, downsample_stride=2, 
            dropout=0.0
    ):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ResBlock(
                in_channels=in_channels if i == 0 else out_channels, 
                out_channels=out_channels,
                kernel_size=kernel_size, 
                stride=downsample_stride if i == 0 else 1,
                dropout=dropout
            )
            for i in range(depth)
        ])
    def forward(self, x):
        return self.blocks(x)
 
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.width = config['width']
        self.depth = config['depth']
        self.stem_out_c = config['stem_out_c']
        self.stem_kernel = config['stem_kernel']
        self.dropout = config['dropout']

        if len(self.width) != len(self.depth):
            raise ValueError('Width and depth must have the same length')
        self.conv_out_channels = self.stem_out_c if len(self.width) == 0 else self.width[-1]

        self.encoder = nn.Sequential(
            # nn.Conv1d(in_channels, self.stem_out_c, kernel_size=self.stem_kernel, padding=self.stem_kernel // 2, stride=2),
            # nn.BatchNorm1d(self.stem_out_c),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            ResBlock(in_channels, self.stem_out_c, kernel_size=self.stem_kernel, stride=2),
            *[DepthBlock(
                depth=self.depth[i],
                in_channels=self.stem_out_c if i == 0 else self.width[i-1], 
                out_channels=self.width[i],
                dropout=self.dropout, 
            ) for i in range(len(self.width))]
        )
    def forward(self, x):
        return self.encoder(x)

class ConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.encoder = Encoder(config)
        self.ap = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(512, winsize // 2)
        self.fc = nn.Linear(self.encoder.conv_out_channels, winsize)
    def forward(self, x):
        emb = self.encoder(x)
        # x = self.conv(x)
        x = self.ap(emb).squeeze(-1)
        seg_logits = self.fc(x)
        seg_logits_processed = F.sigmoid(seg_logits).round()
        # x = torch.repeat_interleave(x, 2, dim=1)
        return emb, seg_logits_processed
    
    def freeze(self, stop_idx=None):
        if stop_idx is None:
            stop_idx = len(self.encoder.encoder)
        for block in self.encoder.encoder[:stop_idx]:
            for param in block.parameters():
                param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = False

class LSTMNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_channels = 6


        self.encoder = ConvNet(config)
        hidden_size = config['lstm_config']['hidden_size']
        # encoder_proj_channels = config['lstm_config']['encoder_proj_channels']
        skip_channels = config['lstm_config']['skip_channels']
        num_layers = config['lstm_config']['num_layers']
        dropout = config['lstm_config']['dropout']
        linear_hidden_size = config['lstm_config']['linear_hidden_size']

        # # Project encoder output to hidden size
        # self.conv_proj = nn.Conv1d(
        #     self.encoder.encoder.conv_out_channels, 
        #     encoder_proj_channels, 
        #     kernel_size=1
        # )

        # Convolution layer from input signal to skip_channels size
        self.conv_skip = nn.Sequential(
            nn.Conv1d(in_channels, skip_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(skip_channels),
            nn.ReLU(),
            # ResBlock(in_channels, skip_channels, kernel_size=7, stride=1),
        )
        self.ap = nn.AdaptiveAvgPool1d(1)

        # Project encoder, skip layer, and time_in_reps to lstm input size
        self.lstm_proj = nn.Linear(
            # encoder_proj_channels + skip_channels + winsize,
            self.encoder.encoder.conv_out_channels + skip_channels + winsize,
            hidden_size
        )

        # LSTM layer on projected encoder, skip layer, and time_in_reps
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout
        )

        # Linear layer to predict the output
        if linear_hidden_size == 0:
            self.fc = nn.Linear(hidden_size, 1)
        else:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, linear_hidden_size),
                nn.ReLU(),
                nn.Linear(linear_hidden_size, 1)
            )
    def forward(self, x):
        N, T, C, L = x.shape

        x = x.view(N*T, C, L)
        
        # Run encoder to get embeddings and segmentation logits
        x_seg, seg_logits = self.encoder(x)

        # x_seg = self.conv_proj(x_seg)
        x_seg = self.ap(x_seg).squeeze(-1)
        x_seg = x_seg.view(N, T, -1)

        # Project segmentation logits to seg_proj_size
        seg_logits = F.sigmoid(seg_logits).round()
        time_in_rep = get_time_in_reps(seg_logits.view(N, T, L).detach().cpu(), LSTM_stride, winsize).to(x.device)

        x_skip = self.conv_skip(x)
        x_skip = self.ap(x_skip).squeeze(-1)
        x_skip = x_skip.view(N, T, -1)

        x = torch.cat([x_seg, x_skip, time_in_rep], dim=2)
        x = self.lstm_proj(x)

        o, (h,c) = self.lstm(x)
        # x = self.fc(o[:, -1, :]) # predict for last time step
        x = self.fc(o)        # predict for all time steps
        return x
    
    def get_optimizer(self, lr, weight_decay=1e-4, betas=(0.9, 0.999)):
        # AdamW optimzer - apply weight decay to linear and conv weights
        # but not to biases and batchnorm layers
        params = self.named_parameters()
        decay_params = [p for n,p in params if p.dim() >= 2]
        no_decay_params = [p for n,p in params if p.dim() < 2]
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], betas=betas, lr=lr)
        return optimizer

class LSTMNetNew(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_channels = 6

        self.encoder = ConvNet(config)
        hidden_size = config['lstm_config']['hidden_size']
        skip_channels = config['lstm_config']['skip_channels']
        num_layers = config['lstm_config']['num_layers']
        dropout = config['lstm_config']['dropout']
        linear_hidden_size = config['lstm_config']['linear_hidden_size']

        self.conv_skip = nn.Sequential(
            nn.Conv1d(in_channels, skip_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(skip_channels),
            nn.ReLU(),
        )
        self.ap = nn.AdaptiveAvgPool1d(1)

        # IMPORTANT: Adjust the input size of lstm_proj.
        # It no longer gets winsize from seg_logits directly,
        # but from the pre-computed time_in_rep.
        # Assuming time_in_rep will be (N, T, winsize) after rewindowing.
        self.lstm_proj = nn.Linear(
            self.encoder.encoder.conv_out_channels + skip_channels + winsize, # time_in_rep's last dimension
            hidden_size
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout
        )

        if linear_hidden_size == 0:
            self.fc = nn.Linear(hidden_size, 1)
        else:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, linear_hidden_size),
                nn.ReLU(),
                nn.Linear(linear_hidden_size, 1)
            )
            
    # CRITICAL CHANGE: time_in_rep is now an input
    def forward(self, x, time_in_rep):
        N, T, C, L = x.shape

        x_flat = x.contiguous().view(N*T, C, L)
        
        x_seg, seg_logits = self.encoder(x_flat)

        x_seg = self.ap(x_seg).squeeze(-1)
        x_seg = x_seg.view(N, T, -1)

        # seg_logits are still computed but NOT used to derive time_in_rep inside forward
        # If seg_logits are not used elsewhere in the graph, consider removing them entirely
        # or ensuring they don't break strict export if their shape is dynamic and not used.
        # For now, we'll keep them as they are a return value from self.encoder
        _ = F.sigmoid(seg_logits).round() # Consume seg_logits if not used later

        # time_in_rep is now an external input to this forward pass
        # Ensure its shape matches (N, T, winsize) as expected by torch.cat

        x_skip = self.conv_skip(x_flat)
        x_skip = self.ap(x_skip).squeeze(-1)
        x_skip = x_skip.view(N, T, -1)

        # Ensure time_in_rep has the correct dimensions (N, T, winsize)
        # Assuming `time_in_rep` passed in here will be (N, T, winsize)
        x = torch.cat([x_seg, x_skip, time_in_rep], dim=2)
        x = self.lstm_proj(x)

        o, (h,c) = self.lstm(x)
        x = self.fc(o)        
        return x
    
    def get_optimizer(self, lr, weight_decay=1e-4, betas=(0.9, 0.999)):
        # AdamW optimzer - apply weight decay to linear and conv weights
        # but not to biases and batchnorm layers
        params = self.named_parameters()
        decay_params = [p for n,p in params if p.dim() >= 2]
        no_decay_params = [p for n,p in params if p.dim() < 2]
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], betas=betas, lr=lr)
        return optimizer

# weights, config = torch.load('best_model-3binary-77.pth', map_location=torch.device("cpu"), weights_only=True)
# model = LSTMNet(config)
# model.load_state_dict(weights)
# model.eval()

# modelNew = LSTMNetNew(config)
# modelNew.load_state_dict(weights)
# modelNew.eval()

# example_input_x = torch.tensor(motionDataFix).contiguous().to(torch.float32)

# temp_encoder_model = ConvNet(config) # Only need the encoder part
# temp_encoder_model.load_state_dict(model.encoder.state_dict()) # Load relevant weights
# temp_encoder_model.eval()

# T, C, L = example_input_x.shape
# x_flat_for_seg = example_input_x.contiguous().view(T, C, L)
# _, seg_logits_for_time_in_rep = temp_encoder_model(x_flat_for_seg)

evaluate_real_time_comparison = np.loadtxt('evaluate-real-time-comparison.txt')
print(evaluate_real_time_comparison.shape)
eval2 = np.loadtxt('eval2.txt')

dataset = []
# firstWindow = evaluate_real_time_comparison[:, :256]
# firstWindow = moving_average(firstWindow)
# dataset.append(firstWindow)
for i in range(32):
    start = (i * 64)
    end = start + 256
    print(start, end)
    window = evaluate_real_time_comparison[:, start:end]
    window = moving_average(window)
    dataset.append(window)
np_dataset = np.array(dataset)
print(np_dataset.shape)
stride = 64

eval2 = eval2.reshape((32, 256, 6)).transpose(0, 2, 1)

# evaluate_real_time_comparison = torch.tensor(evaluate_real_time_comparison).unfold(1, winsize, stride)
# evaluate_real_time_comparison = evaluate_real_time_comparison.reshape((32, 256, 6)).numpy().transpose((0, 2, 1))
# print(evaluate_real_time_comparison.shape)
# print(np.all(np_dataset == evaluate_real_time_comparison))
for i in range(6):
    print(np.max(np_dataset[0, i]))
    print(np.min(np_dataset[0, i]))

import coremltools as ct

mlmodel_convnet = ct.models.MLModel("convnet_model.mlpackage")

lstmInput = np.loadtxt("inputFiles/lstmInput2.txt", delimiter=' ')
lstmInput = lstmInput.reshape((32, 6, 256))

#convnet_input_dict = {"x": lstmInput}
convnet_input_dict = {"x": np_dataset}
convnet_output_dict = mlmodel_convnet.predict(convnet_input_dict)
coreml_seg_logits_np = convnet_output_dict["round_1"]
coreml_seg_logits_np = F.sigmoid(torch.tensor(coreml_seg_logits_np)).numpy().round()

# #print(f"Core ML ConvNet output shape (seg_logits): {coreml_seg_logits_np.shape}")

# encoderLogits = np.loadtxt('inputFiles/encoderLogitsFix.txt', delimiter=' ')
# encoderLogits = encoderLogits.reshape((32, 256))

# #print(np.sum(coreml_seg_logits_np == encoderLogits) / coreml_seg_logits_np.size)

# coreml_seg_logits_np = F.sigmoid(torch.tensor(coreml_seg_logits_np)).round()

# coreml_seg_logits_np = coreml_seg_logits_np.unsqueeze(0)

# time_in_rep = get_time_in_reps(coreml_seg_logits_np, LSTM_stride, winsize)

# time_in_rep = time_in_rep.numpy()

# savetxt_compact('timesInRepsPython.txt', time_in_rep.squeeze(0), fmt='%1.8f', delimiter=' ')
# #np.savetxt('timesInRepsCalc.txt', time_in_rep.squeeze(0), delimiter=' ', fmt='%1.8f')

# timesInReps = np.loadtxt('inputFiles/timesInRepsFix.txt', delimiter=' ')
# timesInReps = timesInReps.reshape(1, 32, 256)
# savetxt_compact('timesInRepsFixReformatted.txt', timesInReps.squeeze(0), fmt='%1.8f', delimiter=' ')

# comparison_time_in_rep = np.sum(np.isclose(time_in_rep, timesInReps, atol=1e-5, rtol=1e-4)) / time_in_rep.size
# max_diff_time_in_rep = np.max(np.abs(time_in_rep - timesInReps))

#print(f"Test 1: {comparison_time_in_rep}")
#print(f"Test 2: {max_diff_time_in_rep}")

#print(np.sum(time_in_rep == timesInReps) / time_in_rep.size)

#print('Inside function testing')
# def get_time_in_reps(preds, stride, winsize, override):
#     # preds: N x T x C
#     time_in_rep = []
#     # consolidated = []
#     end_reps = []
#     starts = None
#     ends = None
#     diff = None
#     startsInit = None
#     endsInit = None

#     for pred in preds: # each element in batch
#         consolidatedi = []
#         for i,predi in enumerate(pred): # each element in time
#             upper = stride if i < len(pred) - 1 else len(predi)
#             consolidatedi.append(predi[:upper])
#         consolidatedi = torch.cat(consolidatedi, axis=0)
#         noConv = consolidatedi
#         consolidatedi = torch.from_numpy(np.convolve(consolidatedi, np.ones(100)/100, mode='same').round()).to(torch.float32)

#         # savetxt_compact('consolidatediPython.txt', consolidatedi.unsqueeze(0), fmt='%1.8f', delimiter=' ')

#         consolidatedi = torch.tensor(override)

#         time_in_repi = torch.zeros_like(consolidatedi).to(torch.float32)
#         end_repsi = []
#         if consolidatedi.sum() > 0:
#             diff = np.diff(consolidatedi)
#             # savetxt_compact('diffSample.txt', torch.tensor(diff).unsqueeze(0), fmt='%1.1f', delimiter=' ')
#             starts = np.where(diff > 0)[0]
#             startsInit = starts
#             ends = np.where(diff < 0)[0]
#             endsInit = ends
#             if len(starts) == 0 or len(ends) == 0:
#                 if len(starts) == 0:
#                     # len(end) == 1 -> start is the beginning of session
#                     starts = np.array([0])
#                 elif len(ends) == 0:
#                     # len(start) == 1 -> end is the end of session
#                     ends = np.array([len(pred[0])])
#             else:
#                 if starts[0] > ends[0]:
#                     # first end has no start -> first start is the beginning of session
#                     starts = np.concatenate([[0], starts])
#                 if ends[-1] < starts[-1]:
#                     # last start has no end -> last end is the end of session
#                     ends = np.concatenate([ends, [len(pred[0])]])
#             for start,end in zip(starts, ends):
#                 end_repsi.append((start + end) // 2)
            
#             for i in range(1, len(end_repsi)):
#                 time_in_repi[end_repsi[i-1]:end_repsi[i]] = (end_repsi[i] - end_repsi[i-1]) / full_winsize
#                 # time_in_rep.append(end_repsi[i] - end_repsi[i-1])

#         end_reps.append(end_repsi)
#         # consolidated.append(consolidatedi)
#         time_in_rep.append(time_in_repi)
#         #print(np.array(time_in_repi).shape, "Numpy")
#     # preds = torch.stack(consolidated, axis=0)
#     time_in_rep = torch.stack(time_in_rep, axis=0)
#     tirReturn = time_in_rep

#     # rewindow
#     built = []

#     for i in range(32):
#         startInd = i * 64
#         window = tirReturn.numpy().squeeze()[startInd:(startInd + 256)]
#         built = np.concatenate((built, window), 0)
#     built = built.reshape(1, 32, 256)
#     time_in_rep = time_in_rep.unfold(1, winsize, stride)

#     #print("In function check", np.all(built == time_in_rep.numpy()))

#     return time_in_rep, consolidatedi, noConv, starts, ends, diff, startsInit, endsInit, tirReturn

# predsMultiArray = np.loadtxt('inputFiles/predsMultiArrayB.txt', delimiter=' ')
# noConvLoaded = np.loadtxt('inputFiles/noConv.txt', delimiter=' ')
# consolidatediRounded = np.loadtxt('inputFiles/consolidatediRoundedBob.txt', delimiter=' ')



# encoderLogits = np.loadtxt('inputFiles/tirInput2.txt', delimiter=' ')
# #encoderLogits = torch.tensor(encoderLogits).reshape((1, 32, 256))
# encoderLogits = torch.tensor(coreml_seg_logits_np).reshape((1, 32, 256))
time_in_rep = get_time_in_reps(torch.tensor(coreml_seg_logits_np).unsqueeze(0), LSTM_stride, winsize).numpy()

# time_in_repLoaded = np.loadtxt('inputFiles/tirOutput2.txt', delimiter=' ')
# time_in_repLoaded = time_in_repLoaded.reshape((1, 32, 256))

# comparison_tir = np.allclose(time_in_rep, time_in_repLoaded, atol=1e-5, rtol=1e-4)
# max_diff_tir = np.max(np.abs(time_in_rep - time_in_repLoaded))

# print(f"Test {comparison_tir}")
# print(f"Test {max_diff_tir}")


# timeinrepA = np.loadtxt('inputFiles/timeinrepA.txt', delimiter=' ')
# timeinrepA = timeinrepA.reshape((1, 32, 256))
# noConv = noConv.numpy()
# print(np.all(noConv == noConvLoaded))
# consolidatediPython = consolidatediPython.numpy()
# print(np.sum(consolidatediRounded == consolidatediPython))
# time_in_rep = time_in_rep.numpy()
# print(np.sum(time_in_rep == timeinrepA))
# startsB = np.loadtxt('inputFiles/startsF.txt', delimiter=' ')
# endsB = np.loadtxt('inputFiles/endsF.txt', delimiter=' ')
# savetxt_compact('startsBPython.txt', torch.tensor(starts).unsqueeze(0), fmt='%1.8f', delimiter=' ')
# savetxt_compact('endsBPython.txt', torch.tensor(ends).unsqueeze(0), fmt='%1.8f', delimiter=' ')
# diffLoaded = np.loadtxt('inputFiles/diffArray.txt', delimiter=' ')
# print("Diff", np.all(diff == diffLoaded))

#print("Starts check", np.all(startsB == starts))
#print(np.all(endsB == ends))

# savetxt_compact('timeinrepPython.txt', torch.tensor(time_in_rep).squeeze(0), fmt='%1.8f', delimiter=' ')

# print(tirReturn.shape)

# tirLoaded = np.loadtxt('inputFiles/timeinrepiY.txt', delimiter=' ')
# tirLoaded.reshape((1, 2240))

# tirReturn = tirReturn.numpy()

# print("TIR", np.all(tirReturn == tirLoaded))

# savetxt_compact('timeinrepiFPython.txt', tirReturn, fmt='%1.8f', delimiter=' ')

# time_in_rep_loaded = np.loadtxt('inputFiles/timeinrepsoutputBob.txt', delimiter=' ')
# time_in_rep_loaded = time_in_rep_loaded.reshape((1, 32, 256))

# comparison_lstm = np.allclose(time_in_rep, time_in_rep_loaded, atol=1e-5, rtol=1e-4)
# max_diff_lstm = np.max(np.abs(time_in_rep, time_in_rep_loaded))

#print(f"Are Core ML LSTMNet outputs close to PyTorch? {comparison_lstm}")
#print(f"Max absolute difference (Core ML LSTMNet vs PyTorch): {max_diff_lstm}")

# comparison_lstm = np.allclose(tirReturn, tirLoaded, atol=1e-5, rtol=1e-4)
# max_diff_lstm = np.max(np.abs(tirReturn - tirLoaded))

# print(f"Are Core ML LSTMNet outputs close to PyTorch? {comparison_lstm}")
# print(f"Max absolute difference (Core ML LSTMNet vs PyTorch): {max_diff_lstm}")

#startsInitLoaded = np.loadtxt('inputFiles/startsInitZ.txt', delimiter=' ')
#endsInitLoaded = np.loadtxt('inputFiles/endsInitZ.txt', delimiter=' ')

#print("Init checks", np.all(startsInit == startsInitLoaded))
#print(np.all(endsInit == endsInitLoaded))

#savetxt_compact('startsInitPython.txt', torch.tensor(startsInit).unsqueeze(0), fmt='%1.8f', delimiter=' ')
#savetxt_compact('endsInitPython.txt', torch.tensor(endsInit).unsqueeze(0), fmt='%1.8f', delimiter=' ')

mlmodel_lstm = ct.models.MLModel("lstm_core_model.mlpackage")

# motionDataTest = torch.unsqueeze(torch.tensor(motionDataTest), 0).numpy()

lstm_input_dict = {
    "x": torch.tensor(np_dataset).unsqueeze(0).numpy(),
    "time_in_rep": time_in_rep
}
lstm_output_dict = mlmodel_lstm.predict(lstm_input_dict)
coreml_lstm_output_np = lstm_output_dict["var_357"]
coreml_lstm_output_np = coreml_lstm_output_np[0]
coreml_lstm_output_np = torch.tensor(coreml_lstm_output_np).squeeze(1).numpy()

#print(coreml_lstm_output_np)

lstmOutput = np.loadtxt('inputFiles/lstmOutput2.txt', delimiter=' ')
lstmOutput = torch.tensor(lstmOutput).to(torch.float32).numpy()

#print(lstmOutput)
print(np.all(coreml_lstm_output_np == lstmOutput))

print(F.sigmoid(torch.tensor(coreml_lstm_output_np)[-1]).round())

#print("Final equality", np.sum(coreml_lstm_output_np == finalPredictions) / coreml_lstm_output_np.size)

# arr1 = torch.tensor(np.arange(0, 2240)).unsqueeze(0)

# arr1 = arr1.unfold(1, winsize, LSTM_stride)

# built = []

# for i in range(32):
#     startInd = i * 64
#     window = np.arange(0, 2240)[startInd:(startInd + 256)]
#     built = np.concatenate((built, window), 0)

# print("Shape", built.shape)

# built = torch.tensor(np.array(built).reshape((1, 32, 256)))

# print(arr1.shape)

# print(np.all(arr1.numpy() == built.numpy()))

# savetxt_compact('built.txt', built.squeeze(0), fmt='%1.8f', delimiter=' ')

# savetxt_compact('unfoldTest.txt', arr1.squeeze(0), fmt='%1.8f', delimiter=' ')

#print(F.sigmoid(torch.tensor(coreml_lstm_output_np)).round())

tirInput2 = np.loadtxt('inputFiles/encoderInput.txt', delimiter=' ')
lstmInput2 = np.loadtxt('inputFiles/lstmInput2.txt', delimiter=' ')
print(np.sum(tirInput2 == lstmInput2))

print(F.sigmoid(torch.tensor(coreml_lstm_output_np)))