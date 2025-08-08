import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from torch import nn
import torch.nn.functional as F
from scipy.stats import linregress
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

in_channels = 6
winsize = 256
LSTM_stride = 64
n_LSTM_windows = 32
full_winsize = LSTM_stride * (n_LSTM_windows-1) + winsize

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

    print("End Reps:", end_reps)

    # rewindow
    time_in_rep = time_in_rep.unfold(1, winsize, stride)

    if not os.path.isfile('fullTIRComparison2.txt'):
        np.savetxt('fullTIRComparison2.txt', time_in_rep.squeeze(0).numpy())
        print("Saved file")

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

weights, config = torch.load('best_model-3binary-77.pth', map_location=torch.device("cpu"))#, weights_only=True)
model = LSTMNet(config).to(torch.device('cpu'))
model.load_state_dict(weights)

model.eval()

modelNew = LSTMNetNew(config)
modelNew.load_state_dict(weights)
modelNew.eval()

# Prepare example_input
example_input_data = []
for i in range(32):
    example_window = []
    for _ in range(256):
        example_sample = [random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-2, 2), 
                          random.uniform(-250, 250), random.uniform(-250, 250), random.uniform(-250, 250)]
        example_window.append(example_sample)
    example_input_data.append(example_window)

def moving_average(data, window_size=15):
    print("Shape:", data.shape)
    X_smooth = np.zeros(data.shape)
    for i,channel in enumerate(data):
        X_smooth[i] = np.convolve(channel, np.ones(window_size)/window_size, mode='same')
    return torch.from_numpy(X_smooth).to(torch.float32)

#evaluate_real_time_comparison = np.loadtxt('evaluate-real-time-comparison.txt')
#print(evaluate_real_time_comparison.shape)
eval2 = np.loadtxt('eval2.txt')

dataset = []
# firstWindow = evaluate_real_time_comparison[:, :256]
# firstWindow = moving_average(firstWindow)
# dataset.append(firstWindow)
# for i in range(32):
#     start = (i * 64)
#     end = start + 256
#     #print(start, end)
#     window = evaluate_real_time_comparison[:, start:end]
#     window = moving_average(window)
#     dataset.append(window)
# np_dataset = np.array(dataset)
#print(np_dataset.shape)
stride = 64

confs = []

np_dataset = np.loadtxt('inputFiles/finalEncoderInput.txt', delimiter=' ').astype('float32')
np_dataset = np_dataset.reshape((32, 6, 256))

finalCheckDS = []

sample = []
for i in range(256):
    accx = np_dataset[0][0][i]
    accy = np_dataset[0][1][i]
    accz = np_dataset[0][2][i]
    gyrx = np_dataset[0][3][i]
    gyry = np_dataset[0][4][i]
    gyrz = np_dataset[0][5][i]

    sample = [accx, accy, accz, gyrx, gyry, gyrz]
    finalCheckDS.append(sample)

for j in range(31):
    sample = []
    for i in range(64):
        accx = np_dataset[1 + j][0][i + 192]
        accy = np_dataset[1 + j][1][i + 192]
        accz = np_dataset[1 + j][2][i + 192]
        gyrx = np_dataset[1 + j][3][i + 192]
        gyry = np_dataset[1 + j][4][i + 192]
        gyrz = np_dataset[1 + j][5][i + 192]

        sample = [accx, accy, accz, gyrx, gyry, gyrz]
        finalCheckDS.append(sample)

finalCheckDS = np.array(finalCheckDS)

print("FinalCheckDS size: ", finalCheckDS.shape)

np.savetxt('./finalCheckDS.txt', finalCheckDS, delimiter=',')

np.savetxt('finalEncoderInputAppend.txt', torch.tensor(np_dataset).flatten().unsqueeze(0).numpy(), delimiter=',')

def savetxt_compact(fname, x, fmt="%.6g", delimiter=','):
    with open(fname, 'w') as fh:
        for row in x:
            line = delimiter.join("0.0" if value == 0 else fmt % value for value in row)
            fh.write(line + '\n')

foldedDataset = []

print(np_dataset[0].shape)

window = np_dataset[0][:]
for i in range(4):
    uniqueData = []
    for j in range(6):
        uniqueData.append(window[j][64 * i : 64 + (64 * i)])
    uniqueData = np.array(uniqueData)
    print("Unique Shape:", uniqueData.shape)
    foldedDataset.append(uniqueData)

#foldedDataset.append(np_dataset[0])

#print(np.array(foldedDataset).shape)

for i in range(31):
    window = np_dataset[i + 1][:]
    uniqueData = []
    for j in range(6):
        uniqueData.append(window[j][192:256])
    uniqueData = np.array(uniqueData)
    print("Unique Shape:", uniqueData.shape)
    foldedDataset.append(uniqueData)

foldedDataset = np.array(foldedDataset)
print(foldedDataset.shape)

foldedDataset = foldedDataset.transpose(0, 2, 1)
foldedDataset = np.concatenate(foldedDataset, 0)

# fakeColumn = np.full((2240, 1), 72)
# fakeText = np.full((2240, 1), 'blah')

# print(foldedDataset.shape)

# foldedDataset = np.concatenate((fakeColumn, foldedDataset, fakeColumn, fakeText, fakeText, fakeColumn, fakeText), axis=1)

print(foldedDataset.shape)

with torch.no_grad():
    for i in range(32):
        partialOutput = model(torch.tensor(np_dataset[0:(i+1)]).unsqueeze(0))
        confs.append(F.sigmoid(partialOutput.squeeze(0).squeeze(1)[-1]))

print("CONFS???:", confs)

#np_dataset2 = np.loadtxt('evaluate-real-time-comparison2.txt').reshape((32, 6, 256)).astype('float32')

#print("Datasets equal?", np.all(np_dataset == np_dataset2))

# with torch.no_grad():
#     partialOutput = model(torch.tensor(np_dataset).unsqueeze(0))

sigmoidlstm = np.loadtxt('inputFiles/finalLSTM.txt', delimiter=' ', dtype='float32')
sigmoidlstm = torch.tensor(sigmoidlstm)
sigmoidlstm = F.sigmoid(sigmoidlstm)
np.savetxt('./finalLSTMComparison.txt', sigmoidlstm, delimiter=' ', fmt='%.18f')

print("Partial Output:", partialOutput)

eval2 = eval2.reshape((32, 256, 6)).transpose(0, 2, 1)

#example_input_x = torch.tensor(example_input_data).transpose(1, 2).contiguous().unsqueeze(0).to(torch.float32)
example_input_x = torch.tensor(np_dataset[0:7]).contiguous().unsqueeze(0).to(torch.float32)

temp_encoder_model = ConvNet(config) # Only need the encoder part
temp_encoder_model.load_state_dict(model.encoder.state_dict()) # Load relevant weights
temp_encoder_model.eval()

N, T, C, L = example_input_x.shape
x_flat_for_seg = example_input_x.contiguous().view(N*T, C, L)
_, seg_logits_for_time_in_rep = temp_encoder_model(x_flat_for_seg)

# Apply sigmoid and round as done in the original forward
seg_logits_for_time_in_rep = F.sigmoid(seg_logits_for_time_in_rep).round()

# Get time_in_rep on CPU, as this is the problematic part
seg_logits_cpu_for_time_in_rep = seg_logits_for_time_in_rep.view(N, T, L).detach().cpu()
precomputed_time_in_rep_cpu = get_time_in_reps(seg_logits_cpu_for_time_in_rep, LSTM_stride, winsize)

precomputed_time_in_rep = precomputed_time_in_rep_cpu.to(example_input_x.device)

# Now, the inputs for torch.export are `example_input_x` and `precomputed_time_in_rep`
example_inputs = (example_input_x, precomputed_time_in_rep)
example_inputs_encoder = (x_flat_for_seg,)

# Now export the model with the two inputs
torch._dynamo.config.capture_dynamic_output_shape_ops = True
traced_model = torch.jit.trace(modelNew, example_inputs)
exported_model = traced_model
#exported_model = torch.export.export(modelNew, example_inputs, strict=False)
exported_encoder = torch.export.export(temp_encoder_model, example_inputs_encoder, strict=False)

with torch.no_grad():
    og_output = model(example_input_x)

with torch.no_grad():
    new_output = modelNew(example_input_x, precomputed_time_in_rep)

# Compare original_output and split_python_output
# Use torch.allclose for numerical stability
print(f"Are Python outputs close? {torch.allclose(og_output, new_output, atol=1e-5, rtol=1e-4)}")
# Adjust atol (absolute tolerance) and rtol (relative tolerance) as needed
# You can also look at the mean absolute error (MAE) or max absolute difference
print(f"Max absolute difference: {torch.max(torch.abs(og_output - new_output))}")


with torch.no_grad():
    _, seg_logits_for_time_in_rep = exported_encoder.module()(x_flat_for_seg)

# Apply sigmoid and round as done in the original forward
seg_logits_for_time_in_rep = F.sigmoid(seg_logits_for_time_in_rep).round()

# Get time_in_rep on CPU, as this is the problematic part
seg_logits_cpu_for_time_in_rep = seg_logits_for_time_in_rep.view(N, T, L).detach().cpu()
precomputed_time_in_rep_cpu = get_time_in_reps(seg_logits_cpu_for_time_in_rep, LSTM_stride, winsize)

precomputed_time_in_rep = precomputed_time_in_rep_cpu.to(example_input_x.device)

with torch.no_grad():
    exported_output = exported_model(example_input_x, precomputed_time_in_rep)

# print(f"Are Python outputs close? {torch.allclose(new_output, exported_output, atol=1e-5, rtol=1e-4)}")
# print(f"Max absolute difference: {torch.max(torch.abs(new_output - exported_output))}")

import coremltools as ct

# ... (all your existing code) ...

# --- Convert LSTMNet model ---
print("\nConverting LSTMNet model...")
try:
    coreml_model_lstm = ct.convert(
        exported_model, # This is your exported_program from torch.export
        inputs=[
            ct.TensorType(name="x", shape=example_input_x.shape, dtype=np.float32), # Specify dtype here
            ct.TensorType(name="time_in_rep", shape=precomputed_time_in_rep.shape, dtype=np.float32) # Specify dtype here
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32, # <--- ADD THIS LINE
        # minimum_deployment_target=ct.target.iOS15 # Keep this if you want a specific target
    )
    coreml_model_lstm.save("lstm_core_model.mlpackage")
    print("LSTMNet model saved as lstm_core_model.mlpackage")
except Exception as e:
    print(f"Failed to convert LSTMNet model: {e}")

# --- Convert ConvNet (encoder) model ---
print("\nConverting ConvNet (encoder) model...")
try:
    coreml_model_encoder = ct.convert(
        exported_encoder, # This is your exported_program for the encoder
        inputs=[
            ct.TensorType(name="x", shape=example_inputs_encoder[0].shape, dtype=np.float32) # Specify dtype here
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32, # <--- ADD THIS LINE
        # minimum_deployment_target=ct.target.iOS15
    )
    coreml_model_encoder.save("convnet_model.mlpackage") # Corrected variable name
    print("ConvNet (encoder) model saved as convnet_model.mlpackage")
except Exception as e:
    print(f"Failed to convert ConvNet (encoder) model: {e}")

mlmodel_lstm = ct.models.MLModel("lstm_core_model.mlpackage")
mlmodel_convnet = ct.models.MLModel("convnet_model.mlpackage")

x_flat_for_seg_np = example_inputs_encoder[0].cpu().numpy()
example_input_x_np = example_input_x.cpu().numpy()
precomputed_time_in_rep_np = precomputed_time_in_rep.cpu().numpy()

convnet_input_dict = {"x": x_flat_for_seg_np}
convnet_output_dict = mlmodel_convnet.predict(convnet_input_dict)

def get_model_output_names(ml_model):
    """Helper function to get output names from an MLModel spec."""
    spec = ml_model.get_spec()
    output_names = [o.name for o in spec.description.output]
    return output_names

def get_model_output_names_and_shapes(ml_model):
    """Helper function to get output names and shapes from an MLModel spec."""
    spec = ml_model.get_spec()
    output_info = []
    for o in spec.description.output:
        # For MLMultiArray, the shape is in the multiArrayType
        if o.type.HasField("multiArrayType"):
            shape = tuple(o.type.multiArrayType.shape)
        else:
            shape = None # Or handle other types if your model has them
        output_info.append({"name": o.name, "shape": shape})
    return output_info

print("\nInspecting ConvNet model outputs:")
convnet_output_info = get_model_output_names_and_shapes(mlmodel_convnet)
print(f"ConvNet output info: {convnet_output_info}")

expected_seg_logits_shape = (N * T, L)

coreml_seg_logits_key = None
coreml_emb_key = None

for output_spec in convnet_output_info:
    if output_spec["shape"] == expected_seg_logits_shape:
        coreml_seg_logits_key = output_spec["name"]
    else:
        coreml_emb_key = output_spec["name"] # The other one must be emb

if coreml_seg_logits_key is None or coreml_emb_key is None:
    raise ValueError(f"Could not reliably determine seg_logits and emb output keys from ConvNet outputs: {convnet_output_info}. Please inspect the .mlpackage in Netron to confirm shapes.")

print(f"Assumed Core ML seg_logits key: '{coreml_seg_logits_key}'")
print(f"Assumed Core ML emb key: '{coreml_emb_key}'")

lstm_output_info = get_model_output_names_and_shapes(mlmodel_lstm)
print(f"LSTMNet output info: {lstm_output_info}")
coreml_lstm_output_key = lstm_output_info[0]["name"]

convnet_input_dict = {"x": x_flat_for_seg_np}
convnet_output_dict = mlmodel_convnet.predict(convnet_input_dict)
coreml_seg_logits_np = convnet_output_dict[coreml_seg_logits_key]
# coreml_emb_np = convnet_output_dict[coreml_emb_key] # You can get this too if you want to compare emb

print(f"Core ML ConvNet output shape (seg_logits): {coreml_seg_logits_np.shape}")

coreml_seg_logits_torch = torch.from_numpy(coreml_seg_logits_np).to(torch.float32)

# Apply sigmoid and round if not already done by the Core ML model
# (If you're unsure, you can always check the min/max values of coreml_seg_logits_np
# or open the .mlpackage in Netron to see if these ops are included after the 'linear' output).
if not np.all((coreml_seg_logits_np == 0) | (coreml_seg_logits_np == 1)):
    coreml_seg_logits_torch = F.sigmoid(coreml_seg_logits_torch).round()


# Reshape for get_time_in_reps: (N*T, L) -> (N, T, L)
coreml_seg_logits_reshaped_for_time_in_rep = coreml_seg_logits_torch.view(N, T, L).detach().cpu()
coreml_precomputed_time_in_rep_torch = get_time_in_reps(
    coreml_seg_logits_reshaped_for_time_in_rep, LSTM_stride, winsize
)
coreml_precomputed_time_in_rep_np = coreml_precomputed_time_in_rep_torch.cpu().numpy()

print(f"Core ML precomputed time_in_rep shape: {coreml_precomputed_time_in_rep_np.shape}")

print("Running LSTMNet Core ML model...")
lstm_input_dict = {
    "x": example_input_x_np,
    "time_in_rep": coreml_precomputed_time_in_rep_np
}
lstm_output_dict = mlmodel_lstm.predict(lstm_input_dict)
coreml_lstm_output_np = lstm_output_dict[coreml_lstm_output_key]
print(f"Core ML LSTMNet output shape: {coreml_lstm_output_np.shape}")

print("\nComparing Core ML outputs with original PyTorch outputs...")

new_output_np = new_output.cpu().numpy()
#print(F.sigmoid(torch.tensor(new_output_np)))
comparison_lstm = np.allclose(F.sigmoid(torch.tensor(new_output_np)).numpy(), F.sigmoid(torch.tensor(coreml_lstm_output_np)).numpy(), atol=1e-5, rtol=1e-4)
max_diff_lstm = np.max(np.abs(F.sigmoid(torch.tensor(new_output_np)).numpy() - F.sigmoid(torch.tensor(coreml_lstm_output_np)).numpy()))

#print(F.sigmoid(torch.tensor(new_output_np)).numpy() - F.sigmoid(torch.tensor(coreml_lstm_output_np)).numpy())

print(f"Are Core ML LSTMNet outputs close to PyTorch? {comparison_lstm}")
print(f"Max absolute difference (Core ML LSTMNet vs PyTorch): {max_diff_lstm}")

seg_logits_for_time_in_rep_np = seg_logits_for_time_in_rep.cpu().numpy()
comparison_seg_logits = np.allclose(seg_logits_for_time_in_rep_np, coreml_seg_logits_np, atol=1e-5, rtol=1e-4)
max_diff_seg_logits = np.max(np.abs(seg_logits_for_time_in_rep_np - coreml_seg_logits_np))

print(f"Are Core ML ConvNet seg_logits close to PyTorch? {comparison_seg_logits}")
print(f"Max absolute difference (Core ML ConvNet seg_logits vs PyTorch): {max_diff_seg_logits}")

comparison_precomputed = np.allclose(precomputed_time_in_rep_cpu.numpy(), coreml_precomputed_time_in_rep_np, atol=1e-5, rtol=1e-4)
max_diff_precomputed = np.max(np.abs(precomputed_time_in_rep_cpu.numpy() - coreml_precomputed_time_in_rep_np))

print(f"Are Core ML-derived precomputed_time_in_rep close to PyTorch-derived? {comparison_precomputed}")
print(f"Max absolute difference (Core ML precomputed vs PyTorch): {max_diff_precomputed}")


print("OG Output:", F.sigmoid(og_output))

tir1 = np.loadtxt('fullTIRComparison.txt')
tir2 = np.loadtxt('fullTIRComparison2.txt')
print(np.all(tir1 == tir2))