import torch
import models  
import utils
import matplotlib.pyplot as plt

WINDOW_SIZE = 60
NUM_TENSORS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD_FACTOR = 1.5

model = models.AEArch(window_size=WINDOW_SIZE)
model = torch.compile(model)
state_dict = torch.load("AEArch-1726921179.684368-20.pt", weights_only=True)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
data = utils.generate_random_time_series(num=NUM_TENSORS, window_size=WINDOW_SIZE, device=DEVICE)

model.eval()
with torch.no_grad():
    with torch.amp.autocast(device_type=DEVICE):  
        reconstructed = model(data)
        errors = torch.mean((reconstructed - data) ** 2, dim=1)

max_threshold = errors.mean() + THRESHOLD_FACTOR * errors.std()
min_threshold = errors.mean() - THRESHOLD_FACTOR * errors.std()
anomalies = (errors > max_threshold) | (errors < min_threshold)
