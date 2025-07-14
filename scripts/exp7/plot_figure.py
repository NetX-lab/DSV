import json 
import matplotlib.pyplot as plt
import time
import os


def find_latest_file(directory, prefix):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.startswith(prefix) and not f.endswith("_reference.json")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    return os.path.join(directory, files[0])



full_attn_file = find_latest_file('.', 'end_to_end_time_full_attn')
window_file = find_latest_file('.', 'end_to_end_time_window')
low_rank_file = find_latest_file('.', 'end_to_end_time_low_rank')

with open(full_attn_file, 'r') as f:
    full_attn_data = json.load(f)

with open(window_file, 'r') as f:
    window_data = json.load(f)

with open(low_rank_file, 'r') as f:
    low_rank_data = json.load(f)

# sort the data by step
full_attn_data = sorted(full_attn_data.items(), key=lambda x: int(x[0]))
low_rank_data = sorted(low_rank_data.items(), key=lambda x: int(x[0]))
window_data = sorted(window_data.items(), key=lambda x: int(x[0]))

# take the value of the data
full_attn_data = [float(x[1]) for x in full_attn_data][-10:]
low_rank_data = [float(x[1]) for x in low_rank_data][-10:]
window_data = [float(x[1]) for x in window_data][-10:]

mean_full_attn_data = sum(full_attn_data) / len(full_attn_data)
mean_low_rank_data = sum(low_rank_data) / len(low_rank_data) 
mean_window_data = sum(window_data) / len(window_data)

input_token = 32000 

throughput_full_attn = input_token / (mean_full_attn_data/1000)
throughput_low_rank = input_token / (mean_low_rank_data/1000)
throughput_window = input_token / (mean_window_data/1000)

label = ['Full Attention', 'Window Attention', 'Ours']


plt.bar(label, [throughput_full_attn, throughput_window, throughput_low_rank], color=['skyblue', 'green', 'red'])
plt.ylabel('Throughput (tokens/s)')
plt.xlabel('Model')
plt.title('Throughput Comparison')
plt.savefig(f'throughput_comparison_{time.strftime("%Y%m%d_%H%M")}.pdf', bbox_inches='tight')