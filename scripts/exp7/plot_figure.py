import json 
import matplotlib.pyplot as plt

full_attn_file = 'end_to_end_time_full_attn.json'
low_rank_file = 'end_to_end_time_low_rank.json'

with open(full_attn_file, 'r') as f:
    full_attn_data = json.load(f)

with open(low_rank_file, 'r') as f:
    low_rank_data = json.load(f)

# sort the data by step
full_attn_data = sorted(full_attn_data.items(), key=lambda x: int(x[0]))
low_rank_data = sorted(low_rank_data.items(), key=lambda x: int(x[0]))

# take the value of the data
full_attn_data = [float(x[1]) for x in full_attn_data][-10:]
low_rank_data = [float(x[1]) for x in low_rank_data][-10:]

mean_full_attn_data = sum(full_attn_data) / len(full_attn_data)
mean_low_rank_data = sum(low_rank_data) / len(low_rank_data) 


input_token = 32000 

throughput_full_attn = input_token / (mean_full_attn_data/1000)
throughput_low_rank = input_token / (mean_low_rank_data/1000)

label = ['Full Attention', 'Ours']


plt.bar(label, [throughput_full_attn, throughput_low_rank], color=['skyblue', 'red'])
plt.ylabel('Throughput (tokens/s)')
plt.xlabel('Model')
plt.title('Throughput Comparison')
plt.savefig('throughput_comparison.pdf', bbox_inches='tight')