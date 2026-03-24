import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.5,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('FastSAM Inference Benchmark (RTX 3050 6GB Laptop GPU)', fontsize=18, fontweight='bold', y=1.02)

configs = ['PyTorch\nFP32 (GPU)', 'PyTorch\nFP16 (GPU)', 'ONNX INT8\nPTQ (GPU)', 'ONNX INT8\nPTQ (CPU)']
latency = [100.86, 103.10, 255.26, 3184.37]
std     = [1.81, 5.48, 10.97, 710.80]
fps     = [9.91, 9.70, 3.92, 0.31]
colors  = ['#5b9bd5', '#70ad47', '#ed7d31', '#c0504d']

bars1 = ax1.bar(configs, latency, color=colors, edgecolor='black', linewidth=1.5, width=0.6, yerr=std, capsize=5)
ax1.set_ylabel('Inference Time (ms)', fontsize=14, fontweight='bold')
ax1.set_title('Mean Inference Latency', fontsize=16, fontweight='bold')
ax1.set_yscale('log')
ax1.set_ylim(50, 5000)
for bar, val in zip(bars1, latency):
    ax1.text(bar.get_x() + bar.get_width()/2, val*1.15, f'{val:.1f}ms', ha='center', va='bottom', fontsize=12, fontweight='bold')

bars2 = ax2.bar(configs, fps, color=colors, edgecolor='black', linewidth=1.5, width=0.6)
ax2.set_ylabel('Frames Per Second (FPS)', fontsize=14, fontweight='bold')
ax2.set_title('Throughput', fontsize=16, fontweight='bold')
ax2.set_ylim(0, 12)
for bar, val in zip(bars2, fps):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.15, f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
out = '/home/ls5255/.gemini/antigravity/brain/a44cb83f-9fff-4bdd-8a5a-1ba663634d74/comprehensive_benchmark_chart.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print('Saved:', out)
