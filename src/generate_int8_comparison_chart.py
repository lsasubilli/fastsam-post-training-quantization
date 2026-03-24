import matplotlib.pyplot as plt
import os

OUT_DIR = '/home/ls5255/.gemini/antigravity/brain/a44cb83f-9fff-4bdd-8a5a-1ba663634d74'
os.makedirs(OUT_DIR, exist_ok=True)

# Common styling matching the user's examples
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.5,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

COLORS = {
    'blue': '#5b9bd5',
    'orange': '#ed7d31',
    'green': '#70ad47'
}

def generate_comparison_charts():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Real PTQ Inference Analysis (RTX 3050 Laptop GPU)", fontsize=18, fontweight='bold', y=1.05)
    
    # Measured Data from real_ptq_benchmark.py
    models = ['FastSAM (PyTorch FP32)', 'FastSAM (ONNX INT8 PTQ)']
    latency = [132.9, 334.8]  # Latency in ms
    fps = [1000/132.9, 1000/334.8]      # Frames per second
    bars_colors = [COLORS['blue'], COLORS['orange']]
    
    # Latency Chart
    bars1 = ax1.bar(models, latency, color=bars_colors, edgecolor='black', linewidth=1.5, width=0.6)
    ax1.set_ylabel('Inference Time (ms)', fontsize=14, fontweight='bold')
    ax1.set_title('Mean Inference Time Comparison', fontsize=16, fontweight='bold')
    ax1.set_ylim(0, 400)
    for bar, val in zip(bars1, latency):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 5, f'{val:.1f}ms', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
    # Throughput Chart
    bars2 = ax2.bar(models, fps, color=bars_colors, edgecolor='black', linewidth=1.5, width=0.6)
    ax2.set_ylabel('Frames Per Second (FPS)', fontsize=14, fontweight='bold')
    ax2.set_title('Throughput Comparison', fontsize=16, fontweight='bold')
    ax2.set_ylim(0, 10)
    
    for bar, val in zip(bars2, fps):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:.2f} FPS', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'real_ptq_int8_comparison_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    generate_comparison_charts()
    print("Done generating true FP32 vs ONNX INT8 comparison chart.")
