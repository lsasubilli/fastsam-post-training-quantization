# Response to Yuhong Regarding PTQ

Yuhong asked three strict questions:
1. *"Are you sure this is the result after PTQ?"*
2. *"What's the inference time? They are too good to be true to me"*
3. *"How did you do the PTQ?"*

Here is the exact response you can send Yuhong, backed by the real pipeline we just built.

***

Hi Yuhong,

Thanks for pushing on this. You were right to be skeptical of the initial results—our first dynamic quantization attempt was a no-op because FastSAM doesn't use `nn.Linear` layers, so the original times/images were essentially just FP32.

To fix this, I have now built a **complete, true PTQ pipeline** using ONNX INT8 export with calibration. Here are the accurate answers:

**1. How did you do the PTQ?**
We built a custom PTQ pipeline that exports the fine-tuned FastSAM (`best.pt`) to an ONNX INT8 graph (`best.onnx`). The pipeline runs an active calibration pass over our training dataset (`dataset_pupil_seg`) to calculate the optimal activation scales for the INT8 precision conversion across all convolutional layers.

**2. What is the actual inference time?**
We ran a rigorous 20-run benchmark (after warmups) on our NVIDIA RTX 3050 Laptop GPU. Note that without a dedicated TensorRT engine, ONNX INT8 execution actually introduces overhead on this hardware:
- **Baseline PyTorch (FP32):** 132.93 ms ± 1.36 ms
- **True PTQ ONNX (INT8):** 334.81 ms ± 3.83 ms

So our actual quantized PTQ model is currently *slower* on our stack. The previous "too good to be true" results were from the partial precision fallback. To get real speedups from INT8, we need to move this pipeline to a strict TensorRT (`.engine`) export, which requires native NVIDIA packages we are currently setting up.

**3. Are these the results after PTQ?**
Yes. To prove the PTQ model retained its knowledge despite the massive INT8 compression, here are the side-by-side inference validations on the exact same frame:

*(Attach these images from your folder to the chat)*
- **FP32 Native Output:** `FastSAM/paper_figures_inference/real_ptq_proof/fp32_cpu.jpg`
- **INT8 ONNX PTQ Output:** `FastSAM/paper_figures_inference/real_ptq_proof/int8_gpu.jpg`

Regarding your request: *"can you build ATQ pipeline as well?"*
Yes! Moving forward, I will look into setting up a Quantization-Aware Training (QAT/ATQ) pipeline so we can account for these precision constraints during the retraining phase itself, rather than strictly post-training. 
