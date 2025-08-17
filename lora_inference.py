import torch

from diffusers import DiffusionPipeline

# def basic_inference():
#     print("=== 基础版本 ===")
#     # 加载您训练的基础模型和LoRA
#     pipeline = DiffusionPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-2-1-base",  # 您使用的基础模型
#         torch_dtype=torch.float16
#     ).to("cuda")
    
#     # 加载您的LoRA权重
#     pipeline.load_lora_weights(
#         "/root/autodl-tmp/",  # 您的LoRA路径
#         weight_name="pytorch_lora_weights.safetensors",  # 指定权重文件名
#         adapter_name="naruto"
#     )
    
#     # activate LoRA and set adapter weight  
#     pipeline.set_adapters("naruto", adapter_weights=0.8)
    
#     # 生成图像
#     images = pipeline("A naruto with blue eyes, high quality, detailed").images
    
#     # 保存图像
#     for i, image in enumerate(images):
#         image.save(f"naruto_basic_{i}.png")
    
#     print("基础版本完成！")
#     return images

def basic_inference_with_batch():
    print("=== 基础版本 ===")
    # 加载您训练的基础模型和LoRA
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",  # 您使用的基础模型
        torch_dtype=torch.float16
    ).to("cuda")
    
    # 加载您的LoRA权重
    pipeline.load_lora_weights(
        "/root/autodl-tmp/",  # 您的LoRA路径
        weight_name="pytorch_lora_weights.safetensors",  # 指定权重文件名
        adapter_name="naruto"
    )
    
    # activate LoRA and set adapter weight  
    pipeline.set_adapters("naruto", adapter_weights=0.8)
    
    # 生成图像
    
    prompts = [
        "A naruto with blue eyes",
        "A naruto ninja in the forest",
        "A naruto character sitting on a tree",
        "A naruto with orange outfit, smiling"
        ]
    
    # 保存图像
    for i, prompt in enumerate(prompts):
        images = pipeline(prompt, num_inference_steps=30, guidance_scale=7.5).images
        images[0].save(f"lora_output_imgs/lora_naruto_batch_{i}.png")
        # images[0].save(f"original_output_imgs/lora_naruto_batch_{i}.png")
    
    print("基础版本完成！")
    return images


# =============================================================================
# 版本2: 优化版本 - 包含性能优化
# =============================================================================

# def optimized_inference():
#     print("=== 优化版本 ===")
#     # 加载基础模型和LoRA
#     pipeline = DiffusionPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-2-1-base",
#         torch_dtype=torch.float16
#     ).to("cuda")
    
#     pipeline.load_lora_weights(
#         "/root/autodl-tmp/lora/naruto",
#         weight_name="pytorch_lora_weights.safetensors",
#         adapter_name="naruto"
#     )
    
#     # 激活LoRA并设置适配器权重
#     pipeline.set_adapters("naruto", adapter_weights=0.8)
    
#     # 性能优化步骤
#     print("应用性能优化...")
    
#     # 1. 融合LoRA到基础模型中（提高推理速度）
#     pipeline.fuse_lora(adapter_names=["naruto"], lora_scale=1.0)
    
#     # 2. 卸载LoRA权重（释放内存）
#     pipeline.unload_lora_weights()
    
#     # 3. 优化内存格式（提高GPU利用率）
#     pipeline.unet.to(memory_format=torch.channels_last)
    
#     # 4. 编译模型（提高推理速度，需要较新的PyTorch版本）
#     try:
#         pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
#         print("✅ 模型编译成功")
#     except Exception as e:
#         print(f"⚠️ 模型编译失败，跳过: {e}")
    
#     # 生成图像
#     print("开始生成图像...")
#     images = pipeline(
#         "A naruto with blue eyes, anime style, high quality, detailed",
#         num_inference_steps=50,
#         guidance_scale=7.5,
#         num_images_per_prompt=4
#     ).images
    
#     # 保存图像
#     for i, image in enumerate(images):
#         image.save(f"naruto_optimized_{i}.png")
    
#     print("优化版本完成！")
#     return images

# =============================================================================
# 版本3: 最简版本 - 如果上面的有问题
# =============================================================================

# def simple_inference():
#     print("=== 最简版本 ===")
#     # 最简单的方式
#     pipeline = DiffusionPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-2-1-base",
#         torch_dtype=torch.float16
#     ).to("cuda")
    
#     # 直接加载LoRA权重（传统方式）
#     pipeline.load_lora_weights("/root/autodl-tmp/lora/naruto")
    
#     # 生成图像
#     images = pipeline("A naruto with blue eyes").images
    
#     for i, image in enumerate(images):
#         image.save(f"naruto_simple_{i}.png")
    
#     print("最简版本完成！")
#     return images

# =============================================================================
# 批量测试不同prompt
# =============================================================================

# def batch_inference():
#     print("=== 批量推理 ===")
#     pipeline = DiffusionPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-2-1-base",
#         torch_dtype=torch.float16
#     ).to("cuda")
    
#     pipeline.load_lora_weights(
#         "/root/autodl-tmp/lora/naruto",
#         weight_name="pytorch_lora_weights.safetensors",
#         adapter_name="naruto"
#     )
#     pipeline.set_adapters("naruto", adapter_weights=0.8)
    
#     # 多个测试prompt
#     prompts = [
#         "A naruto with blue eyes",
#         "A naruto ninja in the forest",
#         "A naruto character sitting on a tree",
#         "A naruto with orange outfit, smiling"
#     ]
    
#     for i, prompt in enumerate(prompts):
#         print(f"生成图像 {i+1}/4: {prompt}")
#         images = pipeline(prompt, num_inference_steps=30).images
#         images[0].save(f"naruto_batch_{i}.png")
    
#     print("批量推理完成！")

if __name__ == "__main__":
    # 选择运行哪个版本
    
    # 1. 基础版本（推荐先试这个）
    basic_inference_with_batch()
    
    # 2. 优化版本（如果基础版本正常运行）
    # optimized_inference()
    
    # 3. 最简版本（如果其他版本有问题）
    # simple_inference()
    
    # 4. 批量测试
    # batch_inference()
    
    print("推理完成！检查生成的图片文件。")
