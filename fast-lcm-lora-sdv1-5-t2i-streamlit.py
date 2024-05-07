import streamlit as st
from diffusers import LCMScheduler, AutoPipelineForText2Image
from diffusers.utils import make_image_grid, load_image
import torch
import time

# Function to bypass the safety checker
def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False

# Initialize Diffusion Pipeline
def initialize_pipeline(use_gpu):
    model_id = "Lykon/dreamshaper-7"
    adapter_id = "latent-consistency/lcm-lora-sdv1-5"

    pipe = AutoPipelineForText2Image.from_pretrained(model_id, variant="fp16")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    if use_gpu:
        pipe.to("cuda")
        st.write("We are currently painting your requested image, please wait and be patient as using GPU will only take a short while.")
    else:
        st.write("We are currently painting your requested image, please wait and be patient as using CPU may take a bit longer.")

    # load and fuse lcm lora
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()

    return pipe

# Perform inference and display the result
def perform_inference(prompt, num_inference_steps, guidance_scale, use_gpu, manual_seed, disable_safety):
    pipe = initialize_pipeline(use_gpu)

    start_time = time.time()
    generator = torch.manual_seed(manual_seed)

    if disable_safety:
        pipe.safety_checker = disabled_safety_checker

    with st.spinner('Generating image...'):
        result = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

    end_time = time.time()
    latency_seconds = int(end_time - start_time)
    latency_minutes, latency_seconds = divmod(latency_seconds, 60)

    st.image(result, caption='Generated Image')
    st.write("Latency:", latency_minutes, "minutes and", latency_seconds, "seconds")
    st.success("Congratulations, your picture has been successfully painted and we hope you are satisfied with the result.")

def main():
    st.title("FAST LCM LoRA SD v1.5 - Text2Image")

    prompt = st.text_input("Enter your prompt:", "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")

    st.sidebar.title("Options")
    num_inference_steps = st.sidebar.slider("Number of Inference Steps", min_value=1, max_value=10, value=4)
    guidance_scale = st.sidebar.slider("Guidance Scale", min_value=0.0, max_value=5.0, value=0.0)
    manual_seed = st.sidebar.slider("Manual Seed", min_value=0, max_value=12013012031030, value=1231231)
    use_gpu = st.sidebar.radio("Select device:", ("CPU", "GPU"), index=0)
    disable_safety = st.sidebar.radio("Disable Safety Checker:", ("False", "True"), index=0)

    if disable_safety == "True":
        st.warning("You have chosen to Disable Safety Checker, which means the generated images may result in explicit and NSFW content. By generating images, you acknowledge that you are ready to bear the consequences and take responsibility for the outcomes.")

    if st.button("Generate Image"):
        perform_inference(prompt, num_inference_steps, guidance_scale, use_gpu == "GPU", manual_seed, disable_safety == "True")

if __name__ == "__main__":
    main()
