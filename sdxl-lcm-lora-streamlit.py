import streamlit as st
from diffusers import DiffusionPipeline, LCMScheduler
import torch
import time

# Function to perform inference and display the result
def perform_inference(prompt, num_inference_steps, guidance_scale, use_gpu, manual_seed):

    # Initialize the Diffusion Pipeline
    if use_gpu:
        # Run on GPU
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

        pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")

        pipe.load_lora_weights(lcm_lora_id)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.to(device="cuda", dtype=torch.float16)

        st.write("We are currently painting your requested image, please wait and be patient as using GPU will only take a short while.")
    else:
        # Run on CPU
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

        pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")

        pipe.load_lora_weights(lcm_lora_id)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        st.write("We are currently painting your requested image, please wait and be patient as using CPU may take a bit longer.")

    # Measure latency
    start_time = time.time()

    generator = torch.Generator(device=pipe.device).manual_seed(manual_seed)
    # Perform inference
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images[0]

    end_time = time.time()
    latency_seconds = end_time - start_time
    latency_minutes = int(latency_seconds // 60)
    latency_seconds = int(latency_seconds % 60)

    # Display the image
    st.image(result, caption='Generated Image')

    # Print the latency
    st.write("Latency:", latency_minutes, "minutes and", latency_seconds, "seconds")
    
    st.success("Congratulations, your picture has been successfully painted and we hope you are satisfied with the result.")

def main():
    # Set page title
    st.title("SDXL LCM LoRA (4 steps)")

    # Get prompt input from user
    prompt = st.text_input("Enter your prompt:", "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux")

    # Create a sidebar for options
    st.sidebar.title("Options")
    num_inference_steps = st.sidebar.slider("Number of Inference Steps", min_value=1, max_value=10, value=4)
    guidance_scale = st.sidebar.slider("Guidance Scale", min_value=0.0, max_value=1.0, value=1.0)

    # Add slider to set manual_seed value
    manual_seed = st.sidebar.slider("Manual Seed", min_value=0, max_value=9999, value=1337)

    # Add radio button to choose GPU or CPU with CPU as default
    use_gpu = st.sidebar.radio("Select device:", ("CPU", "GPU"), index=0)

    # Perform inference when the button is clicked
    if st.button("Generate Image"):
        perform_inference(prompt, num_inference_steps, guidance_scale, use_gpu == "GPU", manual_seed)

if __name__ == "__main__":
    main()
