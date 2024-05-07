import streamlit as st
from diffusers import AutoPipelineForImage2Image, LCMScheduler
from diffusers.utils import make_image_grid, load_image
import torch
import time
from PIL import Image
import io
from io import BytesIO
import requests

# Function to bypass the safety checker
def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False

# Initialize Diffusion Pipeline for Image2Image
def initialize_pipeline(use_gpu):
    model_id = "Lykon/dreamshaper-7"
    adapter_id = "latent-consistency/lcm-lora-sdv1-5"

    pipe = AutoPipelineForImage2Image.from_pretrained(model_id, variant="fp16")

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    if use_gpu:
        pipe.to("cuda")
        st.write("We are currently processing your requested image, please wait and be patient as using GPU will only take a short while.")
        # st.warning("To speed up the image generation process and save computational resources, images will be automatically resized to <= 768.")
    else:
        st.write("We are currently processing your requested image, please wait and be patient as using CPU may take a bit longer. ")
        # st.warning("To speed up the image generation process and save computational resources, images will be automatically resized to <= 768.")

    # load and fuse lcm lora
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()

    return pipe

# Fungsi untuk mengatur ukuran gambar menjadi setengah dari dimensi aslinya, tapi tidak lebih dari 768 baik lebar maupun tinggi
def resize_image_half(image):
    try:
        width, height = image.size
    except AttributeError:
        st.error("Error: Unable to process the image file. Please make sure the file is a valid image.")
        return None

    print("Original Image Size:", width, height)  # Periksa ukuran gambar sebelum diresize

    max_dimension = 512

    # Check if both width and height are greater than or equal to 768
    if width >= max_dimension and height >= max_dimension:
        if width >= height:
            new_width = min(width // 2, max_dimension)
            new_height = min(height * new_width // width, max_dimension)
        else:
            new_height = min(height // 2, max_dimension)
            new_width = min(width * new_height // height, max_dimension)
        
        resized_image = image.resize((new_width, new_height))
        print("Resized Image Size:", new_width, new_height)  # Periksa ukuran gambar setelah diresize
    else:
        resized_image = image

    return resized_image



def perform_inference(prompt, input_option, input_value, num_inference_steps, guidance_scale, strength, use_gpu, disable_safety):
    pipe = initialize_pipeline(use_gpu)

    start_time = time.time()
    generator = torch.manual_seed(0)

    if disable_safety:
        pipe.safety_checker = disabled_safety_checker

    # Load image based on input option
    if input_option == "URL":
        if input_value.strip():
            response = requests.get(input_value)
            if response.status_code == 200:
                image_bytes = BytesIO(response.content)
                init_image = Image.open(image_bytes)
                init_image = resize_image_half(init_image)
            else:
                st.error("Failed to fetch the image from the provided URL. Please make sure the URL is correct.")
                return
    elif input_option == "Local":
        if input_value is not None:
            init_image = Image.open(io.BytesIO(input_value.read()))
            # Resize image
            init_image = resize_image_half(init_image)
        else:
            st.error("Please upload a local image.")
            return
    else:
        st.error("Invalid input option selected.")
        return

    # Check if the image is in RGB mode, if not, convert it to RGB
    if init_image.mode != "RGB":
        init_image = init_image.convert("RGB")

    with st.spinner('Processing image...'):
        result = pipe(
            prompt=prompt,
            image=init_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator
        ).images[0]

    end_time = time.time()
    latency_seconds = int(end_time - start_time)
    latency_minutes, latency_seconds = divmod(latency_seconds, 60)

    # st.image(init_image, caption='Original Image')
    # st.image(result, caption='Processed Image')
    width, height = result.size
    st.image(result, caption=f'Processed Image with Resize(Width: {width}, Height: {height})')
    st.write("Latency:", latency_minutes, "minutes and", latency_seconds, "seconds")
    st.success("Image processing completed successfully.")

def main():
    st.title("FAST LCM LoRA SD v1.5 - Image2Image")

    prompt = st.text_input("Enter your prompt:", "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k")

    st.sidebar.title("Options")

    input_option = st.sidebar.radio("Select Input Option:", ("URL", "Local"))

    uniform_size = st.sidebar.radio("Uniform Size:", ("No", "Yes"))

    if input_option == "URL":
        input_value = st.sidebar.text_input("Enter URL of initial image:", "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png")
        # clear_button = input_value.button("x", on_click=clear_input)
        if input_value:
            response = requests.get(input_value)
            image = Image.open(BytesIO(response.content))
            width, height = image.size
            if uniform_size == "Yes":
                image = resize_image_half(image)
                width, height = image.size
                st.image(image, caption=f'Original Image by the link after resize(Width: {width}, Height: {height})') 
            else:
                st.image(image, caption=f'Original Image by the link (Width: {width}, Height: {height})')
                st.warning("To speed up the image generation process and save computational resources, images will be automatically resized to <= 768.")
            st.write(f"Image size: {len(response.content) / 1024:.2f} KB")
        else:
            st.warning("Please provide a URL.")
    elif input_option == "Local":
        input_value = st.sidebar.file_uploader("Upload Local Image", type=["jpg", "png", "jpeg"])
        if input_value is not None:
            try:
                # Read the image content from BytesIO
                image_content = input_value.read()
                # Reset the cursor position to the beginning
                input_value.seek(0)
                # Try to open the image using PIL
                image = Image.open(io.BytesIO(image_content))
                # Display the image
                width, height = image.size
                if uniform_size == "Yes":
                    # Resize image
                    image = resize_image_half(image)
                    width, height = image.size
                    st.image(image, caption=f'Original Image by upload after resize (Width: {width}, Height: {height})')
                else:
                    st.image(image, caption=f'Original Image by upload (Width: {width}, Height: {height})')
                    st.warning("To speed up the image generation process and save computational resources, images will be automatically resized to <= 768.")
                # Calculate the size of the image content
                image_size_kb = len(image_content) / 1024
                st.write(f"Image size: {image_size_kb:.2f} KB")
            except Exception as e:
                st.error("Error: Unable to process the image file. Please make sure the file is a valid image.")
                st.error(str(e))
        else:
            st.warning("Please upload an image.")
    else:
        st.error("Invalid input option selected.")
    
    num_inference_steps = st.sidebar.slider("Number of Inference Steps",
                                            min_value=1,
                                            max_value=10,
                                            value=4)
    
    guidance_scale = st.sidebar.slider("Guidance Scale",
                                       min_value=0.0,
                                       max_value=5.0,
                                       value=1.0)
    
    strength = st.sidebar.slider("Strength",
                                 min_value=0.0,
                                 max_value=1.0,
                                 value=0.6)
    
    use_gpu = st.sidebar.radio("Select device:", ("CPU", "GPU"), index=0)
    if use_gpu == "GPU":
        st.warning("You are currently using the GPU option. Please ensure your system has an Nvidia GPU with a minimum of 4 GB VRAM and has torch-gpu installed. If not, please use the CPU option. If you continue using the GPU option without meeting these requirements, it will result in an error.")
    
    disable_safety = st.sidebar.radio("Disable Safety Checker:",
                                      ("False", "True"), index=0)
    
    if disable_safety == "True":
        st.warning("You have chosen to Disable Safety Checker, which means the processed images may result in explicit and NSFW content. By processing images, you acknowledge that you are ready to bear the consequences and take responsibility for the outcomes.")

    if st.button("Process Image"):
        perform_inference(prompt,
                          input_option,
                          input_value,
                          num_inference_steps,
                          guidance_scale,
                          strength,
                          use_gpu == "GPU",
                          disable_safety == "True")


if __name__ == "__main__":
    main()
