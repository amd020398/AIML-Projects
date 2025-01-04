from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
print("Stable Diffusion Pipeline loaded successfully!")
