#!/usr/bin/env python3
"""
Example script demonstrating how to use the Wavespeed client to generate images asynchronously.
"""

import logging
import os
import sys
import argparse
import asyncio
from pathlib import Path

# Add the parent directory to the path so we can import the wavespeed package
sys.path.append(str(Path(__file__).parent.parent))

from wavespeed.client import Wavespeed


async def generate_image(client: Wavespeed, args):
    try:
        print(f"Generating image with prompt: '{args.prompt}'...")
        # Use the async_run method for asynchronous image generation
        prediction = await client.async_run(
            modelId="wavespeed-ai/flux-dev",
            input={
                "prompt": args.prompt,
                "strength": args.strength,
                "size": args.size,
                "num_inference_steps": args.steps,
                "guidance_scale": args.guidance,
                "num_images": args.num_images,
                "seed": args.seed,
                "enable_safety_checker": args.safety
            }
        )
        
        print("Image generation successful!")
        print("Response:")
        print(prediction)
        
        # If the response contains image URLs, print them
        if prediction.outputs:
            print("\nGenerated image URLs:")
            for i, img_url in enumerate(prediction.outputs):
                print(f"Image {i+1}: {img_url}")
                
    except Exception as e:
        logging.exception(e)
        print(f"Error generating image: {e}")
        sys.exit(1)
    finally:
        await client.close()


async def main_async():
    parser = argparse.ArgumentParser(description="Generate images using Wavespeed AI API (Async)")
    parser.add_argument("--api-key", type=str, help="Your Wavespeed API key", 
                        default=os.environ.get("WAVESPEED_API_KEY"))
    parser.add_argument("--prompt", type=str, required=True, 
                        help="Text description of the desired image")
    parser.add_argument("--strength", type=float, default=0.6,
                        help="How much to transform the input image (0.0 to 1.0)")
    parser.add_argument("--size", type=str, default="1024*1024",
                        help="Image dimensions in format 'width*height'")
    parser.add_argument("--steps", type=int, default=28,
                        help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=5.0,
                        help="How closely to follow the prompt")
    parser.add_argument("--num-images", type=int, default=1,
                        help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed (-1 for random)")
    parser.add_argument("--safety", action="store_true", default=True,
                        help="Enable content safety filtering")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: API key is required. Provide it with --api-key or set WAVESPEED_API_KEY environment variable.")
        sys.exit(1)
    
    client = Wavespeed(api_key=args.api_key)
    
    # Run the async function
    await generate_image(client, args)


def main():
    """Entry point for the script."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()