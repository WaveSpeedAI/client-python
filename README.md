# Wavespeed Python Client

A Python client for interacting with the Wavespeed AI API.

## Installation

```bash
pip install wavespeed
```

## Usage

### Synchronous Image Generation

```python
from wavespeed import Wavespeed

# Initialize the client with your API key (or set WAVESPEED_API_KEY environment variable)
client = Wavespeed(api_key="YOUR_API_KEY")

# Generate an image and wait for the result
prediction = client.run(
    modelId="wavespeed-ai/flux-dev",
    input={
        "prompt": "A futuristic cityscape with flying cars and neon lights",
        "size": "1024*1024",
        "num_inference_steps": 28,
        "guidance_scale": 5.0,
        "num_images": 1,
        "seed": -1,
        "enable_safety_checker": True
    }
)

# Print the generated image URLs
for i, img_url in enumerate(prediction.outputs):
    print(f"Image {i+1}: {img_url}")
```

### Asynchronous Image Generation

```python
import asyncio
from wavespeed import Wavespeed

async def generate_image():
    # Initialize the client with your API key (or set WAVESPEED_API_KEY environment variable)
    client = Wavespeed(api_key="YOUR_API_KEY")
    
    try:
        # Generate an image and wait for the result
        prediction = await client.async_run(
            modelId="wavespeed-ai/flux-dev",
            input={
                "prompt": "A futuristic cityscape with flying cars and neon lights",
                "size": "1024*1024",
                "num_inference_steps": 28,
                "guidance_scale": 5.0,
                "num_images": 1,
                "seed": -1,
                "enable_safety_checker": True
            }
        )
        
        # Print the generated image URLs
        for i, img_url in enumerate(prediction.outputs):
            print(f"Image {i+1}: {img_url}")
    finally:
        # Always close the client when done
        await client.close()

# Run the async function
asyncio.run(generate_image())
```

### Non-blocking Image Generation

You can also create a prediction without waiting for it to complete:

```python
from wavespeed import Wavespeed

# Initialize the client with your API key (or set WAVESPEED_API_KEY environment variable)
client = Wavespeed(api_key="YOUR_API_KEY")

# Create a prediction without waiting
prediction = client.create(
    modelId="wavespeed-ai/flux-dev",
    input={
        "prompt": "A futuristic cityscape with flying cars and neon lights",
        "size": "1024*1024",
        "num_inference_steps": 28,
        "guidance_scale": 5.0,
        "num_images": 1,
        "seed": -1,
        "enable_safety_checker": True
    }
)

print(f"Prediction created with ID: {prediction.id}")
print(f"Initial status: {prediction.status}")

# Later, you can wait for the prediction to complete
result = prediction.wait()
print(f"Final status: {result.status}")

# Print the generated image URLs
for i, img_url in enumerate(result.outputs):
    print(f"Image {i+1}: {img_url}")
```

## Command Line Examples

The package includes several example scripts that you can use to generate images:

### Basic Image Generation

```bash
# Set your API key as an environment variable
export WAVESPEED_API_KEY="your_api_key_here"

# Run the example script
python examples/generate_image.py --prompt "A futuristic cityscape with flying cars and neon lights"
```

### Asynchronous Image Generation

```bash
# Run the async example script
python examples/async_generate_image.py --prompt "A futuristic cityscape with flying cars and neon lights"
```

### Non-blocking Image Generation

```bash
# Create a prediction and poll for status
python examples/create_generate_image.py --prompt "A futuristic cityscape with flying cars and neon lights"
```

### Command Line Options

```
--prompt TEXT         Text description of the desired image (required)
--strength FLOAT      How much to transform the input image (0.0 to 1.0)
--size WIDTHxHEIGHT   Image dimensions (default: 1024*1024)
--steps INT           Number of inference steps (default: 28)
--guidance FLOAT      How closely to follow the prompt (default: 5.0)
--num-images INT      Number of images to generate (default: 1)
--seed INT            Random seed (-1 for random)
--safety              Enable content safety filtering
```

## API Reference

### Wavespeed Client

```python
Wavespeed(api_key)
```

#### Parameters:

- `api_key` (str): Your Wavespeed API key

### Methods

#### run

```python
run(modelId, input, **kwargs) -> Prediction
```

Generate an image and wait for the result.

#### async_run

```python
async_run(modelId, input, **kwargs) -> Prediction
```

Asynchronously generate an image and wait for the result.

#### create

```python
create(modelId, input, **kwargs) -> Prediction
```

Create a prediction without waiting for it to complete.

#### async_create

```python
async_create(modelId, input, **kwargs) -> Prediction
```

Asynchronously create a prediction without waiting for it to complete.

### Prediction Model

The `Prediction` object contains information about an image generation job:

```python
prediction.id           # Unique ID of the prediction
prediction.model        # Model ID used for the prediction
prediction.status       # Status of the prediction (processing, completed, error)
prediction.outputs      # List of output image URLs
prediction.input        # Input parameters used for the prediction
prediction.created_at   # Creation timestamp
prediction.error        # Error message (if any)
prediction.executionTime # Time taken to execute the prediction in milliseconds
```

#### Methods

```python
prediction.wait() -> Prediction  # Wait for the prediction to complete
prediction.reload() -> Prediction  # Reload the prediction status

# Async versions
await prediction.async_wait() -> Prediction
await prediction.async_reload() -> Prediction
```

## Environment Variables

- `WAVESPEED_API_KEY`: Your Wavespeed API key
- `WAVESPEED_POLL_INTERVAL`: Interval in seconds for polling prediction status (default: 1)
- `WAVESPEED_TIMEOUT`: Timeout in seconds for API requests (default: 60)

## License

MIT
