"""
Tests for the Wavespeed client.
"""

import pytest
import respx
from httpx import Response
from datetime import datetime

from wavespeed.client import WaveSpeed
from wavespeed.schemas.prediction import Prediction, PredictionUrls


@pytest.fixture
def client():
    """Create a test client with a dummy API key."""
    return WaveSpeed(api_key="test_api_key")


@pytest.fixture
async def async_client():
    """Create a test client with a dummy API key and clean up after test."""
    client = WaveSpeed(api_key="test_api_key")
    yield client
    await client.close()


@pytest.fixture
def mock_prediction_response():
    """Create a mock prediction response."""
    return {
        "code": 200,
        "message": "Success",
        "data": {
            "id": "test_prediction_id",
            "model": "wavespeed-ai/flux-dev",
            "input": {
                "prompt": "A test prompt",
                "size": "512*512",
                "num_inference_steps": 20,
                "guidance_scale": 7.0,
                "enable_safety_checker": True,
            },
            "outputs": ["https://example.com/generated_image.jpg"],
            "urls": {
                "get": "https://api.wavespeed.ai/api/v2/predictions/test_prediction_id/result"
            },
            "has_nsfw_contents": [False],
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "error": "",
            "executionTime": 1000
        }
    }


@pytest.fixture
def mock_prediction_in_progress_response():
    """Create a mock prediction in progress response."""
    return {
        "code": 200,
        "message": "Success",
        "data": {
            "id": "test_prediction_id",
            "model": "wavespeed-ai/flux-dev",
            "input": {
                "prompt": "A test prompt",
                "size": "512*512",
                "num_inference_steps": 20,
                "guidance_scale": 7.0,
                "enable_safety_checker": True,
            },
            "outputs": [],
            "urls": {
                "get": "https://api.wavespeed.ai/api/v2/predictions/test_prediction_id/result"
            },
            "has_nsfw_contents": [],
            "status": "processing",
            "created_at": datetime.now().isoformat(),
            "error": "",
            "executionTime": 0
        }
    }


@pytest.fixture
def mock_prediction_completed_response():
    """Create a mock prediction completed response."""
    return {
        "code": 200,
        "message": "Success",
        "data": {
            "id": "test_prediction_id",
            "model": "wavespeed-ai/flux-dev",
            "input": {
                "prompt": "A test prompt",
                "size": "512*512",
                "num_inference_steps": 20,
                "guidance_scale": 7.0,
                "enable_safety_checker": True,
            },
            "outputs": ["https://example.com/generated_image.jpg"],
            "urls": {
                "get": "https://api.wavespeed.ai/api/v2/predictions/test_prediction_id/result"
            },
            "has_nsfw_contents": [False],
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "error": "",
            "executionTime": 1000
        }
    }


@respx.mock
def test_run(client, mock_prediction_response, mock_prediction_completed_response):
    """Test the run method."""
    # Mock the initial API response
    respx.post("https://api.wavespeed.ai/api/v2/wavespeed-ai/flux-dev").mock(
        return_value=Response(200, json=mock_prediction_response)
    )
    
    # Mock the status check response
    respx.get("https://api.wavespeed.ai/api/v2/predictions/test_prediction_id/result").mock(
        return_value=Response(200, json=mock_prediction_completed_response)
    )

    # Call the run method
    prediction = client.run(
        modelId="wavespeed-ai/flux-dev",
        input={
            "prompt": "A test prompt",
            "size": "512*512",
            "num_inference_steps": 20,
            "guidance_scale": 7.0,
        },
    )

    # Verify the response
    assert isinstance(prediction, Prediction)
    assert prediction.id == "test_prediction_id"
    assert prediction.model == "wavespeed-ai/flux-dev"
    assert prediction.status == "completed"
    assert len(prediction.outputs) == 1
    assert prediction.outputs[0] == "https://example.com/generated_image.jpg"
    assert prediction.has_nsfw_contents == [False]


@respx.mock
@pytest.mark.asyncio
async def test_async_run(async_client, mock_prediction_response, mock_prediction_completed_response):
    """Test the async_run method."""
    # Mock the initial API response
    respx.post("https://api.wavespeed.ai/api/v2/wavespeed-ai/flux-dev").mock(
        return_value=Response(200, json=mock_prediction_response)
    )
    
    # Mock the status check response
    respx.get("https://api.wavespeed.ai/api/v2/predictions/test_prediction_id/result").mock(
        return_value=Response(200, json=mock_prediction_completed_response)
    )

    # Call the async_run method
    prediction = await async_client.async_run(
        modelId="wavespeed-ai/flux-dev",
        input={
            "prompt": "A test prompt",
            "size": "512*512",
            "num_inference_steps": 20,
            "guidance_scale": 7.0,
        },
    )

    # Verify the response
    assert isinstance(prediction, Prediction)
    assert prediction.id == "test_prediction_id"
    assert prediction.model == "wavespeed-ai/flux-dev"
    assert prediction.status == "completed"
    assert len(prediction.outputs) == 1
    assert prediction.outputs[0] == "https://example.com/generated_image.jpg"
    assert prediction.has_nsfw_contents == [False]


@respx.mock
def test_create(client, mock_prediction_response):
    """Test the create method."""
    # Mock the API response
    respx.post("https://api.wavespeed.ai/api/v2/wavespeed-ai/flux-dev").mock(
        return_value=Response(200, json=mock_prediction_response)
    )

    # Call the create method
    prediction = client.create(
        modelId="wavespeed-ai/flux-dev",
        input={
            "prompt": "A test prompt",
            "size": "512*512",
            "num_inference_steps": 20,
            "guidance_scale": 7.0,
        },
    )

    # Verify the response
    assert isinstance(prediction, Prediction)
    assert prediction.id == "test_prediction_id"
    assert prediction.model == "wavespeed-ai/flux-dev"
    assert prediction.status == "completed"
    assert len(prediction.outputs) == 1
    assert prediction.outputs[0] == "https://example.com/generated_image.jpg"


@respx.mock
@pytest.mark.asyncio
async def test_async_create(async_client, mock_prediction_response):
    """Test the async_create method."""
    # Mock the API response
    respx.post("https://api.wavespeed.ai/api/v2/wavespeed-ai/flux-dev").mock(
        return_value=Response(200, json=mock_prediction_response)
    )

    # Call the async_create method
    prediction = await async_client.async_create(
        modelId="wavespeed-ai/flux-dev",
        input={
            "prompt": "A test prompt",
            "size": "512*512",
            "num_inference_steps": 20,
            "guidance_scale": 7.0,
        },
    )

    # Verify the response
    assert isinstance(prediction, Prediction)
    assert prediction.id == "test_prediction_id"
    assert prediction.model == "wavespeed-ai/flux-dev"
    assert prediction.status == "completed"
    assert len(prediction.outputs) == 1
    assert prediction.outputs[0] == "https://example.com/generated_image.jpg"


@respx.mock
def test_prediction_wait(client, mock_prediction_in_progress_response, mock_prediction_completed_response):
    """Test the prediction wait method."""
    # Mock the initial API response
    respx.post("https://api.wavespeed.ai/api/v2/wavespeed-ai/flux-dev").mock(
        return_value=Response(200, json=mock_prediction_in_progress_response)
    )
    
    # Mock the status check response
    respx.get("https://api.wavespeed.ai/api/v2/predictions/test_prediction_id/result").mock(
        return_value=Response(200, json=mock_prediction_completed_response)
    )

    # Create a prediction
    prediction = client.create(
        modelId="wavespeed-ai/flux-dev",
        input={
            "prompt": "A test prompt",
            "size": "512*512",
            "num_inference_steps": 20,
            "guidance_scale": 7.0,
        },
    )
    
    # Set a short poll interval for testing
    client.poll_interval = 0.01
    
    # Wait for the prediction to complete
    result = prediction.wait()
    
    # Verify the response
    assert isinstance(result, Prediction)
    assert result.id == "test_prediction_id"
    assert result.status == "completed"
    assert len(result.outputs) == 1
    assert result.outputs[0] == "https://example.com/generated_image.jpg"


@respx.mock
@pytest.mark.asyncio
async def test_prediction_async_wait(async_client, mock_prediction_in_progress_response, mock_prediction_completed_response):
    """Test the prediction async_wait method."""
    # Mock the initial API response
    respx.post("https://api.wavespeed.ai/api/v2/wavespeed-ai/flux-dev").mock(
        return_value=Response(200, json=mock_prediction_in_progress_response)
    )
    
    # Mock the status check response
    respx.get("https://api.wavespeed.ai/api/v2/predictions/test_prediction_id/result").mock(
        return_value=Response(200, json=mock_prediction_completed_response)
    )

    # Create a prediction
    prediction = await async_client.async_create(
        modelId="wavespeed-ai/flux-dev",
        input={
            "prompt": "A test prompt",
            "size": "512*512",
            "num_inference_steps": 20,
            "guidance_scale": 7.0,
        },
    )
    
    # Set a short poll interval for testing
    async_client.poll_interval = 0.01
    
    # Wait for the prediction to complete
    result = await prediction.async_wait()
    
    # Verify the response
    assert isinstance(result, Prediction)
    assert result.id == "test_prediction_id"
    assert result.status == "completed"
    assert len(result.outputs) == 1
    assert result.outputs[0] == "https://example.com/generated_image.jpg"
