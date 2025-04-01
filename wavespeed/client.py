import os
import httpx
from typing import Dict, Any
from urllib.parse import urljoin

from wavespeed.schemas.prediction import Prediction


class WaveSpeed:
    """
    A client for interacting with the Wavespeed AI API.
    """
    
    BASE_URL = "https://api.wavespeed.ai/api/v2/"
    
    def __init__(self, api_key: str):
        """
        Initialize the Wavespeed client.
        
        Args:
            api_key: Your Wavespeed API key
        """
        self.api_key = api_key
        if not api_key:
            api_key = os.environ.get("WAVESPEED_API_KEY", '')
        if not api_key:
            raise ValueError("API key is required.")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.async_client = httpx.AsyncClient(headers=self.headers)
        self.client = httpx.Client(headers=self.headers)
        self.poll_interval = float(os.environ.get("WAVESPEED_POLL_INTERVAL", 1)) # seconds
        self.timeout = int(os.environ.get("WAVESPEED_TIMEOUT", 60)) # seconds
    
    async def async_run(
        self,
        modelId: str,
        input: Dict[str, Any],
        **kwargs
    ) -> Prediction:
        """
        Generate an image using the Wavespeed AI API.
        
        Args:
            modelId: The ID of the model to use
            input: Input parameters for the model
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The API response as a dictionary
        """
        url = urljoin(self.BASE_URL, modelId)
        
        payload = input
        
        # Reset client if it's closed
        if self.async_client.is_closed:
            self.async_client = httpx.AsyncClient()
            
        response = await self.async_client.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout
        )
        
        # Raise an exception for HTTP errors
        response.raise_for_status()
        data = response.json()
        prediction = Prediction(**data['data'])
        prediction._client = self
        return await prediction.async_wait()
    
    def run(
        self,
        modelId: str,
        input: Dict[str, Any],
        **kwargs
    ) -> Prediction:
        """
        Generate an image using the Wavespeed AI API.
        
        Args:
            modelId: The ID of the model to use
            input: Input parameters for the model
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The API response as a dictionary
        """
        url = urljoin(self.BASE_URL, modelId)
        
        payload = input
        
        response = self.client.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout
        )

        # Raise an exception for HTTP errors
        response.raise_for_status()
        data = response.json()
        prediction = Prediction(**data['data'])
        prediction._client = self
        return prediction.wait()
    
    async def async_create(self, modelId: str, input: Dict[str, Any], **kwargs) -> Prediction:
        url = urljoin(self.BASE_URL, modelId)
        payload = input
        response = await self.async_client.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout
        )
        # Raise an exception for HTTP errors
        response.raise_for_status()
        data = response.json()
        prediction = Prediction(**data['data'])
        prediction._client = self
        return prediction
    
    def create(self, modelId: str, input: Dict[str, Any], **kwargs) -> Prediction:
        url = urljoin(self.BASE_URL, modelId)
        payload = input
        response = self.client.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout
        )
        # Raise an exception for HTTP errors
        response.raise_for_status()
        data = response.json()
        prediction = Prediction(**data['data'])
        prediction._client = self
        return prediction
        
    async def close(self):
        """Close the httpx client session."""
        self.client.close()
        await self.async_client.aclose()
        
    def __str__(self) -> str:
        """String representation of the client."""
        return f"WaveSpeed()"