import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import pydantic
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wavespeed.client import WaveSpeed

class PredictionUrls(BaseModel):
    """URLs associated with a prediction."""
    get: str


class Prediction(BaseModel):
    """Response from a prediction API call."""
    _client: "WaveSpeed" = pydantic.PrivateAttr()

    id: str
    model: str
    input: Dict[str, Any] | None = None
    outputs: List[str]
    urls: PredictionUrls
    has_nsfw_contents: List[bool]
    status: str
    created_at: datetime
    error: str = ""
    executionTime: int

    def wait(self) -> "Prediction":
        while self.status not in ['completed', 'failed']:
            time.sleep(self._client.poll_interval)
            print('Waiting for prediction to complete: ', self.urls.get, type(self.urls.get))
            response = self._client.client.get(self.urls.get)
            response.raise_for_status()
            data = response.json()['data']
            self._update_from_dict(data)
        return self
    
    async def async_wait(self) -> "Prediction":
        while self.status not in ['completed', 'failed']:
            await asyncio.sleep(self._client.poll_interval)
            response = await self._client.async_client.get(self.urls.get)
            response.raise_for_status()
            data = response.json()['data']
            self._update_from_dict(data)
        return self
    
    async def async_reload(self) -> "Prediction":
        response = await self._client.async_client.get(self.urls.get)
        response.raise_for_status()
        data = response.json()['data']
        self._update_from_dict(data)
        return self
    
    def reload(self) -> "Prediction":
        response = self._client.client.get(self.urls.get)
        response.raise_for_status()
        data = response.json()['data']
        self._update_from_dict(data)
        return self
    
    def _update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update the object from a dictionary, handling nested objects properly."""
        for key, value in data.items():
            if key == 'urls' and isinstance(value, dict):
                self.urls = PredictionUrls(**value)
            elif hasattr(self, key):
                setattr(self, key, value)

class PredictionResponse(BaseModel):
    code: int
    message: str
    data: Prediction