from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx


class APIClient:
    def __init__(self, host: str, port: int) -> None:
        self.base_url = f"http://{host}:{port}"
        asyncio.run(wait_until_alive(self.base_url))

    async def _get_async(self, endpoint: str) -> httpx.Response:
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            return await client.get(endpoint, timeout=None)

    async def _post_async(self, endpoint: str, json: Any | None = None) -> httpx.Response:
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            return await client.post(endpoint, json=json, timeout=None)


async def wait_until_alive(url: str, max_time: float = 300, print_interval: float = 5) -> None:
    elapsed_time, last_print_time = 0.0, time.time()
    async with httpx.AsyncClient(base_url=url) as client:
        while elapsed_time < max_time:
            last_time = time.time()
            try:
                await client.get("/alive", timeout=1)
            except httpx.RequestError:
                await asyncio.sleep(1)
            else:
                print(f"Successfully connected to {url}!")
                return
            elapsed_time += last_time - time.time()
            if elapsed_time - last_print_time > print_interval:
                print(f"Waiting for server at {url} to be alive...")
                last_print_time = elapsed_time
    raise TimeoutError(f"Server at {url} did not respond within {max_time} seconds.")
