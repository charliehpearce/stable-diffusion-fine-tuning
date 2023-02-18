import os
from typing import List

import aiofiles


class LocalStorage:
    """
    Class to emulate cloud storage.
    """

    STORAGE_DIR = "app/mock_storage/"

    @staticmethod
    async def load_bytes_from_storage(storage_path: str) -> bytes:
        async with aiofiles.open(
            os.path.join(LocalStorage.STORAGE_DIR, storage_path), mode="rb"
        ) as f:
            contents = await f.read()
        return contents

    @staticmethod
    async def list_files_in_storage(storage_path: str) -> List[str]:
        files_in_dir = os.listdir(os.path.join(LocalStorage.STORAGE_DIR, storage_path))
        return files_in_dir

    @staticmethod
    async def save_bytes_to_storage(file_bytes: bytes, storage_path: str):
        async with aiofiles.open(
            os.path.join(LocalStorage.STORAGE_DIR, storage_path), mode="wb"
        ) as f:
            await f.write(file_bytes)
