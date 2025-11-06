"""
파일 해시 관리 클래스
"""

import hashlib
from typing import Dict

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class FileHashManager:
    """
    파일 해시 관리 클래스
    
    Attributes:
        hash_algorithm: 해시 알고리즘 (기본값: sha256)
    """

    def __init__(self, hash_algorithm: str = "sha256"):
        """
        Args:
            hash_algorithm: 해시 알고리즘 이름
        """
        self.hash_algorithm = hash_algorithm

    def calculate_file_hash(self, file_path: str) -> str:
        """
        파일의 해시 값을 계산합니다.

        Args:
            file_path: 파일 경로

        Returns:
            str: 파일 해시 값
        """
        hash_func = hashlib.new(self.hash_algorithm)
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def is_file_modified(self, file_path: str, stored_hash: str) -> bool:
        """
        파일이 수정되었는지 확인합니다.

        Args:
            file_path: 파일 경로
            stored_hash: 저장된 해시 값

        Returns:
            bool: 파일이 수정되었으면 True, 그렇지 않으면 False
        """
        current_hash = self.calculate_file_hash(file_path)
        return current_hash != stored_hash
