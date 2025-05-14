from abc import ABC, abstractmethod

class TextSplitterInterface(ABC):
    """
    Interface for text splitting classes.
    """

    @abstractmethod
    def split_text(self, text: str, chunk_size: int, chunk_overlap: int, separators: list) -> list:
        """
        Splits the input text into smaller chunks.

        Args:
            text (str): The text to be split.
            chunk_size (int): The size of each chunk.
            chunk_overlap (int): The number of overlapping characters between chunks.
            separators (list): A list of characters or strings to use as separators.

        Returns:
            list: A list of text chunks.
        """
        pass