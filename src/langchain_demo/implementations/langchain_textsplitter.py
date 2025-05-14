from langchain_demo.interfaces.text_splitter_interface import TextSplitterInterface
from langchain.text_splitter import RecursiveCharacterTextSplitter

class LangchainTextSplitter(TextSplitterInterface):
    """
    A class that implements the TextSplitterInterface using Langchain's text splitting capabilities.
    """

    def split_text(self, text: str, 
                   chunk_size: int = 500, 
                   chunk_overlap: int = 50, 
                   separators: list = None) -> list:
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
        
        return text_splitter.split_documents(text)
    