from abc import ABC, abstractmethod

class ModelApiCall(ABC):
    """
    Abstract base class for making API calls to a model.
    """
    def __init__(self,
                 model: str,
                 system_prompt: str,
                 api_key: str | None = None,
                 messages: list[dict] | None = None,
                 temperature: float = 0.0,
                 max_tokens: int = 512,
                 **kwargs):
        
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.messages = messages if messages is not None else [] 
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.additional_params = kwargs

    @abstractmethod
    def call(self,
             message: str) -> tuple[str, dict]:
             
        """
        Make an API call to the model with the given prompt.

        Args:
            message (str): A message to be sent to the model.
                        
        Returns:
            str: The model's response.
        """
        pass
    
    @abstractmethod
    async def async_call(self,
                 message: str) -> tuple[str, dict]:
                 
        """
        Asynchronously make an API call to the model with the given prompt.

        Args:
            message (str): A message to be sent to the model.
                        
        Returns:
            str: The model's response.
        """
        pass

    def _add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})