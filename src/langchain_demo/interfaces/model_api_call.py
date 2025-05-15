from abc import ABC, abstractmethod

class ModelApiCall(ABC):
    """
    Abstract base class for making API calls to a model.
    """
    def __init__(self,
                 api_key: str = None,
                 model: str = None,
                 system_prompt: str = None,
                 messages: list[dict] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 512):
        
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.messages = messages if messages is not None else [] 
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def call(self,
             messages: list):
             
        """
        Make an API call to the model with the given prompt.

        Args:
            messages (list): A list of messages to be sent to the model.
                        
        Returns:
            str: The model's response.
        """
        pass