from openai import OpenAI
import asyncio
from langchain_demo.interfaces.model_api_call import ModelApiCall

class OpenAIAPICall(ModelApiCall):
    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt: str,
        messages: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        self.client = OpenAI(api_key=self.api_key)

    def call(self, message: str):
        messages_to_send = []
        if self.system_prompt:
            messages_to_send.append({"role": "system", "content": self.system_prompt})
        messages_to_send.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages_to_send,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        response_text = response.choices[0].message.content
        role = response.choices[0].message.role
        response_usage = {
            'input_tokens': response.usage.prompt_tokens if response.usage else 0,
            'output_tokens': response.usage.completion_tokens if response.usage else 0,
            'total_tokens': response.usage.total_tokens if response.usage else 0
        }

        self._add_message('user', message)
        self._add_message(role, response_text)

        return response_text, response_usage
    
    async def async_call(self, message: str):
        """Asynchronous call to OpenAI API."""
        return await asyncio.to_thread(self.call, message)