from openai import OpenAI
from langchain_demo.interfaces.model_api_call import ModelApiCall

class OpenAIAPICall(ModelApiCall):
    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o-mini",
        system_prompt: str = None,
        messages: list[dict] = None,
        temperature: float = 0.0,
        max_tokens: int = 512
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            messages=messages or [],
            temperature=temperature,
            max_tokens=max_tokens
        )

        self.client = OpenAI(api_key=self.api_key)

    def call(self, message: str):
        full_messages = []

        if self.system_prompt:
            full_messages.append({"role": "system", "content": self.system_prompt})

        full_messages.extend(self.messages)  
        full_messages.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        response_text = response.choices[0].message.content
        response_usage = {
            'input_tokens': response.usage.prompt_tokens,
            'output_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        }

        return response_text, response_usage
