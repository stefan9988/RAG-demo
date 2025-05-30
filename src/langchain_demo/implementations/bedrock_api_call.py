from langchain_demo.interfaces.model_api_call import ModelApiCall
import json
import boto3
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class BedrockAPICall(ModelApiCall):
    def __init__(
        self,
        model: str,
        system_prompt: str,
        api_key: str | None = None,
        messages: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        self.aws_access_key_id = kwargs.get('aws_access_key_id')
        self.aws_secret_access_key = kwargs.get('aws_secret_access_key')
        self.region = kwargs.get('region', 'us-east-1')
        
        self.session_kwargs = {
            'aws_access_key_id': self.aws_access_key_id,
            'aws_secret_access_key': self.aws_secret_access_key,
            'region_name': self.region
        }

        self.session = boto3.Session(**self.session_kwargs)
        self.bedrock_runtime = self.session.client("bedrock-runtime")
        
    def _get_body(self, full_messages: list[dict]):
        """Constructs the body for the Bedrock API call."""
        if 'anthropic' in self.model:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "messages": full_messages,
                "system": self.system_prompt,
                "temperature": self.temperature,            
            })
        elif 'meta' in self.model:
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{self.system_prompt}<|eot_id|>"
            for msg in full_messages:
                role_tag = "user" if msg["role"] == "user" else "assistant"
                prompt += f"<|start_header_id|>{role_tag}<|end_header_id|>\n{msg['content']}<|eot_id|>"
            prompt += "<|start_header_id|>assistant<|end_header_id|>"

            body = json.dumps({
                "prompt": prompt,
                "max_gen_len": self.max_tokens,
                "temperature": self.temperature,                
            })
            
        elif 'mistral' in self.model:
            has_system_prompt = any(msg['role'] == 'system' for msg in full_messages)
            if not has_system_prompt:
                system_msg = {
                        "role": "system",
                        "content": self.system_prompt
                    }
                full_messages.insert(0, system_msg)
                
            body = json.dumps({  
                "messages": full_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,                
            })

        else:
            body = None
            raise ValueError(f"""Cannot create body for model {self.model}. 
                             Please implement this method for the specific model.""")
        return body

    def _format_response(self, response):
        """Formats the response based on the model provider."""
        if "anthropic" in self.model:                
            return response['content'][0]['text'], response['usage'] 
        elif "mistral" in self.model:
            return response['choices'][0]['message']['content'], None
        elif "meta" in self.model:
            usage = {
                'input_tokens': response['prompt_token_count'],
                'output_tokens': response['generation_token_count']
                    }
            return response['generation'], usage
        else:
            raise ValueError("Model provider not supported")
    
    def call(self, message: str):
        self._add_message("user", message)
        
        try:
            response = self.bedrock_runtime.invoke_model(
                        modelId=self.model,
                        contentType="application/json",
                        accept="application/json",
                        body=self._get_body(self.messages),
                    )      
                    
            response = json.loads(response['body'].read())
            response_text, response_usage = self._format_response(response)
            
        except Exception as e:
            logger.error(f"Error calling Bedrock API: {e}")
            response_text, response_usage = None, None
        
        return response_text, response_usage
        
    async def async_call(self, message: str):
        """Asynchronous call to the Bedrock API."""
        return await asyncio.to_thread(self.call, message)

            

        
        