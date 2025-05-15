from langchain_demo.interfaces.model_api_call import ModelApiCall
import json
import boto3

class BedrockAPICall(ModelApiCall):
    def __init__(
        self,
        api_key: str = None,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        region: str = "us-east-1",
        model: str = "anthropic.claude-3-haiku-20240307-v1:0",
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
        self.session_kwargs = {
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
            'region_name': region
        }
        self.session = boto3.Session(**self.session_kwargs)
        self.bedrock_runtime = self.session.client("bedrock-runtime")

    def call(self, message: str):
        full_messages = []
        full_messages.extend(self.messages)  
        full_messages.append({"role": "user", "content": message})
        
        body = json.dumps({   
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "messages": full_messages,
            "system": self.system_prompt,
            "temperature": self.temperature,            
        })
        response = self.bedrock_runtime.invoke_model(
                    modelId=self.model,
                    contentType="application/json",
                    accept="application/json",
                    body=body,
                )      
                
        response = json.loads(response['body'].read())
        
        return response['content'][0]['text'], response['usage'] 