from openai import AsyncOpenAI, AsyncAzureOpenAI
from config import config
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

def get_agent_model_name(key: str) -> str :
    config_key_prefix = f'llm.agents.{key}'
    model_name = config.get(f'{config_key_prefix}.model_name')

    if model_name == None:
        config_key_prefix = 'llm'
        # fallback to defaults
        model_name = config.get(f'{config_key_prefix}.model_name')
    return model_name

def get_agent_open_ai_client(key: str, timeout: float = None) -> AsyncOpenAI :
    config_key_prefix = f'llm.agents.{key}'
    base_url = config.get(f'{config_key_prefix}.base_url')

    if base_url == None:
        config_key_prefix = 'llm'
        # fallback to defaults
        base_url = config.get(f'{config_key_prefix}.base_url')
    
    key = f"{config_key_prefix.upper().replace('.', '_')}_API_KEY"
    api_key = config.get_api_key(key)
    api_version = config.get(f'{config_key_prefix}.api_version')
    

    if api_version == None:
        return AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
    else:
        return AsyncAzureOpenAI(
            api_version=api_version,
            azure_endpoint=base_url,
            api_key=api_key,
            timeout=timeout,
        )
