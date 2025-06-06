from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from config import config
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def load_prompt(prompt_key: str) -> Optional[str]:
    """Loads the raw text content of a prompt file specified in the config."""
    config_key_path = f"prompts.{prompt_key}"
    
    logger.debug(f"Attempting to load prompt using config key: '{config_key_path}'")
    prompt_file_rel_path_str = config.get(config_key_path)

    if not prompt_file_rel_path_str:
        logger.error(f"Prompt key path '{config_key_path}' not found in configuration.")
        return None

    logger.debug(f"Relative path found in config: '{prompt_file_rel_path_str}'")
    logger.debug(f"Project root directory from config: '{config.root_dir}'")

    try:
        # Build absolute path
        full_path = (config.root_dir / Path(prompt_file_rel_path_str)).resolve()
        logger.debug(f"Calculated absolute path to check: '{full_path}'")

        # Check existence and type before reading
        path_exists = full_path.exists()
        path_is_file = full_path.is_file()
        logger.debug(f"Check result - Path exists: {path_exists}, Path is file: {path_is_file}")

        if path_exists and path_is_file:
            content = full_path.read_text(encoding="utf-8")
            logger.info(f"Successfully loaded prompt content for key '{prompt_key}' from {full_path}")
            return content
        
        elif path_exists and not path_is_file:
            logger.error(f"Path exists but is not a file: {full_path}")
            return None
        
        else:
            logger.error(f"Prompt file not found at calculated path: {full_path}")
            return None
    
    except TypeError as e:
        logger.error(f"Invalid path format for prompt key '{prompt_key}' in config: {prompt_file_rel_path_str}. Error: {e}")
        return None
    
    except Exception as e:
        # Permission errors
        logger.exception(f"Error reading prompt file {full_path} for key '{prompt_key}': {e}")
        return None


def create_react_prompt_template() -> Optional[ChatPromptTemplate]:
    """
    Creates the ChatPromptTemplate for the ReAct Supervisor Agent.
    Loads the system prompt using the 'react_supervisor' key from config.
    """
    logger.debug("Attempting to create ReAct prompt template...")
    system_prompt_content = load_prompt('react_supervisor')
    if system_prompt_content is None:
        logger.error("Failed to load system prompt content using key 'react_supervisor'. Cannot create template.")
        return None

    try:
        system_message = SystemMessagePromptTemplate.from_template(system_prompt_content)
        prompt = ChatPromptTemplate.from_messages([
                system_message,
                MessagesPlaceholder(variable_name="messages"),
            ])
        logger.info("ReAct prompt template created successfully using key 'react_supervisor'.")
        return prompt
    except Exception as e:
        logger.exception(f"Failed to create ChatPromptTemplate from loaded content: {e}")
        return None


def format_prompt(prompt_name: str, variables: dict) -> str:
    """
    Loads a prompt by its name and then replaces all placeholders
    defined in the variables dictionary.
    Placeholders in the prompt template should be like {{key_name}}.
    """
    template_string = load_prompt(prompt_name)
    for key, value in variables.items():
        placeholder = "{{" + str(key) + "}}"
        template_string = template_string.replace(placeholder, str(value))
    return template_string