import yaml
import os
from dotenv import load_dotenv
from pathlib import Path
import logging
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)

class Singleton(type):
    _instances: Dict[type, Any] = {}
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Config(metaclass=Singleton):
    _config_data: Optional[Dict[str, Any]] = None
    _root_dir: Path = Path(__file__).parent.parent

    def __init__(self, config_path: str = "config/config.yaml", env_file: str = ".env"):
        if self._config_data is None:
            self._load_env(env_file)
            self._load_config(config_path)
            if self._config_data is None:
                 logger.critical("Configuration loading failed. Application might not work correctly.")
                 self._config_data = {}
            else:
                 logger.info("Configuration loaded successfully.")

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def _load_env(self, env_file: str):
        try:
            full_env_path = self._root_dir / env_file
            if full_env_path.exists():
                loaded = load_dotenv(dotenv_path=full_env_path, override=True)
                if loaded:
                     logger.debug(f"Loaded environment variables from: {full_env_path}")
                else:
                     logger.warning(f".env file found at {full_env_path}, but no variables were loaded.")
            else:
                logger.warning(f".env file not found at {full_env_path}. Relying on system environment variables.")
        except Exception as e:
            logger.error(f"Error loading .env file from {env_file}: {e}", exc_info=True)


    def _load_config(self, config_path: str):        
        full_config_path = self._root_dir / config_path
        try:
            with open(full_config_path, 'r', encoding='utf-8') as stream:
                self._config_data = yaml.safe_load(stream)
                if self._config_data is None:
                     logger.warning(f"Configuration file {full_config_path} is empty or invalid.")
                     self._config_data = {}
                else:
                     logger.debug(f"Loaded configuration from: {full_config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {full_config_path}")
            self._config_data = None
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file {full_config_path}: {e}", exc_info=True)
            self._config_data = None
        except Exception as e:
            logger.error(f"Unexpected error loading config {full_config_path}: {e}", exc_info=True)
            self._config_data = None

    def get(self, key_path: str, default: Any = None) -> Any:
        if self._config_data is None:
            logger.warning("Config accessed before loading or load failed.")
            return default

        keys = key_path.split('.')
        value = self._config_data
        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value[key]
                else:
                    logger.debug(f"Config path '{key_path}' invalid structure at '{key}'. Value is not a dictionary.")
                    return default
            return value
        except KeyError:
            logger.debug(f"Configuration key not found: '{key_path}'. Returning default.")
            return default
        except Exception as e:
            logger.error(f"Error retrieving config key '{key_path}': {e}", exc_info=True)
            return default

    def get_api_key(self, key_name: str) -> Optional[str]:
        api_key = os.getenv(key_name)
        return api_key

    def get_prompt_path(self, prompt_key: str) -> Optional[Path]:
        prompt_config_path = f"prompts.{prompt_key}.system"
        prompt_file_rel_path = self.get(prompt_config_path)

        if not prompt_file_rel_path:
            logger.error(f"Prompt key path '{prompt_config_path}' not found in configuration.")
            return None

        try:
            full_path = self.root_dir / Path(prompt_file_rel_path)
            if full_path.exists() and full_path.is_file():
                return full_path
            else:
                logger.error(f"Prompt file not found or is not a file at configured path: {full_path}")
                return None
        except TypeError as e:
            logger.error(f"Invalid path format for prompt '{prompt_key}' in config: {prompt_file_rel_path}. Error: {e}")
            return None

    def set_override(self, key_path: str, value: Any) -> bool:
        """
        Overrides a configuration value in the loaded config using dot notation.
        This modifies the in-memory config object.
        True if successful, False if not.
        """
        if self._config_data is None:
            logger.error("Cannot override config: Configuration not loaded.")
            return False

        keys = key_path.split('.')
        data_ptr = self._config_data
        try:
            for key in keys[:-1]:
                if key not in data_ptr or not isinstance(data_ptr[key], dict):
                    data_ptr[key] = {}
                    logger.debug(f"Created intermediate config dict for key: {key}")
                data_ptr = data_ptr[key]

            final_key = keys[-1]
            if isinstance(data_ptr, dict):
                old_value = data_ptr.get(final_key)
                data_ptr[final_key] = value
                logger.info(f"Config override set: '{key_path}' = {value} (was: {old_value})")
                return True
            else:
                logger.error(f"Cannot set config override: Path '{key_path}' leads to a non-dictionary element before final key.")
                return False
        except Exception as e:
            logger.error(f"Error setting config override for '{key_path}': {e}", exc_info=True)
            return False

try:
    config = Config()
except Exception as e:
    logger.critical(f"Failed to initialize Config singleton: {e}", exc_info=True)
    class DummyConfig:
         def get(self, key_path: str, default: Any = None) -> Any: return default
         def get_api_key(self, key_name: str) -> Optional[str]: return None
         def get_prompt_path(self, prompt_key: str) -> Optional[Path]: return None
         def set_override(self, key_path: str, value: Any) -> bool: return False
         @property
         def root_dir(self) -> Path: return Path(".")
    config = DummyConfig()