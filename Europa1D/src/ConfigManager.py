import json
import os
from typing import Dict, Any

class ConfigManager:
    """
    Singleton manager to load and store configuration from config.json.
    """
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            if config_path is None:
                # Default to src/config.json
                config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            cls._instance.load_config(config_path)
        return cls._instance

    def load_config(self, filepath: str):
        """Loads JSON and merges it with default internal dictionaries."""
        try:
            with open(filepath, 'r') as f:
                self._config = json.load(f)
        except Exception as e:
            print(f"Failed to load config from {filepath}: {e}")
            self._config = {}

    @classmethod
    def get(cls, section: str, key: str, default: Any = None) -> Any:
        """Fetch a value from the loaded JSON dict, fallback to default."""
        if cls._instance is None:
            cls()  # instantiate if not already
        
        sec = cls._instance._config.get(section, {})
        return sec.get(key, default)
    
    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Returns the full parsed dictionary configuration."""
        if cls._instance is None:
            cls()
        return cls._instance._config
