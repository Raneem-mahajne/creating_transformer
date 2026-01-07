"""
Configuration loader for transformer training
"""
import yaml
import os
from pathlib import Path
from IntegerStringGenerator import OddEvenIndexRule, EvenToOddTransitionRule, EvenRepeatLastOddRule, EvenAbsDiffRule, CopyModuloRule, SuccessorRule, ConditionalTransformRule, LookupPermutationRule, ParityBasedRule, EvenGreaterThan10Rule, TwoTokenParityRule, IntegerStringGenerator


def load_config(config_name: str):
    """
    Load configuration from a YAML file.
    
    Args:
        config_name: Name of the config file (without .yaml extension)
        
    Returns:
        Dictionary containing the configuration
    """
    config_path = Path("configs") / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['name', 'data', 'model', 'training']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    return config


def get_generator_from_config(config: dict) -> IntegerStringGenerator:
    """
    Create a generator instance from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        IntegerStringGenerator instance
    """
    data_config = config['data']
    generator_type = data_config['generator_type']
    min_value = data_config['min_value']
    max_value = data_config['max_value']
    
    if generator_type == "OddEvenIndexRule":
        return OddEvenIndexRule(min_value=min_value, max_value=max_value)
    elif generator_type == "EvenToOddTransitionRule":
        return EvenToOddTransitionRule(min_value=min_value, max_value=max_value)
    elif generator_type == "EvenRepeatLastOddRule":
        return EvenRepeatLastOddRule(min_value=min_value, max_value=max_value)
    elif generator_type == "EvenAbsDiffRule":
        return EvenAbsDiffRule(min_value=min_value, max_value=max_value)
    elif generator_type == "CopyModuloRule":
        period = data_config.get('period', 3)  # Default period of 3
        return CopyModuloRule(min_value=min_value, max_value=max_value, period=period)
    elif generator_type == "SuccessorRule":
        return SuccessorRule(min_value=min_value, max_value=max_value)
    elif generator_type == "ConditionalTransformRule":
        return ConditionalTransformRule(min_value=min_value, max_value=max_value)
    elif generator_type == "LookupPermutationRule":
        seed = data_config.get('seed', 42)
        return LookupPermutationRule(min_value=min_value, max_value=max_value, seed=seed)
    elif generator_type == "ParityBasedRule":
        return ParityBasedRule(min_value=min_value, max_value=max_value)
    elif generator_type == "EvenGreaterThan10Rule":
        return EvenGreaterThan10Rule(min_value=min_value, max_value=max_value)
    elif generator_type == "TwoTokenParityRule":
        return TwoTokenParityRule(min_value=min_value, max_value=max_value)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")

