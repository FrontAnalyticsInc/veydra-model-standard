"""Veydra Model Standard - Base classes for VMS-compliant models."""

import numpy as np
from scipy.integrate import solve_ivp
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

class Submodel:
    """
    Base class for VMS-compliant submodels.
    
    Handles all the boilerplate for variable management, leaving submodels
    to focus only on their specific physics/logic implementation.
    
    Usage:
        1. Define VARIABLES dict at module level with complete info
        2. Inherit from this class  
        3. Implement calculate_derivatives(self, all_stocks) method
        4. Add your physics methods - everything else is handled automatically!
    
    The base class provides:
        - get_variables() - auto-detects VARIABLES from module
        - get_defaults() - extracts default values
        - get_validation() - extracts min/max/step/type rules  
        - get_definitions() - extracts name/units/description/category
        - get_value() - parameter access with caching and defaults
        
    Subclasses only need to implement:
        - calculate_derivatives() - required by VMS standard
        - your physics methods - the actual model logic
        
    That's it! No boilerplate needed.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """Initialize with parameter dictionary."""
        self.params = params
        self._variable_cache = {}
    
    def get_value(self, key: str) -> Any:
        """Get parameter value with default fallback and caching."""
        if key not in self._variable_cache:
            variables = self.__class__.get_variables()
            default = variables.get(key, {}).get('default', None)
            if default is None:
                raise KeyError(f"Unknown parameter: {key}")
            self._variable_cache[key] = self.params.get(key, default)
        return self._variable_cache[key]
    
    @classmethod
    def get_variables(cls) -> Dict[str, Any]:
        """Return this submodel's variables.
        
        Default implementation tries to find VARIABLES in the module.
        Subclasses can override this method if they use a different pattern.
        """
        # Try to auto-detect VARIABLES from the module
        import sys
        module = sys.modules[cls.__module__]
        if hasattr(module, 'VARIABLES'):
            return getattr(module, 'VARIABLES').copy()
        else:
            raise NotImplementedError(
                f"Subclass {cls.__name__} must implement get_variables() "
                f"or define VARIABLES dict in module {cls.__module__}"
            )
    
    @classmethod  
    def get_defaults(cls) -> Dict[str, Any]:
        """Extract just the default values."""
        return {key: var.get('default') for key, var in cls.get_variables().items()}
    
    @classmethod
    def get_validation(cls) -> Dict[str, Any]:
        """Extract just the validation rules."""
        validation = {}
        for key, var in cls.get_variables().items():
            val_rules = {rule: var[rule] for rule in ['min', 'max', 'step', 'type'] if rule in var}
            if val_rules:
                validation[key] = val_rules
        return validation
    
    @classmethod
    def get_definitions(cls) -> Dict[str, Any]:
        """Extract just the human-readable definitions."""
        definitions = {}
        for key, var in cls.get_variables().items():
            def_data = {field: var[field] for field in ['name', 'units', 'description', 'category'] if field in var}
            if def_data:
                definitions[key] = def_data
        return definitions

    def calculate_derivatives(self, all_stocks: Dict[str, float]) -> List[float]:
        """Calculate derivatives for this submodel.
        
        Args:
            all_stocks: Dictionary mapping stock names to current values
            
        Returns:
            List of derivative values for this submodel's stocks
        """
        raise NotImplementedError("Submodels must implement calculate_derivatives()")


class VeydraModelStandard:
    """
    Lightweight base class for Veydra system dynamics models.
    
    Provides minimal structure - models should implement their own explicit logic.
    """
    def __init__(self, params: Dict[str, Any]):
        """Initialize with parameter dictionary."""
        self.params = params
    
    def _clean_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clean parameter dictionary by extracting 'value' if needed."""
        cleaned = {}
        for key, val in params.items():
            if isinstance(val, dict) and 'value' in val:
                cleaned[key] = val['value']
            else:
                cleaned[key] = val
        return cleaned
    
    def _resolve_parameters(self, user_params: Dict[str, Any], all_variables: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters with defaults and validation - explicit and clear."""
        resolved = {}
        
        # Start with defaults explicitly
        for key, var_def in all_variables.items():
            resolved[key] = var_def.get('default', 0.0)
        
        # Override with user parameters, applying validation explicitly
        for key, value in user_params.items():
            if key in all_variables:
                var_def = all_variables[key]
                
                # Apply type-specific validation explicitly
                if var_def.get('type') == 'slider':
                    # Apply min/max constraints explicitly
                    min_val = var_def.get('min', value)
                    max_val = var_def.get('max', value)
                    value = max(min_val, min(max_val, float(value)))
                elif var_def.get('type') == 'boolean':
                    value = bool(value)
                
                resolved[key] = value
            else:
                # Allow unknown parameters for backward compatibility
                resolved[key] = value
        
        return resolved
    
    @classmethod
    def auto_discover_variables(cls):
        """
        Simple method for models to define their variables.
        Override this in each model to specify variables explicitly.
        """
        return {}
    


class SimulationContext:
    """Container for all simulation context information passed to submodels."""
    
    def __init__(self, 
                 current_time: float,
                 current_datetime: datetime,
                 all_params: Dict[str, Any],
                 all_stocks: Dict[str, float],
                 intervention_active: bool = False):
        self.current_time = current_time  # Elapsed time in days
        self.current_datetime = current_datetime  # Actual datetime
        self.all_params = all_params  # All model parameters
        self.all_stocks = all_stocks  # All current stock values
        self.intervention_active = intervention_active  # Whether interventions are active
        
    def get_param(self, key: str, default=None):
        """Get a parameter value with optional default."""
        return self.all_params.get(key, default)
        
    def get_stock(self, key: str, default=0.0):
        """Get a stock value with optional default."""
        return self.all_stocks.get(key, default)