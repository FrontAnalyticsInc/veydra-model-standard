"""Veydra Model Standard - Base classes for VMS-compliant models.

Provides the core infrastructure for building system dynamics models:
- Submodel: base class for individual model components
- VeydraModelStandard: base class for model orchestrators
- SimulationContext: time-aware parameter resolution with native intervention gating
- Summary/stacked output helpers for multi-scenario execution
"""

import math
import numpy as np
from scipy.integrate import solve_ivp
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime


# ---------------------------------------------------------------------------
# Meta-keys stripped from input before parameter resolution
# ---------------------------------------------------------------------------
META_KEYS = frozenset([
    'include_baseline',
    '__calibration_params__',
    '__initial_state__',
    '__time_window__',
    '__output_format__',
    '__summary_stats__',
    '__summary_variable__',
    'scenarios',
])

DEFAULT_SUMMARY_STATS = ['final_value', 'peak', 'min', 'mean', 'growth_pct']


# ---------------------------------------------------------------------------
# Summary statistics helpers
# ---------------------------------------------------------------------------
def compute_summary_stats(values: List[float],
                          stat_names: Optional[List[str]] = None) -> Dict[str, float]:
    """Compute named summary statistics for a single time-series.
    
    Args:
        values: List of numeric values (one per time point).
        stat_names: Which stats to compute (default: DEFAULT_SUMMARY_STATS).
        
    Returns:
        Dict mapping stat name -> computed value.
    """
    if not values:
        return {s: 0.0 for s in (stat_names or DEFAULT_SUMMARY_STATS)}
    
    stats = stat_names or DEFAULT_SUMMARY_STATS
    result = {}
    
    for stat in stats:
        if stat == 'final_value':
            result[stat] = float(values[-1])
        elif stat == 'peak':
            result[stat] = float(max(values))
        elif stat == 'min':
            result[stat] = float(min(values))
        elif stat == 'mean':
            result[stat] = float(sum(values) / len(values))
        elif stat == 'growth_pct':
            initial = values[0] if values[0] != 0 else 1e-9
            result[stat] = float(((values[-1] - values[0]) / abs(initial)) * 100)
        elif stat == 'variance':
            mean_val = sum(values) / len(values)
            result[stat] = float(sum((v - mean_val) ** 2 for v in values) / len(values))
        elif stat == 'time_to_peak':
            result[stat] = float(values.index(max(values)))
        else:
            result[stat] = 0.0  # Unknown stat -- safe fallback
    
    return result


def build_summary_table(scenario_results: Dict[str, dict],
                        primary_stock: str,
                        stat_names: Optional[List[str]] = None) -> dict:
    """Build a compact summary table across multiple scenarios.
    
    Args:
        scenario_results: Dict of scenario_id -> single-run result dict.
        primary_stock: Which stock variable to summarise.
        stat_names: Which stats to compute.
        
    Returns:
        Dict with 'columns' (list of str) and 'rows' (list of lists).
    """
    stats = stat_names or DEFAULT_SUMMARY_STATS
    columns = ['scenario_id'] + stats
    rows = []
    
    for scenario_id, result in scenario_results.items():
        stock_values = result.get('stocks', {}).get(primary_stock, [])
        computed = compute_summary_stats(stock_values, stats)
        rows.append([scenario_id] + [computed[s] for s in stats])
    
    return {'columns': columns, 'rows': rows, 'primary_stock': primary_stock}


def build_stacked_table(scenario_results: Dict[str, dict],
                        output_config: Optional[dict] = None) -> dict:
    """Build a long-format stacked table for all scenarios.
    
    Each row is [scenario_id, time, variable, value].
    Variable names use shorthand (last segment after '.').
    
    Args:
        scenario_results: Dict of scenario_id -> single-run result dict.
        output_config: Optional dict with 'stocks' and 'flows' sets to filter output.
        
    Returns:
        Dict with 'columns' (list of str) and 'rows' (list of lists).
    """
    columns = ['scenario_id', 'time', 'variable', 'value']
    rows = []
    
    for scenario_id, result in scenario_results.items():
        time_points = result.get('time', [])
        
        # Stocks
        for var_name, values in result.get('stocks', {}).items():
            if output_config and var_name not in output_config.get('stocks', set()):
                continue
            short_name = var_name.split('.')[-1]
            for i, t in enumerate(time_points):
                val = values[i] if i < len(values) else 0.0
                rows.append([scenario_id, t, short_name, val])
        
        # Flows
        for var_name, values in result.get('flows', {}).items():
            if output_config and var_name not in output_config.get('flows', set()):
                continue
            short_name = var_name.split('.')[-1]
            for i, t in enumerate(time_points):
                val = values[i] if i < len(values) else 0.0
                rows.append([scenario_id, t, short_name, val])
    
    return {'columns': columns, 'rows': rows}


def apply_time_window(result: dict, time_window: dict) -> dict:
    """Filter a single-run result dict to only include time points within a window.
    
    Args:
        result: Single-run result with 'time', 'stocks', 'flows' keys.
        time_window: Dict with 'start' and/or 'end' keys.
        
    Returns:
        New result dict with filtered arrays.
    """
    if not time_window or not result.get('time'):
        return result
    
    start = time_window.get('start', 0)
    end = time_window.get('end', float('inf'))
    
    time_points = result['time']
    # Find indices within the window
    indices = [i for i, t in enumerate(time_points) if start <= t <= end]
    
    if not indices:
        return result
    
    filtered = dict(result)  # shallow copy
    filtered['time'] = [time_points[i] for i in indices]
    filtered['stocks'] = {
        k: [v[i] for i in indices] for k, v in result.get('stocks', {}).items()
    }
    filtered['flows'] = {
        k: [v[i] for i in indices] for k, v in result.get('flows', {}).items()
    }
    return filtered

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
        
        # First try the normal module resolution
        if cls.__module__ in sys.modules and cls.__module__ != 'builtins':
            module = sys.modules[cls.__module__]
            if hasattr(module, 'VARIABLES'):
                return getattr(module, 'VARIABLES').copy()
        
        # Fallback for Pyodide/browser environments where __module__ might be 'builtins'
        # In this case, look for VARIABLES in the same namespace as the class
        try:
            # Get the globals from the frame where this method is being called
            import inspect
            frame = inspect.currentframe().f_back
            caller_globals = frame.f_globals
            if 'VARIABLES' in caller_globals:
                return caller_globals['VARIABLES'].copy()
        except:
            pass
        
        # Another fallback: check __main__ module
        try:
            import __main__
            if hasattr(__main__, 'VARIABLES'):
                return getattr(__main__, 'VARIABLES').copy()
        except:
            pass
            
        raise NotImplementedError(
            f"Subclass {cls.__name__} must implement get_variables() "
            f"or define VARIABLES dict in module {cls.__module__}. "
            f"Module info: {cls.__module__} in {list(sys.modules.keys())[-3:] if sys.modules else 'no modules'}"
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
    
    Provides:
    - Parameter cleaning / resolution / validation
    - run_multi_scenario() for batch execution with shared calibration
    - Output formatting (full / summary / stacked / all)
    """
    def __init__(self, params: Dict[str, Any]):
        """Initialize with parameter dictionary."""
        self.params = params
    
    def _clean_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clean parameter dictionary by extracting 'value' if needed.
        Preserves dynamic (time-varying) parameter specs and strips meta-keys."""
        cleaned = {}
        for key, val in params.items():
            # Skip meta-keys — they are handled separately
            if key in META_KEYS:
                continue
            if isinstance(val, dict) and val.get('__dynamic__'):
                # Preserve dynamic parameter spec as-is
                cleaned[key] = val
            elif isinstance(val, dict) and 'value' in val:
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
                
                # Dynamic (time-varying) params bypass scalar validation
                if isinstance(value, dict) and value.get('__dynamic__'):
                    resolved[key] = value
                elif var_def.get('type') == 'slider':
                    # Apply min/max constraints explicitly
                    min_val = var_def.get('min', value)
                    max_val = var_def.get('max', value)
                    value = max(min_val, min(max_val, float(value)))
                    resolved[key] = value
                elif var_def.get('type') == 'boolean':
                    resolved[key] = bool(value)
                else:
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

    # ------------------------------------------------------------------
    # Multi-scenario execution
    # ------------------------------------------------------------------
    def run_multi_scenario(self, config: Dict[str, Any]) -> dict:
        """Run one or many scenarios in a single call with shared calibration.
        
        Input config shape:
            {
                "include_baseline": true,           # default false
                "simulation.duration": 104,         # shared simulation params
                "simulation.intervention_start_time": 52,
                "__calibration_params__": {...},     # pre-intervention overrides
                "__initial_state__": {...},          # optional stock overrides at t=0
                "__time_window__": {"start": 0, "end": 104},
                "__output_format__": "full",         # "full"|"summary"|"stacked"|"all"
                "__summary_stats__": [...],          # stat names for summary mode
                "__summary_variable__": "...",       # stock to summarise (default: first)
                "scenarios": [
                    {"id": "s1", "params": {"retail.order_quantity": 800}},
                    {"id": "s2", "params": {"retail.order_lead_time": 8.0}},
                ]
            }
            
        Returns:
            Combined result dict with requested output format(s).
        """
        # ---- extract meta-keys ----
        include_baseline = config.get('include_baseline', False)
        calibration_params = config.get('__calibration_params__', {})
        initial_state = config.get('__initial_state__', {})
        time_window = config.get('__time_window__', None)
        output_format = config.get('__output_format__', 'full')
        summary_stats = config.get('__summary_stats__', None)
        summary_variable = config.get('__summary_variable__', None)
        scenarios_list = config.get('scenarios', [])
        
        # ---- extract shared simulation params ----
        sim_params = {}
        for key, val in config.items():
            if key.startswith('simulation.'):
                sim_params[key] = val
        
        # ---- determine intervention_start_time ----
        intervention_start = None
        ist_val = sim_params.get('simulation.intervention_start_time')
        if ist_val is not None:
            try:
                intervention_start = float(ist_val['value'] if isinstance(ist_val, dict) and 'value' in ist_val else ist_val)
            except (TypeError, ValueError):
                pass
        # Fallback to model's own params
        if intervention_start is None and hasattr(self, 'params'):
            ist_val2 = self.params.get('simulation.intervention_start_time')
            if ist_val2 is not None:
                try:
                    intervention_start = float(ist_val2)
                except (TypeError, ValueError):
                    pass
        
        # ---- run baseline (pure defaults + calibration, no scenario overrides) ----
        scenario_results: Dict[str, dict] = {}
        
        if include_baseline:
            baseline_params = dict(sim_params)
            # For baseline, calibration params are applied as regular params
            # (no intervention gating needed — there are no scenario overrides to gate)
            if calibration_params:
                baseline_params.update(calibration_params)
            if initial_state:
                baseline_params['__initial_state__'] = initial_state
            
            # Clear scenario context so SimulationContext uses plain params
            self._scenario_context = None
            baseline_result = self.run_simulation(baseline_params)
            if time_window and baseline_result.get('success'):
                baseline_result = apply_time_window(baseline_result, time_window)
            scenario_results['baseline'] = baseline_result
        
        # ---- If no explicit scenarios list, treat top-level non-sim params as "current" ----
        if not scenarios_list:
            override_params = {}
            for key, val in config.items():
                if key not in META_KEYS and not key.startswith('simulation.'):
                    override_params[key] = val
            if override_params:
                scenarios_list = [{'id': 'current', 'params': override_params}]
        
        # ---- run each scenario ----
        for scenario_def in scenarios_list:
            scenario_id = scenario_def.get('id', f'scenario_{len(scenario_results)}')
            scenario_params = scenario_def.get('params', {})
            
            # Build merged params: simulation config + scenario overrides
            merged = dict(sim_params)
            merged.update(scenario_params)
            if initial_state:
                merged['__initial_state__'] = initial_state
            
            # Determine which keys are scenario overrides (non-simulation, post-intervention)
            override_keys = set(scenario_params.keys()) - {k for k in scenario_params if k.startswith('simulation.')}
            
            # Set scenario context for native intervention gating in SimulationContext
            if intervention_start is not None and override_keys:
                # Build param defaults for pre-intervention period:
                # calibration values take precedence, then model variable defaults
                param_defaults = dict(calibration_params)
                try:
                    all_vars = self.auto_discover_variables() if hasattr(self, 'auto_discover_variables') else {}
                    for k in override_keys:
                        if k not in param_defaults and k in all_vars:
                            param_defaults[k] = all_vars[k].get('default', 0.0)
                except Exception:
                    pass
                
                self._scenario_context = {
                    'intervention_start_time': intervention_start,
                    'scenario_params': override_keys,
                    'calibration_params': calibration_params,
                    'param_defaults': param_defaults,
                }
            else:
                self._scenario_context = None
            
            result = self.run_simulation(merged)
            if time_window and result.get('success'):
                result = apply_time_window(result, time_window)
            scenario_results[scenario_id] = result
        
        # Clear scenario context after batch
        self._scenario_context = None
        
        # ---- determine primary stock for summary ----
        primary_stock = summary_variable
        if not primary_stock:
            # Use first stock from first successful result
            for res in scenario_results.values():
                if res.get('success') and res.get('stocks'):
                    primary_stock = next(iter(res['stocks']))
                    break
        
        # ---- build output in requested format(s) ----
        output = {'success': True}
        
        # Filter to only successful scenarios for summary/stacked
        successful = {k: v for k, v in scenario_results.items() if v.get('success')}
        
        if output_format in ('full', 'all'):
            output['scenarios'] = scenario_results
        
        if output_format in ('summary', 'all'):
            if primary_stock and successful:
                output['summary'] = build_summary_table(successful, primary_stock, summary_stats)
            else:
                output['summary'] = {'columns': [], 'rows': [], 'primary_stock': primary_stock}
        
        if output_format in ('stacked', 'all'):
            output_config = getattr(self, 'output_config', None)
            output['stacked'] = build_stacked_table(successful, output_config)
        
        # If only one format and it's not 'all', set top-level success from scenario results
        any_failed = any(not v.get('success') for v in scenario_results.values())
        if any_failed:
            output['success'] = True  # partial success — individual failures in their results
            output['warnings'] = [
                f"Scenario '{k}' failed: {v.get('error', 'unknown')}"
                for k, v in scenario_results.items() if not v.get('success')
            ]
        
        return output


class SimulationContext:
    """Container for all simulation context information passed to submodels.
    
    Provides native support for:
    - Intervention gating: before intervention_start_time, scenario overrides
      are replaced with calibration values (or model defaults)
    - Dynamic (time-varying) parameters: function expressions, array/drawn curves
    - Plain scalar parameter lookup (fast path, no overhead)
    """
    
    def __init__(self, 
                 current_time: float,
                 current_datetime: datetime,
                 all_params: Dict[str, Any],
                 all_stocks: Dict[str, float],
                 intervention_active: bool = False,
                 # --- New scenario context fields (optional) ---
                 intervention_start_time: Optional[float] = None,
                 scenario_param_keys: Optional[set] = None,
                 calibration_params: Optional[Dict[str, Any]] = None,
                 param_defaults: Optional[Dict[str, Any]] = None):
        self.current_time = current_time
        self.current_datetime = current_datetime
        self.all_params = all_params
        self.all_stocks = all_stocks
        self.intervention_active = intervention_active
        
        # Scenario context for native intervention gating
        self.intervention_start_time = intervention_start_time
        self.scenario_param_keys = scenario_param_keys or set()
        self.calibration_params = calibration_params or {}
        self.param_defaults = param_defaults or {}
        
        self._dynamic_cache = {}  # Cache for compiled dynamic parameter expressions
        
    def get_param(self, key: str, default=None):
        """Get a parameter value with optional default.
        
        Handles three concerns in priority order:
        1. Intervention gating — if before intervention_start_time and key is a
           scenario override, return calibration value or model default instead.
        2. Dynamic parameters — if the value is a __dynamic__ spec, resolve it
           to a scalar at current_time.
        3. Plain scalar — return the value from all_params directly.
        """
        # --- 1. Intervention gating ---
        if (self.intervention_start_time is not None 
                and self.current_time < self.intervention_start_time
                and key in self.scenario_param_keys):
            # Before intervention: use calibration value if provided, else model default
            if key in self.calibration_params:
                return self.calibration_params[key]
            if key in self.param_defaults:
                return self.param_defaults[key]
            # Fall through to plain lookup (which will have the model default)
        
        value = self.all_params.get(key, default)
        
        # --- 2. Dynamic parameter resolution ---
        if isinstance(value, dict) and value.get('__dynamic__'):
            return self._resolve_dynamic_param(key, value, default)
        
        # --- 3. Plain scalar ---
        return value
    
    def _resolve_dynamic_param(self, key: str, spec: dict, default=None):
        """Resolve a dynamic parameter spec to a scalar value at current_time."""
        t = self.current_time
        mode = spec.get('mode', 'function')
        
        try:
            # Prefer pre-computed points (generated by frontend JS)
            points = spec.get('points', [])
            if points and mode in ('array', 'drawn', 'function'):
                val = self._interpolate_points(points, t)
                if val is not None:
                    return val
            
            if mode == 'function':
                return self._eval_function_param(key, spec, t)
            
            return spec.get('defaultValue', default)
        except Exception:
            return spec.get('defaultValue', default)
    
    def _eval_function_param(self, key: str, spec: dict, t: float):
        """Evaluate a math expression like 'sin(t * 0.1) * 50 + 100'."""
        expression = spec.get('expression', '')
        if not expression:
            return spec.get('defaultValue', 0.0)
        
        if key not in self._dynamic_cache:
            self._dynamic_cache[key] = {
                'code': compile(expression, f'<param:{key}>', 'eval'),
                'math_ns': {
                    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                    'exp': math.exp, 'log': math.log, 'sqrt': math.sqrt,
                    'abs': abs, 'min': min, 'max': max, 'pow': pow,
                    'pi': math.pi, 'e': math.e,
                    'floor': math.floor, 'ceil': math.ceil,
                }
            }
        
        cached = self._dynamic_cache[key]
        return float(eval(cached['code'], {'__builtins__': {}}, 
                         {**cached['math_ns'], 't': t}))
    
    @staticmethod
    def _interpolate_points(points, t: float):
        """Linearly interpolate from a sorted list of [time, value] points."""
        if not points or len(points) < 2:
            return None
        
        if t <= points[0][0]:
            return float(points[0][1])
        if t >= points[-1][0]:
            return float(points[-1][1])
        
        lo, hi = 0, len(points) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if points[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        
        t0, v0 = points[lo]
        t1, v1 = points[hi]
        if t1 == t0:
            return float(v0)
        frac = (t - t0) / (t1 - t0)
        return float(v0 + frac * (v1 - v0))
        
    def get_stock(self, key: str, default=0.0):
        """Get a stock value with optional default."""
        return self.all_stocks.get(key, default)