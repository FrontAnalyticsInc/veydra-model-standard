# Veydra Model Standard

The standard base class for all Veydra system dynamics models.

## Installation

```bash
pip install veydra-model-standard
```

For development from local source:

```bash
pip install -e .
```

## Usage

```python
from veydra_model_standard import VeydraModelStandard

class MyModel(VeydraModelStandard):
    def __init__(self):
        super().__init__()
        # Your model implementation
```

## Features

- Standardized model interface
- Built-in variable management
- Configuration loading
- Simulation engine
- Data export capabilities

## Development

```bash
# Install in editable mode
cd veydra-model-standard
pip install -e .

# Run tests
python -m pytest tests/
```