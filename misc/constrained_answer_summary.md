# constrained_answer Implementation Summary

## Changes Made

Successfully created `aix.prompts.constrained_answer()` as a flexible, drop-in replacement for `oa.constrained_answer()`.

### Key Features

1. **Same API**: Compatible signature with `oa.constrained_answer`
2. **Varied Responses**: Uses model's default temperature for natural variation (like `oa`)
3. **Flexible Prompting**: Configurable prompt enhancement via `enhance_prompt` parameter
4. **Type Safety**: Properly converts JSON responses to expected Python types

### New Parameters

- `temperature` (float, optional): Controls response randomness
  - `None` (default): Uses model's default temperature (~1.0 for variety)
  - `0.0`: Deterministic responses
  - `1.0+`: More creative/varied responses

- `enhance_prompt` (bool, default=False): Controls prompt modification
  - `False` (default): Minimal JSON instruction, matches `oa` behavior
  - `True`: Adds detailed constraints and formatting instructions

### Supported Constraints

- `list[str]`: List of valid string options
- `list[int]`: List of valid integer options
- `list[float]`: List of valid float options
- `bool`: True/False only
- `int`: Any integer
- `float`: Any number
- `tuple[float, float]`: Numerical range (min, max)

### Usage Examples

```python
from functools import partial
from collections import Counter
from dol import Pipe
from aix.prompts import constrained_answer

display = lambda x: print('\n'.join(['result\tcount', *(f'{xi[0]}\t{xi[1]}' for xi in x.items())]))

poll = Pipe(
     partial(constrained_answer, model="gpt-4o-mini", n=10),
     Counter,
     display
 )

# Get varied responses (uses default temperature)
poll('how tall is a tree?', float)
# Output:
# result    count
# 20.0      2
# 15.0      4
# 30.0      1
# ...

# Boolean questions
poll('Is Python a good language?', bool)

# Multiple choice
poll('Which is better: cats or dogs?', ['cats', 'dogs'])

# With explicit temperature for more determinism
constrained_answer('What is 2+2?', int, temperature=0.0)  # Always same answer

# With enhanced prompts for stricter enforcement
constrained_answer(
    'Pick a color',
    ['red', 'blue', 'green'],
    enhance_prompt=True  # Adds detailed constraints to prompt
)
```

### Migration from `oa`

Simply replace:
```python
from oa import constrained_answer
```

with:
```python
from aix.prompts import constrained_answer
```

All existing code will work as-is!

### Implementation Details

**Helper Function**: `_enhance_prompt_for_json()`
- Separated prompt enhancement logic into reusable helper
- Only called when `enhance_prompt=True`

**Minimal Prompt Enhancement** (when `enhance_prompt=False`):
- Adds minimal JSON instruction with type hint
- Required because OpenAI needs "json" in the prompt for `json_object` mode
- Example: `"how tall is a tree?\nRespond in JSON format with an 'answer' field containing a number."`

**Temperature Handling**:
- Default: `None` (uses model's default, typically ~1.0)
- Matches `oa` behavior for natural response variation
- Can be explicitly set for more control

**Type Conversion**:
- JSON responses automatically converted to expected Python types
- Handles int/float/bool/str conversions
- Gracefully falls back if conversion fails

## Files Modified

1. `/Users/thorwhalen/Dropbox/py/proj/t/aix/aix/prompts.py`
   - Added `_enhance_prompt_for_json()` helper function
   - Added `constrained_answer()` function

2. `/Users/thorwhalen/Dropbox/py/proj/t/aix/aix/__init__.py`
   - Added `constrained_answer` to imports and `__all__`

## Testing

All test cases pass:
- ✓ Boolean constraints
- ✓ String list options
- ✓ Integer constraints
- ✓ Float constraints
- ✓ Integer list options
- ✓ Numerical ranges
- ✓ Multiple samples (n>1)
- ✓ Varied responses (matches `oa` behavior)
