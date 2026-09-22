# aix.prompts

Prompt-based function creation for AIX.

Transform text prompts into callable Python functions with automatic parameter extraction.

This module provides the core functionality for creating reusable AI-powered functions
from natural language prompts. It supports both text generation and structured output.

### Examples

Create a simple text function:

```pycon
>>> from aix.prompts import prompt_func
>>> translate = prompt_func("Translate to French: {text}")
>>> translate(text="Hello world")
'Bonjour le monde'
```

Create a function with structured output:

```pycon
>>> extract_person = prompt_func(
...     "Extract person information from: {text}",
...     output_schema={"name": str, "age": int}
... )
>>> extract_person(text="Alice is 30 years old")
{'name': 'Alice', 'age': 30}
```

Multiple parameters:

```pycon
>>> compare = prompt_func(
...     "Compare {item1} and {item2} in terms of {aspect}"
... )
>>> compare(item1="Python", item2="JavaScript", aspect="performance")
'Python generally has better...'
```

### Functions

| [`constrained_answer`](#aix.prompts.constrained_answer)(prompt, valid_answers, \*)   | Get an answer from the LLM constrained to a set of valid answers or types.   |
|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| [`prompt_func`](#aix.prompts.prompt_func)(template, \*[, output_schema, ...]) | Create a callable function from a prompt template.                           |
| [`prompt_to_json`](#aix.prompts.prompt_to_json)(template, schema, \*\*kwargs)    | Create a function that returns structured JSON output.                       |
| [`prompt_to_text`](#aix.prompts.prompt_to_text)(template, \*\*kwargs)            | Create a function that returns text output.                                  |

### Classes

| [`CommonFuncs`](#aix.prompts.CommonFuncs)([model])   | Collection of commonly used prompt functions.   |
|-------------------------------------------------------------------------|-------------------------------------------------|
| [`PromptFuncs`](#aix.prompts.PromptFuncs)([model])   | Collection of prompt-based functions.           |

### Exceptions

| [`ConstraintViolation`](#aix.prompts.ConstraintViolation)(message, \*[, answer, ...])   | Raised when an LLM answer does not satisfy the requested constraint.   |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|

### *class* aix.prompts.CommonFuncs(model=None)

Bases: [`PromptFuncs`](#aix.prompts.PromptFuncs)

Collection of commonly used prompt functions.

### Examples

```pycon
>>> from aix.prompts import common_funcs
>>> common_funcs.summarize(text="Long article...")
'Summary...'
>>> common_funcs.extract_keywords(text="Article about AI")
['AI', 'artificial', 'intelligence']
```

### *exception* aix.prompts.ConstraintViolation(message, , answer=None, valid_answers=None)

Bases: [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

Raised when an LLM answer does not satisfy the requested constraint.

A subclass of `ValueError` so that callers who already guard
[`constrained_answer()`](#aix.prompts.constrained_answer) with `except ValueError` keep working.

#### answer

The offending value, as it came back from the model.

#### valid_answers

The constraint the value failed, verbatim as passed in.

### Examples

```pycon
>>> violation = ConstraintViolation("nope", answer="nope", valid_answers=["a"])
>>> violation.answer, violation.valid_answers
('nope', ['a'])
>>> isinstance(violation, ValueError)
True
```

### *class* aix.prompts.PromptFuncs(model=None, \*\*default_kwargs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Collection of prompt-based functions.

Provides a namespace for organizing related prompt functions with
attribute-based access.

### Examples

```pycon
>>> funcs = PromptFuncs()
>>> funcs.add('summarize', "Summarize: {text}")
>>> funcs.add('translate', "Translate {text} to {language}")
>>> funcs.summarize(text="Long article...")
'Summary...'
>>> funcs.translate(text="Hello", language="Spanish")
'Hola'
```

#### add(name, template, , output_schema=None, \*\*kwargs)

Add a function to the collection.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name (will be accessible as attribute)
  * **template** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Prompt template
  * **output_schema** (`Union`[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict), [`type`](https://docs.python.org/3/builtins/functions.html#type)]) – Optional schema for structured output
  * **\*\*kwargs** – Additional parameters for prompt_func
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### keys()

Get all function names.

### aix.prompts.constrained_answer(prompt, valid_answers, , model=None, temperature=None, enhance_prompt=False, n=1, on_violation='raise', max_retries=0)

Get an answer from the LLM constrained to a set of valid answers or types.

Uses JSON mode to ensure the LLM returns a valid response based on constraints.
More flexible than the oa version - works with any model that supports JSON mode
via LiteLLM.

This can be seen as a facade for some common structured output use cases, as well
as a convenient tool to do response statistics and validation (via n>1).

* **Parameters:**
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The question or prompt to ask the LLM
  * **valid_answers** (`Union`[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`int`](https://docs.python.org/3/builtins/functions.html#int)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)], [`type`](https://docs.python.org/3/builtins/functions.html#type), [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – 

    Can be:
    - list[str]: List of valid string options
    - list[int]: List of valid integer options
    - list[float]: List of valid float options
    - bool: Constrains answer to True or False
    - int: Any integer
    - float: Any number
    - tuple[float, float]: Numerical range (min, max) inclusive
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The model to use for the LLM (default: uses DFLT_CHAT_MODEL)
  * **temperature** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Temperature for sampling (default: None, uses model’s default).
    Higher values (e.g., 1.0) give more random/varied results.
    Lower values (e.g., 0.0) give more deterministic results.
  * **enhance_prompt** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, adds explicit instructions to the prompt about
    JSON formatting and constraints. If False (default), relies on
    response_format alone. Default is False to match oa behavior.
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of times to call the LLM (default: 1)
  * **on_violation** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'raise'`, `'return'`]) – What to do when the model’s answer does not satisfy
    `valid_answers`. `'raise'` (default) raises `ConstraintViolation`.
    `'return'` returns the model’s value unchecked – the behaviour of
    this function before enforcement existed, kept as an escape hatch.
  * **max_retries** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – How many times to re-ask the model after a violation
    before giving up (default: 0, i.e. ask exactly once). Ignored when
    `on_violation='return'`, which never detects a violation.
* **Returns:**
  One of the valid answers, respecting the type constraint.
  If n > 1, returns a list of answers.
* **Raises:**
  * [**ConstraintViolation**](#aix.prompts.ConstraintViolation) – If the answer is not a member of an options list,
        falls outside a `(min, max)` range, or cannot be converted to the
        declared type – unless `on_violation='return'`.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If the response is not JSON, or has no “answer” field.

### Examples

```pycon
>>> # String options
>>> answer = constrained_answer(
...     "Is Python compiled or interpreted?",
...     ["compiled", "interpreted", "both"]
... )
>>> answer in ["compiled", "interpreted", "both"]
True
```

```pycon
>>> # Boolean
>>> answer = constrained_answer(
...     "Is Python dynamically typed?",
...     bool
... )
>>> isinstance(answer, bool)
True
```

```pycon
>>> # Integer options
>>> answer = constrained_answer(
...     "How many wheels does a car have?",
...     [2, 3, 4, 6, 8]
... )
>>> answer in [2, 3, 4, 6, 8]
True
```

```pycon
>>> # Numerical range
>>> answer = constrained_answer(
...     "What is a reasonable hourly rate for a senior Python developer? (USD)",
...     (50.0, 300.0)
... )
>>> 50.0 <= answer <= 300.0
True
```

```pycon
>>> # Multiple samples for statistics
>>> answers = constrained_answer(
...     "Which is better: cats or dogs?",
...     ["cats", "dogs"],
...     n=10
... )
>>> len(answers)
10
```

```pycon
>>> # Tolerate an off-constraint answer instead of raising
>>> answer = constrained_answer(
...     "Is Python compiled or interpreted?",
...     ["compiled", "interpreted"],
...     on_violation="return",
... )
```

### aix.prompts.prompt_func(template, , output_schema=None, egress=None, model=None, temperature=None, name=None, \*\*chat_kwargs)

Create a callable function from a prompt template.

This is the main function for creating prompt-based functions. It automatically
detects parameters from the template and creates a function with those parameters.

Without output_schema: Returns text
With output_schema: Returns structured data (dict, list, etc.)
With egress: Returns whatever the egress post-processor returns.

Templates use the default-aware `{name:default}` dialect (matching
`oa.prompt_function`): `{name}` is a required parameter, `{name:default}`
supplies a default value (the text after the colon), and braces inside
`` fenced `` regions are left literal. Plain `{name}`-only templates behave
exactly as before.

```pycon
>>> greet = prompt_func("Greet {name} in {language:English}")
>>> greet.param_names
['name', 'language']
```

* **Parameters:**
  * **template** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Prompt template with {var} placeholders for parameters
  * **output_schema** (`Union`[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict), [`type`](https://docs.python.org/3/builtins/functions.html#type)]) – Optional schema for structured output. Can be:
    - Dict mapping field names to types: {“name”: str, “age”: int}
    - A single type for simple outputs: str, int, list, etc.
    - None for plain text output (default)
  * **egress** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Optional post-processor `(result) -> Any` applied to the output
    before returning — on both the text and structured paths. Lets a caller
    keep “prompt → typed Python value” inside the facade (e.g. parse the
    LLM text into a list of lines/ids) instead of wrapping the returned
    function. `None` (default) returns the raw result unchanged.
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model to use for this function
  * **temperature** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Temperature for generation
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional `__name__` for the generated function (for tracing /
    identity). Defaults to `"prompt_based_function"`.
  * **\*\*chat_kwargs** – Additional parameters passed to chat()
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)
* **Returns:**
  Callable function with parameters derived from template

### Examples

```pycon
>>> # Simple text generation
>>> summarize = prompt_func("Summarize this text: {text}")
>>> summarize(text="Long article...")
'Brief summary...'
```

```pycon
>>> # Structured output
>>> extract = prompt_func(
...     "Extract contact info from: {text}",
...     output_schema={"name": str, "email": str, "phone": str}
... )
>>> result = extract(text="Contact John at john@example.com, 555-1234")
>>> result['name']
'John'
```

```pycon
>>> # Multiple parameters
>>> compare = prompt_func(
...     "Compare {item1} and {item2} in terms of {aspect}. "
...     "Keep it under {word_limit} words."
... )
>>> compare(
...     item1="Python",
...     item2="Java",
...     aspect="learning curve",
...     word_limit=50
... )
'Python has a gentler learning curve...'
```

```pycon
>>> # With specific model
>>> creative_writer = prompt_func(
...     "Write a creative story about {topic}",
...     model="gpt-4o",
...     temperature=1.5
... )
>>> creative_writer(topic="a time-traveling cat")
'Once upon a time, there was a cat named Whiskers...'
```

### aix.prompts.prompt_to_json(template, schema, \*\*kwargs)

Create a function that returns structured JSON output.

This is an explicit alias for prompt_func with output_schema.

* **Parameters:**
  * **template** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Prompt template
  * **schema** (`Union`[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict), [`type`](https://docs.python.org/3/builtins/functions.html#type)]) – Output schema
  * **\*\*kwargs** – Additional parameters for prompt_func
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)
* **Returns:**
  Function that returns structured data

### Examples

```pycon
>>> extract = prompt_to_json(
...     "Extract name and age from: {text}",
...     schema={"name": str, "age": int}
... )
>>> result = extract(text="Alice is 30")
>>> isinstance(result, dict)
True
```

### aix.prompts.prompt_to_text(template, \*\*kwargs)

Create a function that returns text output.

This is an explicit alias for prompt_func without output_schema.

* **Parameters:**
  * **template** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Prompt template
  * **\*\*kwargs** – Additional parameters for prompt_func
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)
* **Returns:**
  Function that returns text

### Examples

```pycon
>>> summarize = prompt_to_text("Summarize: {text}")
>>> result = summarize(text="Long text...")
>>> isinstance(result, str)
True
```
