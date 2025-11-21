# Contributing to AIX

Thank you for your interest in contributing to AIX! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/thorwhalen/aix.git
   cd aix
   ```

2. **Install in development mode**:
   ```bash
   pip install -e .
   ```

3. **Install development dependencies**:
   ```bash
   pip install pytest pytest-cov pytest-mock ruff
   ```

4. **Set up API keys** (for testing):
   ```bash
   export OPENAI_API_KEY=your-key-here
   export ANTHROPIC_API_KEY=your-key-here
   export OPENROUTER_API_KEY=your-key-here
   ```

## Code Style

AIX follows the **i2mint philosophy**:
- **Mapping interfaces** for collections
- **Functional approach** over verbose OOP
- **Clean, pythonic code**
- **Comprehensive docstrings with examples**

### Linting

We use Ruff for linting:

```bash
ruff check aix/
ruff format aix/
```

### Type Hints

Use type hints where appropriate:

```python
def chat(
    prompt: Union[str, Iterable[dict]],
    *,
    model: str = None,
    temperature: float = None,
) -> str:
    ...
```

### Docstrings

Every public function needs:
- Clear description
- Parameter documentation
- Return value documentation
- At least one example (preferably as doctest)

Example:

```python
def chat(prompt: str, *, model: str = None) -> str:
    """Send a chat prompt and get a response.

    Args:
        prompt: The text prompt to send
        model: Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4')

    Returns:
        Generated text response

    Examples:
        >>> chat("What is 2+2?")  # doctest: +SKIP
        'The answer is 4.'
    """
```

## Testing

### Running Tests

```bash
# Run all tests
./run_tests.sh

# Or manually:
pytest tests/ -v
pytest --doctest-modules aix/
```

### Writing Tests

1. **Unit tests** go in `tests/test_*.py`
2. Use **mocking** to avoid requiring API keys
3. Follow existing test patterns

Example:

```python
@patch('aix.chat._litellm_completion')
def test_simple_chat(mock_completion):
    """Test simple chat with string prompt."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response"
    mock_completion.return_value = mock_response

    result = chat("Hello")

    assert result == "Test response"
```

### Test Coverage

Aim for:
- **80%+ code coverage** for new features
- **All public functions** have tests
- **Edge cases** are covered

## Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation

3. **Run tests**:
   ```bash
   ./run_tests.sh
   ```

4. **Commit with clear messages**:
   ```bash
   git commit -m "feat: Add new feature X"
   ```

   Use conventional commits:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `test:` - Test additions/changes
   - `refactor:` - Code refactoring

5. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **PR Description**:
   - Describe what changed and why
   - Reference any related issues
   - Include examples if applicable

## Adding New Features

### New Core Function

If adding a new core function (like `chat`, `embeddings`):

1. Create module in `aix/`
2. Wrap LiteLLM (don't expose directly)
3. Follow Mapping/functional patterns
4. Add comprehensive tests
5. Add example in `examples/`
6. Export in `aix/__init__.py`
7. Document in README.md

### New Model Source

If adding support for a new model source:

1. Create source class in `aix/ai_models/sources.py`
2. Inherit from `ModelSource`
3. Implement `discover_models()` method
4. Add tests
5. Document usage

## Documentation

### README Updates

When adding new features, update:
- Quick Start section
- Core Features section
- Examples section

### Examples

Add practical examples in `examples/`:
- Clear, runnable code
- Commented explanations
- Cover common use cases

### Docstrings

Update docstrings when:
- Function signature changes
- New parameters added
- Return value changes
- New exceptions raised

## Code Review

All PRs require review. Reviewers will check:
- Code quality and style
- Test coverage
- Documentation completeness
- Performance implications
- Breaking changes

## Questions?

- **Discussions**: Use GitHub Discussions
- **Issues**: Report bugs or request features
- **Email**: Contact maintainers

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
