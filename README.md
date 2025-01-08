
# aix
Artificial Intelligence eXtensions

Fast access to your favorite A.I. tools. 

To install:	```pip install aix```



# AI chat

`aix.chat_funcs` gathers the models you have access to in your environment 
(depends on what you have installed (e.g. `google.generativeai`, `oa` (itself a facade to `openai` functionality), etc.)).


```python
from aix import chat_funcs

list(chat_funcs)
```




    ['gemini-1.5-flash',
     'gpt-4',
     'gpt-4-32k',
     'gpt-4-turbo',
     'gpt-3.5-turbo',
     'o1-preview',
     'o1-mini',
     'gpt-4o',
     'gpt-4o-mini']



`chat_funcs` is a dictionary whos keys are the names of the available models, and 
values are a `chat` function with the model set to that model name. 


```python
chat_funcs['o1-mini']
```



    functools.partial(<function chat at 0x1355f68c0>, model='o1-mini')



Note that different providers have different interfaces, but the functions that 
`chat_funcs` provides all have `prompt` as their first argument. 


```python
from inspect import signature

signature(chat_funcs['o1-mini'])
```




    <Sig (prompt=None, *, model='o1-mini', messages=None, frequency_penalty: 'Optional[float] | NotGiven' = NOT_GIVEN, function_call: 'completion_create_params.FunctionCall | NotGiven' = NOT_GIVEN, functions: 'Iterable[completion_create_params.Function] | NotGiven' = NOT_GIVEN, logit_bias: 'Optional[Dict[str, int]] | NotGiven' = NOT_GIVEN, logprobs: 'Optional[bool] | NotGiven' = NOT_GIVEN, max_tokens: 'Optional[int] | NotGiven' = NOT_GIVEN, n: 'Optional[int] | NotGiven' = NOT_GIVEN, parallel_tool_calls: 'bool | NotGiven' = NOT_GIVEN, presence_penalty: 'Optional[float] | NotGiven' = NOT_GIVEN, response_format: 'completion_create_params.ResponseFormat | NotGiven' = NOT_GIVEN, seed: 'Optional[int] | NotGiven' = NOT_GIVEN, service_tier: "Optional[Literal['auto', 'default']] | NotGiven" = NOT_GIVEN, stop: 'Union[Optional[str], List[str]] | NotGiven' = NOT_GIVEN, stream: 'Optional[Literal[False]] | Literal[True] | NotGiven' = NOT_GIVEN, stream_options: 'Optional[ChatCompletionStreamOptionsParam] | NotGiven' = NOT_GIVEN, temperature: 'Optional[float] | NotGiven' = NOT_GIVEN, tool_choice: 'ChatCompletionToolChoiceOptionParam | NotGiven' = NOT_GIVEN, tools: 'Iterable[ChatCompletionToolParam] | NotGiven' = NOT_GIVEN, top_logprobs: 'Optional[int] | NotGiven' = NOT_GIVEN, top_p: 'Optional[float] | NotGiven' = NOT_GIVEN, user: 'str | NotGiven' = NOT_GIVEN, extra_headers: 'Headers | None' = None, extra_query: 'Query | None' = None, extra_body: 'Body | None' = None, timeout: 'float | httpx.Timeout | None | NotGiven' = NOT_GIVEN)>




```python
signature(chat_funcs['gemini-1.5-flash'])
```




    <Sig (prompt: str, *, model='gemini-1.5-flash', generation_config: 'generation_types.GenerationConfigType | None' = None, safety_settings: 'safety_types.SafetySettingOptions | None' = None, stream: 'bool' = False, tools: 'content_types.FunctionLibraryType | None' = None, tool_config: 'content_types.ToolConfigType | None' = None, request_options: 'helper_types.RequestOptionsType | None' = None)>



For tab-completion convenience, the (python identifier version of the) models 
were placed as attributes of `chat_funcs`, so you can access them directly there.


```python
print(chat_funcs.gemini_1_5_flash('What is the capital of France?'))
```

    The capital of France is **Paris**. 
    



```python
print(chat_funcs.gpt_3_5_turbo('What is the capital of France?'))
```

    The capital of France is Paris.


There's also a dictionary called `chat_models` that contains the same keys:


```python
from aix import chat_models

list(chat_models)
```




    ['gemini-1.5-flash',
     'gpt-4',
     'gpt-4-32k',
     'gpt-4-turbo',
     'gpt-3.5-turbo',
     'o1-preview',
     'o1-mini',
     'gpt-4o',
     'gpt-4o-mini']



But here the values are some useful metadatas on the model, like pricing...


```python
chat_models['gpt-4o']
```




    {'price_per_million_tokens': 5.0,
     'pages_per_dollar': 804,
     'performance_on_eval': 'Efficiency-optimized version of GPT-4 for better performance on reasoning tasks',
     'max_input': 8192,
     'provider': 'openai'}



The corresponding attributes are only the name of the model (the key itself):


```python
chat_models.gpt_4
```



    'gpt-4'



This is for the convenience of entering a model name in a different context, with 
less errors than if you were typing the name as a string. 
For example, you can enter it yourself in the general `chat` function:


```python
from aix import chat, chat_models

chat('How many Rs in "Strawberry"?', model=chat_models.gpt_4o, frequency_penalty=0.5)  
```




    'The word "Strawberry" contains two instances of the letter \'R\'.'




# Extras (older version -- might deprecate or change interface)

Want all your faves at your fingertips?

Never remember where to import that learner from?

Say `LinearDiscriminantAnalysis`?

... was it `from sklearn`?

... was it `from sklearn.linear_model`?

... ah no! It was `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis`.

Sure, you can do that. Or you can simply type `from aix.Lin...` click tab, and; there it is! 
Select, enter, and moving on with real work.

*Note: This is meant to get off the ground quickly 
-- once your code is stable, you should probably import your stuff directly from it's origin*


# Coming up

Now that the AI revolution is on its way, we'll add the ability to find, and one day,
use the right AI tool -- until the day that AI will do even that for us...