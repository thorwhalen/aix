# aix.gen_ai.openai_genai

OpenAI GenAI API functionality (legacy backend, via the optional `oa` package).

`oa` is imported on the first call, never at `import aix` time: importing
`oa` resolves an OpenAI API key as a side effect – raising when none is
configured, and *prompting for one on stdin* in an interactive session – so an
eager import made `import aix` crash or block for anyone who has `oa`
installed but no key set ([https://github.com/thorwhalen/aix/issues/54](https://github.com/thorwhalen/aix/issues/54)).
