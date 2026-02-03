from utu.utils import LLMOutputParser

I_CODE_BLOCK = """
这里是一些说明文本。

```python
def hello():
    print("Hello, world!")
```

```json
{"key": "value"}
```

```javascript
console.log("Hello, world!");
```
"""


def test_extract_code_blocks():
    code_blocks = LLMOutputParser.extract_code_blocks(I_CODE_BLOCK)
    print(code_blocks)


def test_extract_code_python():
    code = LLMOutputParser.extract_code_python(I_CODE_BLOCK)
    print(code)


def test_extract_code_json():
    code = LLMOutputParser.extract_code_json(I_CODE_BLOCK)
    print(code)
    code = LLMOutputParser.extract_code_json(I_CODE_BLOCK, try_parse=True)
    print(code)


def test_camel_to_snake():
    code = LLMOutputParser.camel_to_snake("ExtractVideoFrames")
    print(code)


def test_snake_to_camel():
    code = LLMOutputParser.snake_to_camel("extract_video_frames")
    print(code)
    code = LLMOutputParser.snake_to_camel("extract_video_frames", pascal=False)
    print(code)
