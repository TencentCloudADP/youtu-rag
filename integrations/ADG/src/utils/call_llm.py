import os
import time
import json
import requests
import re
import sys
import random
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    # Fallback for older openai versions
    OpenAI = None
    AzureOpenAI = None

try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent.parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment variables from {env_file}")
    else:
        # Fallback to default .env file
        load_dotenv()
        print("Loaded environment variables from .env file")
except ImportError:
    print("Warning: python-dotenv not available, using system environment variables")

from utils.logger import logger

class LLMCompletionCall:
    def __init__(self, max_retries: int = 5, initial_retry_delay: float = 1.0):
        """
        Initialize LLM completion call with retry mechanism.
        
        Args:
            max_retries: Maximum number of retries for API calls (default: 5)
            initial_retry_delay: Initial delay between retries in seconds (default: 1.0)
        """
        self.llm_model = os.getenv("UTU_LLM_MODEL", "deepseek-v3-0324")
        self.llm_base_url = os.getenv("UTU_LLM_BASE_URL", "https://api.lkeap.cloud.tencent.com/v1")
        self.llm_api_key = os.getenv("UTU_LLM_API_KEY", "")
        # 检测是否为腾讯云托管模型（不需要API key）
        is_tencent_hosted_no_auth = ('ti.tencentcs.com' in self.llm_base_url or 
                                     'ms-' in self.llm_model.lower())
        if not self.llm_api_key and not is_tencent_hosted_no_auth:
            raise ValueError("LLM API key not provided")
        self.openai_provider = os.getenv("OPENAI_PROVIDER", "openai").lower()
        
        # Retry configuration
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        
        # Initialize client based on available libraries
        if OpenAI is not None and AzureOpenAI is not None:
            if self.openai_provider == "azure":
                self.api_version = os.getenv("API_VERSION", "2025-01-01-preview")
                self.client = AzureOpenAI(
                        azure_endpoint=self.llm_base_url,
                        api_key=self.llm_api_key,
                        api_version=self.api_version,
                    )
            else:
                self.client = OpenAI(base_url=self.llm_base_url, api_key = self.llm_api_key)
        else:
            # Fallback to requests-based implementation
            self.client = None

    def call_api(self, content: str) -> str:
        """
        Call API to generate text with retry mechanism.
        
        Args:
            content: Prompt content
            
        Returns:
            Generated text response
        """
        # 检测是否使用腾讯云托管的小模型（utuv1, qwen3, ms-* 等，需要特殊参数）
        is_tencent_hosted = ('utuv1' in self.llm_model.lower() or 
                             'qwen' in self.llm_model.lower() or
                             'ms-' in self.llm_model.lower() or
                             'ti.tencentcs.com' in self.llm_base_url)
        
        for attempt in range(self.max_retries + 1):
            try:
                if is_tencent_hosted:
                    # 腾讯云托管模型使用 requests 调用，支持特殊参数
                    return self._call_utuv1_api(content)
                elif self.client is not None:
                    # Use OpenAI client
                    completion = self.client.chat.completions.create(
                        model=self.llm_model,
                        messages=[{"role": "user", "content": content}],
                        temperature=0.3
                    )
                    raw = completion.choices[0].message.content or ""
                    clean_completion = self._clean_llm_content(raw)
                    return clean_completion
                else:
                    # Use requests-based fallback
                    return self._call_api_with_requests(content)
                    
            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit = any(keyword in error_msg for keyword in [
                    'rate limit', 'rate_limit', '429', 'too many requests', 
                    'quota', 'throttle', 'concurrent'
                ])
                
                if attempt < self.max_retries:
                    # Calculate exponential backoff with jitter
                    delay = self.initial_retry_delay * (2 ** attempt)
                    # Add jitter (random 0-50% of delay)
                    jitter = random.uniform(0, delay * 0.5)
                    total_delay = delay + jitter
                    
                    if is_rate_limit:
                        # For rate limit errors, use longer delay
                        total_delay = max(total_delay, 5.0 + random.uniform(0, 3.0))
                        logger.warning(f"Rate limit hit, retrying in {total_delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    else:
                        logger.warning(f"API call failed: {e}, retrying in {total_delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    
                    time.sleep(total_delay)
                else:
                    logger.error(f"LLM API call failed after {self.max_retries} retries. Error: {e}")
                    raise e

    def _clean_llm_content(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)
        # 去掉空的思考标签 <think></think>
        t = re.sub(r"<think>\s*</think>", "", t, flags=re.IGNORECASE)
        # 去掉首尾成对的 <think>...</think> 包裹层
        t = re.sub(r"^\s*<think>([\s\S]*?)</think>\s*", r"\1", t, flags=re.IGNORECASE)
        # 再次收尾去空白
        t = t.strip()

        if "```json" in t:
            t = t.split("```json", 1)[1].split("```", 1)[0].strip()
            pass
        elif "```python" in t:
            t = t.split("```python", 1)[1].split("```", 1)[0].strip()
            pass

        # 若返回包含解释性文本，尝试提取首个 JSON 对象，减少解析失败
        # 简单括号计数匹配首个 { ... }
        # if "{" in t:
        #     start = t.find("{")
        #     brace = 0
        #     end = -1
        #     for i in range(start, len(t)):
        #         if t[i] == "{":
        #             brace += 1
        #         elif t[i] == "}":
        #             brace -= 1
        #             if brace == 0:
        #                 end = i
        #                 break
        #     if end != -1:
        #         candidate = t[start : end + 1].strip()
        #         # 只在 candidate 看起来像 JSON 时替换
        #         if candidate.startswith("{") and candidate.endswith("}"):
        #             t = candidate
        # fence_re = re.compile(r"^\s*```(?:\s*\w+)?\s*\n(?P<body>[\s\S]*?)\n\s*```\s*$", re.MULTILINE)
        # m = fence_re.match(t)
        # if m:
        #     t = m.group("body").strip()
        # else:
        #     if t.startswith("```") and t.endswith("```") and len(t) >= 6:
        #         t = t[3:-3].strip()

        # if t.lower().startswith("json\n"):
        #     t = t.split("\n", 1)[1].strip()

        return t
    
    def _call_utuv1_api(self, content: str) -> str:
        """
        Call utuv1-2b/qwen3/ms-* model with specific parameters.
        
        Args:
            content: Prompt content
            
        Returns:
            Generated text response
        """
        url = f"{self.llm_base_url}/chat/completions"
        
        # 根据模型类型选择参数
        # Qwen3-4B 最大context length是40960，需要合理设置max_tokens
        if 'qwen3' in self.llm_model.lower() or 'qwen' in self.llm_model.lower():
            # Qwen3-4B模型：最大40960 tokens，进一步收紧max_tokens，减少超长风险
            payload = {
                "model": self.llm_model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.3,
                "max_tokens": 8000,  # 更保守，给messages留足余量
                "stream": False
            }
        elif 'ms-' in self.llm_model.lower():
            # ms-* 模型使用简单参数
            payload = {
                "model": self.llm_model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.3,
                "max_tokens": 8000,  # 保守设置
                "stream": False
            }
        else:
            # utuv1 等模型使用推荐参数
            payload = {
                "model": self.llm_model,
                "messages": [{"role": "user", "content": content}],
                # utuv1-2b 推荐参数
                "temperature": 0.6,
                "max_tokens": 32768,
                "top_p": 0.95,
                "top_k": 20,
                "presence_penalty": 1.5,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        
        # 对于不需要认证的端点，不发送Authorization header
        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"
        
        # timeout 收紧到 60s，避免长时间阻塞
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                raw = result["choices"][0]["message"]["content"] or ""
                clean_completion = self._clean_llm_content(raw)
                return clean_completion
            else:
                raise Exception(f"Unexpected response format: {result}")
        elif response.status_code == 429:
            raise Exception(f"Rate limit exceeded (429): {response.text}")
        else:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    
    def _call_api_with_requests(self, content: str) -> str:
        """
        Fallback method using requests for older OpenAI library versions.
        Note: This method is called by call_api which handles retries.
        
        Args:
            content: Prompt content
            
        Returns:
            Generated text response
        """
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.3
        }
        
        response = requests.post(
            f"{self.llm_base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=60  # Increased timeout for concurrent requests
        )
        
        if response.status_code == 200:
            result = response.json()
            raw = result["choices"][0]["message"]["content"] or ""
            clean_completion = self._clean_llm_content(raw)
            return clean_completion
        elif response.status_code == 429:
            # Rate limit error - will be caught and retried by call_api
            raise Exception(f"Rate limit exceeded (429): {response.text}")
        else:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")