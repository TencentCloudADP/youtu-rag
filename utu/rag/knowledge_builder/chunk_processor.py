"""HiChunk Document Processor

Uses HiChunk model to perform semantic hierarchical segmentation on long documents,
generating Markdown documents with title hierarchy.

1. Receive text content (from OCR or raw text extraction)
2. Call HiChunk model for hierarchical recognition
3. Return Markdown document with hierarchical structure including level-1 headings, level-2 headings, etc.

"""

import logging
import os
import re
import time
from typing import Optional

import openai
import requests
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


def replace_jinhao(line, replacement=None):
    if replacement is not None and re.match(r'^( *#*)*', line)[0].strip() != '':
        return re.sub(r'^( *#*)*', replacement, line, count=1)
    else:
        return line


def count_jinhao(line):
    return re.match(r'^( *#*)*', line)[0].count('#')


def is_english(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def sentence_split_en(line):
    res = filter(lambda l: l.strip() != '', sent_tokenize(line))
    res = list(map(lambda l: l.strip(), res))
    idx = 0
    while idx < len(res) - 1:
        if len(res[idx]) < 10: 
            res[idx + 1] = res[idx] + ' ' + res[idx + 1]
            res.pop(idx)
        else:
            idx += 1
    return res


def sentence_split_zh(line):
    res = []
    pre_idx = 0
    for i in range(1, len(line)-1):
        if line[i] != '。':  
            continue
        if len(line[pre_idx: i + 1].strip()) <= 5:
            continue

        res.append(line[pre_idx: i + 1].strip())
        pre_idx = i + 1

    if pre_idx < len(line):
        res.append(line[pre_idx:])

    return res



def sentence_split(line):
    if is_english(line):
        return sentence_split_en(line)
    else:
        return sentence_split_zh(line)


def sentence_truncation(line, head_limit=15, tail_limit=15):
    total_limit = head_limit+tail_limit
    if is_english(line):
        len_factor = 10
    else:
        len_factor = 1

    if 0 < total_limit * len_factor < len(line):
        _head_limit = head_limit * len_factor
        _tail_limit = len(line) - tail_limit * len_factor
        line = line[:_head_limit] + line[_tail_limit:]
    return line


def text2sentence(lines, replacement=None, head_limit=15, tail_limit=15):
    res = []
    for idx, line in enumerate(lines):
        res.extend(sentence_split(line))

    for idx, temp in enumerate(res):
        _temp = replace_jinhao(temp, '# ')
        _temp = sentence_truncation(_temp, head_limit, tail_limit)
        _temp = replace_jinhao(_temp, f"{'#'*count_jinhao(temp)} ")
        _temp = replace_jinhao(_temp, replacement)
        res[idx] = _temp+'\n'
    return res


PROMPT = ('You are an assistant good at reading and formatting documents, and you are also skilled at distinguishing '
          'the semantic and logical relationships of sentences between document context. The following is a text that '
          'has already been divided into sentences. Each line is formatted as: "{line number} @ {sentence content}". '
          'You need to segment this text based on semantics and format. There are multiple levels of granularity for '
          'segmentation, the higher level number means the finer granularity of the segmentation. Please ensure that '
          'each Level One segment is semantically complete after segmentation. A Level One segment may contain '
          'multiple Level Two segments, and so on. Please incrementally output the starting line numbers of each level '
          'of segments, and determine the level of the segment, as well as whether the content of the sentence at the '
          'starting line number can be used as the title of the segment. Finally, output a list format result, '
          'where each element is in the format of: "{line number}, {segment level}, {be a title?}".'
          '\n\n>>> Input text:\n')


def index_format(idx, line):
    return f'{idx} @ {line}'


def points2clip(points, start_idx, end_idx):
    clips = []
    pre_p = start_idx
    for p in points:
        if p == start_idx or p >= end_idx:
            continue
        clips.append([pre_p, p])
        pre_p = p
    clips.append([pre_p, end_idx])
    return clips


def parse_answer_chunking_point(answer_string, max_level):
    level_dict_en = {
        0: 'Level One', 1: 'Level Two', 2: 'Level Three', 3: 'Level Four', 4: 'Level Five',
        5: 'Level Six', 6: 'Level Seven', 7: 'Level Eight', 8: 'Level Nine', 9: 'Level Ten',
    }
    local_chunk_points = {level_dict_en[i]: [] for i in range(max_level)}

    if not answer_string:
        return list(local_chunk_points.values())
    
    for line in answer_string.split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split(', ')
        if len(parts) < 3:
            continue
        point, level = parts[0], parts[1]
        if level in local_chunk_points:
            try:
                local_chunk_points[level].append(int(point))
            except ValueError:
                continue

    res = list(local_chunk_points.values())
    for idx, _ in enumerate(res):
        if len(_) == 0:
            continue
        keep_idx = list(filter(lambda i: _[i] > _[i-1], range(1, len(_))))
        res[idx] = [_[0]] + list(map(lambda i: _[i], keep_idx))
    return res


def check_answer_point(first_level_points, start_idx, end_idx):
    if len(first_level_points) > 0 and first_level_points[0] < start_idx:
        return False
    for idx in range(1, len(first_level_points)):
        p = first_level_points[idx]
        if p <= first_level_points[idx-1] or p > end_idx:
            return False
    return True


def build_residual_lines(lines, global_chunk_points, start_idx, window_size, recurrent_type):
    if recurrent_type in [0, 1]:
        return []
    assert recurrent_type == 2, f'Not implemented for recurrent_type: {recurrent_type}'

    last_first_point = 0
    if len(global_chunk_points[0]) > 0:
        last_first_point = global_chunk_points[0][-1]
    current_second_points = filter(lambda p: p >= last_first_point, global_chunk_points[1])
    temp_second_clips = points2clip(current_second_points, last_first_point, start_idx)

    pre_seg_num, post_seg_num, line_num = 2, 3, 20
    while True:
        residual_second_clips = temp_second_clips
        if len(temp_second_clips) > (pre_seg_num + post_seg_num):
            residual_second_clips = (
                    temp_second_clips[:pre_seg_num] + temp_second_clips[len(temp_second_clips)-post_seg_num:]
            )
        residual_lines = []
        for rsc in residual_second_clips:
            pre_sent_idx, post_sent_idx = rsc[0], min(rsc[1], rsc[0]+line_num)
            residual_lines.extend(lines[pre_sent_idx: post_sent_idx])
        if len('\n'.join(residual_lines)) < window_size/2:
            return residual_lines

        pre_seg_num, post_seg_num, line_num = pre_seg_num-1, post_seg_num-1, line_num-5
        if pre_seg_num * post_seg_num * line_num <= 0:
            return []


def union_chunk_points(local_chunk_points, global_chunk_points, max_idx):
    for idx, level in enumerate(global_chunk_points):
        global_chunk_points[idx].extend(filter(lambda p: p < max_idx, local_chunk_points[idx]))
    return global_chunk_points


class HiChunkInferenceEngine:
    def __init__(self, window_size, line_max_len, max_level, prompt, base_url):
        self.window_size = window_size
        self.line_max_len = line_max_len
        self.max_level = max_level
        self.prompt = prompt
        self.base_url = base_url
        self.llm = openai.Client(
            base_url=f"{self.base_url}/v1",
            api_key="[empty]",
            timeout=120.0
        )

    def init_chunk_points(self):
        return [[] for i in range(self.max_level)]

    def build_input_instruction(self, prompt, global_start_idx, sentences, window_size, residual_lines=None):
        q = prompt
        residual_index = 0
        while residual_lines is not None and residual_index < len(residual_lines):
            line_text = index_format(residual_index, residual_lines[residual_index])
            q += line_text
            residual_index += 1
        assert self.count_length(q) <= window_size, 'residual lines exceeds window size'

        local_start_idx = 0
        cur_token_num = self.count_length(q)
        end = False
        while global_start_idx < len(sentences):
            line_text = index_format(local_start_idx + residual_index, sentences[global_start_idx])
            line_token_num = self.count_length(line_text)
            if cur_token_num + line_token_num > window_size:
                break
            cur_token_num += line_token_num
            q += line_text
            local_start_idx += 1
            global_start_idx += 1
        if global_start_idx == len(sentences):
            end = True
        return q, end, local_start_idx

    def call_llm(self, input_text, max_retries=3, retry_delay=2):
        logger.debug(f"call_llm input length: {len(input_text)}")

        for attempt in range(max_retries):
            try:
                response = self.llm.chat.completions.create(
                    model='HiChunk',
                    messages=[{'role': 'user', 'content': input_text}],
                    temperature=0.0,
                    max_tokens=4096,
                    extra_body={
                        "chat_template_kwargs": {"add_generation_prompt": True, "enable_thinking": False}
                    }
                )
                result = response.choices[0].message.content
                logger.debug(f"call_llm response length: {len(result) if result else 0}")
                logger.debug(f"call_llm response preview: {result[:500] if result else 'None'}")
                return result

            except openai.APIStatusError as e:
                if e.status_code == 503:
                    if attempt < max_retries - 1:
                        logger.warning(f"HiChunk LLM service unavailable (503), retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"HiChunk LLM service unavailable after {max_retries} retries")
                        raise
                else:
                    logger.error(f"LLM API error: {e}")
                    raise
            except openai.APITimeoutError:
                if attempt < max_retries - 1:
                    logger.warning(f"LLM request timeout, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error calling LLM: {str(e)}")
                raise

        raise Exception(f"Failed to call LLM after {max_retries} retries")

    def count_length(self, text, max_retries=3, retry_delay=2):
        for attempt in range(max_retries):
            try:
                raw_response = requests.post(
                    url=f'{self.base_url}/tokenize',
                    json={'model': 'HiChunk', 'prompt': text},
                    timeout=30
                )

                if attempt == 0: 
                    logger.debug(f"tokenize API response text: {raw_response.text[:500]}")

                if raw_response.status_code == 503:
                    if attempt < max_retries - 1:
                        logger.warning(f"HiChunk service unavailable (503), retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise Exception(f"HiChunk service unavailable after {max_retries} retries")

                raw_response.raise_for_status() 

                response = raw_response.json()
                return response['count']

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.warning(f"Request timeout, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to parse tokenize response. Error: {str(e)}, Status: {raw_response.status_code if 'raw_response' in locals() else 'N/A'}, Content: {raw_response.text[:1000] if 'raw_response' in locals() else 'N/A'}")
                raise

        raise Exception(f"Failed to get token count after {max_retries} retries")

    def pre_process(self, document):
        lines = map(lambda l: l.strip(), document.split('\n'))
        lines = list(filter(lambda l: len(l) != 0, lines))
        origin_lines = text2sentence(lines, None, -1, 0)
        input_lines = text2sentence(lines, '', self.line_max_len, 0)
        return input_lines, origin_lines

    @staticmethod
    def post_process(origin_lines, global_chunk_points):
        origin_lines_remove_jinhao = [replace_jinhao(l, '') for l in origin_lines]
        total_points = sorted(
            [[__, i + 1] for i, _ in enumerate(global_chunk_points) for __ in _],
            key=lambda p: p[0]
        )
        splits = []
        pre_level, pre_point = 1, 0
        for i, [p, level] in enumerate(total_points):
            if p == 0:
                continue
            splits.append([''.join(origin_lines_remove_jinhao[pre_point: p]), pre_level])
            pre_level = level
            pre_point = p
        splits.append([''.join(origin_lines_remove_jinhao[pre_point:]), pre_level])
        return splits

    def iterative_inf(self, lines, recurrent_type=1):
        error_count, start_idx = 0, 0
        raw_qa, residual_lines = [], []
        global_chunk_points = self.init_chunk_points()
        while start_idx < len(lines):
            residual_sent_num = len(residual_lines)
            question, is_end, question_sent_num = self.build_input_instruction(
                self.prompt, start_idx, lines, self.window_size, residual_lines
            )
            question_token_num = self.count_length(question)
            logger.debug(f'question len: {len(question)}, {question_token_num}')
            start_time = time.time()
            answer = self.call_llm(question)
            inf_time = time.time() - start_time
            answer_token_num = self.count_length(answer)
            logger.debug(f'answer len: {answer_token_num}')

            tmp = {
                'question': question, 'answer': answer, 'start_idx': start_idx, 'end_idx': start_idx+question_sent_num,
                'residual_sent_num': residual_sent_num, 'time': inf_time,
                'question_token_num': question_token_num, 'answer_token_num': answer_token_num,
            }

            try:
                logger.debug(f"Parsing answer, length: {len(answer) if answer else 0}, content: {answer[:500] if answer else 'None'}")
                local_chunk_points = parse_answer_chunking_point(answer, self.max_level)
                if not check_answer_point(local_chunk_points[0], 0, question_sent_num+residual_sent_num-1):
                    logger.warning('Chunk check error')
                    tmp['status'] = 'check error'
                    local_chunk_points = self.init_chunk_points()
                    local_chunk_points[0].append(start_idx)
                    error_count += 1
                else:
                    tmp['status'] = 'check ok'
                    for idx, points in enumerate(local_chunk_points):
                        filter_points = filter(lambda p: p >= residual_sent_num, points)
                        local_chunk_points[idx] = [p - residual_sent_num + start_idx for p in filter_points]
            except Exception as e:
                logger.warning(f'Chunk parse error: {str(e)}', exc_info=True)
                logger.debug(f"Failed answer content: {answer}")
                tmp['status'] = 'parse error'
                local_chunk_points = self.init_chunk_points()
                local_chunk_points[0].append(start_idx)
                error_count += 1

            raw_qa.append(tmp)

            if is_end:
                start_idx += question_sent_num
                global_chunk_points = union_chunk_points(local_chunk_points, global_chunk_points, start_idx)
                break
            if len(local_chunk_points[0]) > 1 and recurrent_type in [1, 2]:
                start_idx = local_chunk_points[0][-1]
                global_chunk_points = union_chunk_points(local_chunk_points, global_chunk_points, start_idx)
                residual_lines = []
            else:
                start_idx += question_sent_num
                global_chunk_points = union_chunk_points(local_chunk_points, global_chunk_points, start_idx)
                residual_lines = build_residual_lines(
                    lines, global_chunk_points, start_idx, self.window_size, recurrent_type
                )

        return {
            'global_chunk_points': global_chunk_points,
            'raw_qa': raw_qa,
            'error_count': error_count,
        }

    def inference(self, document, recurrent_type=1):
        input_lines, origin_lines = self.pre_process(document)
        chunked_result = self.iterative_inf(input_lines, recurrent_type=recurrent_type)
        chunks = self.post_process(origin_lines, chunked_result['global_chunk_points'])
        chunked_document = '\n'.join(['#'*c[1] + ' ' + c[0] for c in chunks])
        logger.info(f"chunked_document length: {len(chunked_document)}, preview: {chunked_document[:2000]}...")
        return chunked_document


class ChunkProcessor:
    def __init__(self, chunk_config: dict):
        self.enabled = chunk_config.get("enabled", False)

        if not self.enabled:
            logger.warning("ChunkProcessor initialized but chunk.enabled=False")
            return

        chunk_base_url = chunk_config.get("base_url")
        chunk_model = chunk_config.get("model")

        fallback_base_url = os.getenv("UTU_LLM_BASE_URL")
        fallback_model = os.getenv("UTU_LLM_MODEL")

        if chunk_base_url and chunk_model:
            self.base_url = chunk_base_url
            self.model = chunk_model
            self.mode = "hichunk"
            logger.info(f"✓ Using dedicated HiChunk model: {self.model}")
        elif fallback_base_url and fallback_model:
            self.base_url = fallback_base_url
            self.model = fallback_model
            self.mode = "llm_fallback"
            logger.info(f"✓ Using fallback LLM for chunk: {self.model} (专用 chunk 模型未配置)")
        else:
            raise ValueError(
                "Chunk processing requires either:\n"
                "  1. chunk.base_url + chunk.model (dedicated)\n"
                "  2. UTU_LLM_BASE_URL + UTU_LLM_MODEL (fallback)"
            )

        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                logger.info("Downloading nltk punkt_tab data...")
                nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to initialize nltk: {e}. English text processing may fail.")

        self.engine = HiChunkInferenceEngine(
            window_size=16384,
            line_max_len=100,
            max_level=10,
            prompt=PROMPT,
            base_url=self.base_url
        )

        logger.info(f"✓ ChunkProcessor initialized: {self.base_url} (mode: {self.mode})")

    async def chunk_document(self, text: str) -> Optional[str]:
        """
        Intelligently chunk document into hierarchical sections

        Args:
            text: Input text (can be multi-page merged text)

        Returns:
            str: Chunked Markdown document (with hierarchical headings)
            None: If processing fails

        Examples:
            Input:
                "Chapter 1 Introduction\nThis is introduction content...\nChapter 2 Methods\nThis is methods..."

            Output:
                "# Chapter 1 Introduction\nThis is introduction content...\n## Chapter 2 Methods\nThis is methods..."
        """
        if not self.enabled:
            logger.warning("Chunk processing is disabled (enabled=False)")
            return None

        if not text or len(text.strip()) < 50:
            logger.warning(f"Text too short for chunking: {len(text)} chars")
            return None

        try:
            logger.info(f"Starting HiChunk processing: {len(text)} chars")


            import asyncio
            chunked_text = await asyncio.to_thread(
                self.engine.inference, text, recurrent_type=2
            )

            if not chunked_text:
                logger.error("HiChunk model returned empty response")
                return None

            logger.info(f"✓ HiChunk processing completed: {len(chunked_text)} chars output")
            return chunked_text

        except Exception as e:
            logger.error(f"HiChunk processing failed: {str(e)}", exc_info=True)
            return None

    def validate_config(self) -> bool:
        if not self.enabled:
            return True  # Validation passes when disabled

        if not self.base_url:
            logger.error("chunk.base_url is required when enabled=True")
            return False

        if not self.model:
            logger.error("chunk.model is required when enabled=True")
            return False

        return True
