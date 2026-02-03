"""
ADG Benchmark Runner V4 - 分步执行 + 丰富上下文
"""

import os
import sys
import json
import datetime
import argparse
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
from threading import Lock

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.model.adg_benchmark.adg_workflow_v4 import build_adg_workflow_v4, ADGWorkflowStateV4
from utils.logger import logger

file_write_lock = Lock()

def load_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['queries']

def ensure_output_dir(model_name):
    base_out = project_root / 'output' / 'close_source'
    base_out.mkdir(parents=True, exist_ok=True)
    model_out = base_out / model_name
    model_out.mkdir(parents=True, exist_ok=True)
    return model_out

def print_separator(title=None):
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print(f"{'='*60}")

def process_single_query(query, table_dir, workflow, output_file, pbar=None):
    """处理单个查询"""
    try:
        qid = query['id']
        question = query['Question']
        file_name = query['FileName']
        question_type = query.get('QuestionType', '')
        
        file_path = table_dir / f"{file_name}.xlsx"
        
        initial_state = ADGWorkflowStateV4(
            user_question=question,
            question_type=question_type,
            current_table={
                "file_name": f"{file_name}.xlsx",
                "file_path": str(file_path)
            },
        )
        
        config = {"recursion_limit": 60, "configurable": {"thread_id": str(qid)}}

        try:
            final_state = workflow.invoke(initial_state, config=config)
            final_answer = final_state.get("final_answer", "")
            execution_trace = final_state.get("execution_trace", [])
        except Exception as invoke_e:
            if "Recursion limit" in str(invoke_e):
                logger.warning(f"Query {qid} hit recursion limit.")
                final_answer = "[Final Answer]: Processing timeout"
                execution_trace = []
            else:
                raise invoke_e
        
        # Visualization 处理
        if question_type == 'Visualization':
            all_code = "\n".join([t.get("code", "") for t in execution_trace if t.get("code")])
            if "plt." not in final_answer and "plt." in all_code:
                final_answer = all_code
            if "plt.show()" not in final_answer:
                final_answer += "\nplt.show()"
        
        rec = {
            'id': qid,
            'QuestionType': question_type,
            'SubQType': query.get('SubQType', ''),
            'FileName': file_name,
            'model_answer': final_answer,
            'prompt': ''
        }
        
        # Structure Comprehending Swap
        if question_type == 'Structure Comprehending':
            swap_file_name = f"{file_name}_swap"
            swap_file_path = table_dir / f"{swap_file_name}.xlsx"
            
            if swap_file_path.exists():
                swap_state = ADGWorkflowStateV4(
                    user_question=question,
                    question_type=question_type,
                    current_table={
                        "file_name": f"{swap_file_name}.xlsx",
                        "file_path": str(swap_file_path)
                    }
                )
                swap_final = workflow.invoke(swap_state, config=config)
                rec['model_answer_swap'] = swap_final.get("final_answer", "")
            else:
                rec['model_answer_swap'] = ""
        
        with file_write_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        
        if pbar:
            pbar.update(1)
            
        return True
            
    except Exception as e:
        logger.error(f"Error processing query {query.get('id')}: {e}")
        import traceback
        traceback.print_exc()
        
        err_rec = {
            'id': query.get('id'),
            'QuestionType': query.get('QuestionType'),
            'model_answer': f"[Final Answer]: Error - {str(e)[:100]}",
            'error': str(e)
        }
        with file_write_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(err_rec, ensure_ascii=False) + '\n')
        
        if pbar:
            pbar.update(1)
            
        return False

def run_inference(dataset_name='final', max_questions=None, max_workers=10, data_sample_rows=150, question_types=None, start_id=None):
    """运行推理
    
    Args:
        dataset_name: 数据集名称 ('final' or 'long')
        max_questions: 最大问题数
        max_workers: 并发线程数
        data_sample_rows: 展示给 LLM 的数据行数
        question_types: 要处理的题型列表，如 ['Visualization', 'Structure Comprehending']
        start_id: 起始问题ID，只处理 id >= start_id 的问题
    """
    if dataset_name == 'long':
        dataset_file = 'QA_long.json'
    else:
        dataset_file = 'QA_final.json'
        
    dataset_path = project_root / 'benchmarks' / 'realhitbench' / dataset_file
    table_dir = project_root / 'benchmarks' / 'realhitbench' / 'tables'
    meta_graphs_dir = project_root / 'output' / 'realhitbench' / 'tables'
    
    print_separator(f"Loading Dataset: {dataset_file}")
    queries = load_dataset(dataset_path)
    
    # 按起始ID筛选
    if start_id is not None:
        original_count = len(queries)
        queries = [q for q in queries if q.get('id', 0) >= start_id]
        logger.info(f"Filtered by start_id {start_id}: {original_count} -> {len(queries)} queries")
        print(f"  Filtered: start_id >= {start_id} ({len(queries)} questions)")
    
    # 按题型筛选
    if question_types:
        original_count = len(queries)
        queries = [q for q in queries if q.get('QuestionType', '') in question_types]
        logger.info(f"Filtered by question types {question_types}: {original_count} -> {len(queries)} queries")
        print(f"  Filtered: {question_types} ({len(queries)} questions)")
    
    if max_questions:
        queries = queries[:max_questions]
        logger.info(f"Limited to {max_questions} queries")
        
    print_separator(f"Initializing ADG Workflow V4 (data_sample_rows={data_sample_rows})")
    workflow, instance = build_adg_workflow_v4(str(meta_graphs_dir), data_sample_rows=data_sample_rows)
    
    model_name = 'adg_v4'
    out_dir = ensure_output_dir(model_name)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = out_dir / f"predictions_{model_name}_{timestamp}.jsonl"
    
    print_separator(f"Starting Inference (Workers: {max_workers})")
    logger.info(f"Total queries: {len(queries)}")
    
    with tqdm(total=len(queries), desc="Processing") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_query, query, table_dir, workflow, output_file, pbar)
                for query in queries
            ]
            
            success_count = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    if future.result():
                        success_count += 1
                except Exception as exc:
                    logger.error(f"Thread exception: {exc}")
    
    print_separator("Inference Complete")
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Success: {success_count}/{len(queries)} ({100*success_count/len(queries):.1f}%)")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADG Benchmark Runner V4")
    parser.add_argument('--dataset', type=str, default='final', choices=['final', 'long'])
    parser.add_argument('--max_questions', type=int, default=10)
    parser.add_argument('--max_workers', type=int, default=10)
    parser.add_argument('--data_sample_rows', type=int, default=150,
                        help='Number of rows to show LLM (default: 150)')
    parser.add_argument('--question_types', type=str, nargs='+', default=None,
                        help='Filter by question types, e.g.: --question_types Visualization "Structure Comprehending"')
    parser.add_argument('--start_id', type=int, default=None,
                        help='Start from this question ID (only process queries with id >= start_id)')
    args = parser.parse_args()
    
    run_inference(args.dataset, args.max_questions, args.max_workers, args.data_sample_rows, args.question_types, args.start_id)

