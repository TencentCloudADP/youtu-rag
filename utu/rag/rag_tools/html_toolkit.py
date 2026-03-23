"""HTML Toolkit for generating HTML dashboard from configuration."""

import json
import logging
import os
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
from jinja2 import Template

from ...config import ToolkitConfig
from ...tools.base import register_tool, TOOL_PROMPTS
from ...utils import SimplifiedAsyncOpenAI, LLMOutputParser
from .base_toolkit import BaseRAGToolkit

logger = logging.getLogger(__name__)


class RowsConfigValidator:
    """配置验证器，验证 rows 配置的合法性"""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证配置的合法性
        
        Args:
            config: 看板配置字典
            
        Returns:
            验证后的配置
            
        Raises:
            ValueError: 配置不合法时抛出异常
        """
        if 'rows' not in config:
            raise ValueError("配置必须包含 'rows' 字段")
        
        rows = config['rows']
        if not isinstance(rows, list):
            raise ValueError("'rows' 必须是列表")
        
        for row_idx, row in enumerate(rows):
            if 'modules' not in row:
                raise ValueError(f"第{row_idx+1}行缺少 'modules' 字段")
            
            modules = row['modules']
            if not isinstance(modules, list):
                raise ValueError(f"第{row_idx+1}行的 'modules' 必须是列表")
            
            # 验证总宽度不超过12
            total_span = sum(m.get('span', 12) for m in modules)
            if total_span > 12:
                raise ValueError(f"第{row_idx+1}行的模块总宽度({total_span})超过12")
        
        return config


def calculate_grid_columns(modules: List[Dict[str, Any]]) -> str:
    """
    计算CSS Grid的grid-template-columns值
    
    Args:
        modules: 模块列表
        
    Returns:
        CSS grid-template-columns值，如 "3fr 6fr 3fr"
    """
    spans = [module['span'] for module in modules]
    return ' '.join([f"{span}fr" for span in spans])


class DashboardRenderer:
    """
    看板渲染引擎 - 动态布局版
    
    使用方式:
        config = {
            "title": "数据看板",
            "subtitle": "2024年度数据",
            "update_time": "2025-01-09",
            "rows": [
                {
                    "modules": [
                        {"type": "kpi", "span": 3, "data": {...}},
                        {"type": "pie", "span": 9, "data": {...}}
                    ]
                }
            ]
        }
        html = DashboardRenderer.render(config)
    """
    TEMPLATE = None

    def __init__(self, template_path: Optional[str] = None, template: Optional[str] = None):
        self.template_path = template_path
        self.template = template
        self._template_loaded = False
    
    def _load_template(self):
        """加载HTML模板
        
        从 self.template_path 或 self.template 加载模板内容。
        如果 self.template 已经提供，直接使用；否则从文件加载。
        """
        if not self._template_loaded:
            # 如果直接提供了 template 字符串，使用它
            if self.template:
                self.__class__.TEMPLATE = self.template
                logger.info("Successfully loaded HTML template from config string")
            # 否则从文件加载
            elif self.template_path:
                template_path = Path(self.template_path)
                assert template_path.exists(), f"Template file {self.template_path} does not exist"
                
                try:
                    with open(self.template_path, 'r', encoding='utf-8') as f:
                        template_data = yaml.safe_load(f)
                        self.__class__.TEMPLATE = template_data.get('default', '')
                    logger.info(f"Successfully loaded HTML template from {self.template_path}")
                except Exception as e:
                    logger.error(f"Failed to load HTML template from {self.template_path}: {e}")
                    raise RuntimeError(f"Failed to load HTML template from {self.template_path}: {e}")
            else:
                raise ValueError("Either template_path or template must be provided in config")
            
            self._template_loaded = True

    def render(self, config: Dict[str, Any]) -> str:
        """
        渲染看板HTML（统一渲染入口）
        
        Args:
            config: 看板配置字典（纯字典格式，不包含任何可执行代码）
                
        Returns:
            完整的HTML字符串
        """
        # 加载模板（如果尚未加载）
        self._load_template()
        
        # 验证配置
        validated_config = RowsConfigValidator.validate_config(config)
        
        # 标准化每个模块的数据
        for row in validated_config['rows']:
            for module in row['modules']:
                module['data'] = self._normalize_module_data(module['type'], module['data'])
            # 计算grid-template-columns
            row['grid_columns'] = calculate_grid_columns(row['modules'])
        
        # 渲染模板
        template = Template(self.__class__.TEMPLATE)
        return template.render(config=validated_config)
    
    @staticmethod
    def _normalize_module_data(module_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化和验证模块数据（内部方法）
        
        Args:
            module_type: 模块类型
            data: 原始数据字典
            
        Returns:
            标准化后的数据字典
        """
        normalizers = {
            'kpi': DashboardRenderer._normalize_kpi_card,
            'pie': DashboardRenderer._normalize_pie_chart,
            'line': DashboardRenderer._normalize_line_chart,
            'bar': DashboardRenderer._normalize_bar_chart,
            'column': DashboardRenderer._normalize_column_chart,
            'metric_cards': DashboardRenderer._normalize_metric_cards,
            'insights': DashboardRenderer._normalize_insights,
            'gradient_text': DashboardRenderer._normalize_gradient_text,
            'stats_card': DashboardRenderer._normalize_stats_card,
            'quote': DashboardRenderer._normalize_quote,
            'progress': DashboardRenderer._normalize_progress,
            'notification': DashboardRenderer._normalize_notification,
            'timeline': DashboardRenderer._normalize_timeline,
            'feature': DashboardRenderer._normalize_feature,
            'radar': DashboardRenderer._normalize_radar,
            'text': lambda d: d,  # text 和 table 直接返回原数据
            'table': lambda d: d,
        }
        
        normalizer = normalizers.get(module_type)
        if normalizer:
            return normalizer(data)
        else:
            raise ValueError(f"不支持的模块类型: {module_type}")
    
    # ========== 内部数据标准化方法（原 create 方法改为内部方法） ==========
    
    @staticmethod
    def _normalize_kpi_card(data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化KPI卡片数据"""
        color_map = {
            'blue': 'bg-blue-card',
            'orange': 'bg-orange-card',
            'purple': 'bg-purple-card',
            'red': 'bg-red-card'
        }
        
        # 转换color为color_class
        color = data.get('color', 'blue')
        result = data.copy()
        result['color_class'] = color_map.get(color, 'bg-blue-card')
        
        # 设置默认值
        result.setdefault('sub_label', '')
        result.setdefault('badge', None)
        
        return result
    
    @staticmethod
    def _normalize_pie_chart(data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化环形图数据"""
        result = data.copy()
        
        # 设置默认的center_value_formatter
        if 'center_value_formatter' not in result or result['center_value_formatter'] is None:
            result['center_value_formatter'] = """(v) => {
                if (v >= 1000000000000) {
                    // 万亿级别
                    const wanyi = v / 1000000000000;
                    return wanyi >= 10 ? wanyi.toFixed(1) + '万亿' : wanyi.toFixed(2) + '万亿';
                } else if (v >= 100000000) {
                    // 亿级别
                    const yi = v / 100000000;
                    return yi >= 100 ? yi.toFixed(1) + '亿' : yi.toFixed(2) + '亿';
                } else if (v >= 10000) {
                    // 万级别
                    const wan = v / 10000;
                    return wan >= 100 ? wan.toFixed(0) + '万' : wan.toFixed(2) + '万';
                } else {
                    // 小于1万，显示千分位
                    return v.toLocaleString('zh-CN');
                }
            }"""
        
        # 设置默认值
        result.setdefault('center_label', '总计')
        result.setdefault('series_name', '占比')
        
        return result
    
    @staticmethod
    def _normalize_line_chart(data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化折线图数据"""
        result = data.copy()
        
        # 处理系列数据，添加默认值
        processed_series = []
        default_colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#06b6d4']
        series = result.get('series', [])
        
        for i, s in enumerate(series):
            processed = s.copy()
            # 如果没有指定颜色，使用默认颜色
            if 'color' not in processed:
                processed['color'] = default_colors[i % len(default_colors)]
            
            # 如果启用面积但没有指定渐变色，自动生成
            if processed.get('showArea') and 'areaColorStart' not in processed:
                color = processed['color']
                processed['areaColorStart'] = f"{color}40"  # 25% 透明度
                processed['areaColorEnd'] = f"{color}00"    # 完全透明
            
            processed_series.append(processed)
        
        result['series'] = processed_series
        result.setdefault('xaxis_name', '')
        result.setdefault('yaxis_name', '')
        
        return result

    @staticmethod
    def _normalize_bar_chart(data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化条形图数据（自动排序）"""
        result = data.copy()
        
        # 按value排序
        if 'data' in result and isinstance(result['data'], list):
            result['data'] = sorted(result['data'], key=lambda x: x.get('value', 0))
        
        result.setdefault('label_formatter', '{c}%')
        
        return result
    
    @staticmethod
    def _normalize_column_chart(data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化柱状图数据"""
        result = data.copy()
        
        # 处理系列数据，添加默认值
        processed_series = []
        default_colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#06b6d4']
        series = result.get('series', [])
        
        for i, s in enumerate(series):
            processed = s.copy()
            # 如果没有指定颜色，使用默认颜色
            if 'color' not in processed:
                processed['color'] = default_colors[i % len(default_colors)]
            
            processed_series.append(processed)
        
        result['series'] = processed_series
        result.setdefault('xaxis_name', '')
        result.setdefault('yaxis_name', '')
        
        return result
    
    @staticmethod
    def _normalize_insights(data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化洞察列表数据"""
        result = data.copy()
        
        # 如果是单个洞察项，转为包含insights列表的格式
        if 'title' in result and 'description' in result and 'insights' not in result:
            # 单个洞察项，转为洞察列表格式
            result = {
                'title': result.get('list_title', '核心洞察'),
                'insights': [{
                    'title': result['title'],
                    'description': result['description'],
                    'color': result.get('color', '#3b82f6')
                }]
            }
        else:
            # 已经是洞察列表格式，确保每个insights项有color
            if 'insights' in result:
                for insight in result['insights']:
                    insight.setdefault('color', '#3b82f6')
        
        return result
    
    @staticmethod
    def _normalize_metric_cards(data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化描述性指标卡片数据"""
        return data  # 直接返回，无需额外处理
    
    @staticmethod
    def _normalize_gradient_text(data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化渐变文本卡片数据"""
        result = data.copy()
        result.setdefault('icon', None)
        return result
    
    @staticmethod
    def _normalize_stats_card(data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化统计数字卡片数据"""
        result = data.copy()
        result.setdefault('icon_color', '#3b82f6')
        result.setdefault('change', None)
        return result
    
    @staticmethod
    def _normalize_quote(data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化引用卡片数据"""
        return data  # 直接返回，无需额外处理
    
    @staticmethod
    def _normalize_progress(data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化进度卡片数据"""
        result = data.copy()
        
        # 将items转为progress_items（避免与dict.items()冲突）
        if 'items' in result and 'progress_items' not in result:
            result['progress_items'] = result.pop('items')
        
        return result
    
    @staticmethod
    def _normalize_notification(data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化通知卡片数据"""
        return data  # 直接返回，无需额外处理
    
    @staticmethod
    def _normalize_timeline(data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化时间轴卡片数据"""
        result = data.copy()
        
        # 将items转为timeline_items（避免与dict.items()冲突）
        if 'items' in result and 'timeline_items' not in result:
            result['timeline_items'] = result.pop('items')
        
        return result
    
    @staticmethod
    def _normalize_feature(data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化特色功能卡片数据"""
        return data  # 直接返回，无需额外处理
    
    @staticmethod
    def _normalize_radar(data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化雷达图数据"""
        result = data.copy()
        result.setdefault('subtitle', None)
        result.setdefault('series_name', '数据')
        return result


class HTMLToolkit(BaseRAGToolkit):
    """HTML Toolkit for generating HTML dashboards.
    
    Methods:
        - html_painter(html_config: str, output_filename: Optional[str] = None) -> str
    """

    def __init__(self, config: ToolkitConfig = None):
        """Initialize HTML Toolkit.
        
        Args:
            config: Toolkit configuration
        """
        super().__init__(config)
        
        # 从配置中获取模板路径
        self.template_path = self.config.config.get('template_path', None)
        self.template = self.config.config.get('template', None)
        
        self.renderer = DashboardRenderer(self.template_path, self.template)
        logger.info("Successfully loaded DashboardRenderer")
        
        # 加载 HTML Designer 提示词
        self.html_designer_sp = TOOL_PROMPTS.get(self.config.config.get("html_designer_sp"), "html_designer_sp")
        self.html_designer_up = TOOL_PROMPTS.get(self.config.config.get("html_designer_up"), "html_designer_up")
        logger.info("Successfully loaded HTML Designer prompts")

        self.llm = SimplifiedAsyncOpenAI(
            type=os.environ.get("UTU_LLM_TYPE"),
            api_key=os.environ.get("UTU_LLM_API_KEY"),
            base_url=os.environ.get("UTU_LLM_BASE_URL"),
            model=os.environ.get("UTU_LLM_MODEL"),
        )

        self.html_config = None
        self.report_path = None
        self.work_dir = None

    @register_tool
    async def load_markdown(self, report_path: str) -> str:
        """Load markdown file from report path.
        
        Args:
            report_path (str): Path to markdown file.
            
        Returns:
            str: Markdown content.
        """
        markdown = open(report_path, "r").read()
        return markdown

    @register_tool
    async def html_designer(self, layout: str, report_path: str) -> str:
        """Design HTML dashboard from prompt.
        
        This tool takes a prompt describing the desired dashboard layout and generates
        a dashboard configuration JSON that can be used with the html_painter tool.
        
        Args:
            prompt (str): User prompt describing the desired dashboard layout.
            
        Returns:
            str: JSON string containing the dashboard configuration.
        """
        layout = LLMOutputParser.extract_code_json(layout, "json")
        layout = layout['layout_design']
        markdown = open(report_path, "r").read()

        system_prompt = self.html_designer_sp
        input_str = self.html_designer_up.format(markdown=markdown, layout=layout)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_str}
        ]
        response = await self.llm.query_one(messages=messages)
        res = LLMOutputParser.extract_code_json(response, "json")
        self.html_config = res
        self.report_path = report_path
        self.work_dir = Path(self.report_path).parent

        return json.dumps(res, indent=4, ensure_ascii=False)
    

    @register_tool
    async def html_painter(
        self,
        output_filename: Optional[str] = None
    ) -> str:
        """Generate HTML dashboard from configuration JSON.
        
        This tool converts a dashboard configuration JSON into a complete HTML file
        using the DashboardRenderer. The HTML file will be saved to the current
        working directory.
        
        Args:
            html_config (str): JSON string containing the dashboard configuration.
                Expected format:
                {
                    "title": "Dashboard Title",
                    "subtitle": "Subtitle",
                    "update_time": "YYYY-MM-DD",
                    "rows": [
                        {
                            "modules": [
                                {"type": "kpi", "span": 3, "data": {...}},
                                {"type": "pie", "span": 9, "data": {...}}
                            ]
                        }
                    ]
                }
            output_filename (str, optional): Custom filename for the output HTML file.
                If not provided, generates a timestamped filename like "dashboard_20250213_143022.html".
                
        Returns:
            str: Absolute path to the generated HTML file.
            
        Raises:
            ValueError: If DashboardRenderer is not available or config parsing fails.
            RuntimeError: If HTML generation or file writing fails.
        """
        logger.info(f"[tool] html_painter: generating HTML dashboard")
        
        if self.renderer is None:
            raise ValueError("DashboardRenderer is not available. Please check the import path.")
        
        try:
            # Parse JSON config
            config_dict = self.html_config
            logger.info(f"Parsed config with {len(config_dict.get('rows', []))} rows")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse html_config JSON: {e}")
            raise ValueError(f"Invalid JSON format in html_config: {e}")
        
        try:
            # Render HTML using DashboardRenderer
            html_content = self.renderer.render(config_dict)
            logger.info(f"Successfully rendered HTML (length: {len(html_content)} chars)")
            
        except Exception as e:
            logger.error(f"Failed to render HTML: {e}")
            raise RuntimeError(f"HTML rendering failed: {e}")
        
        try:
            # Determine output filename
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"dashboard_{timestamp}.html"
            
            # Ensure filename has .html extension
            if not output_filename.endswith('.html'):
                output_filename += '.html'
            
            # Get current working directory and construct absolute path
            cwd = Path(self.work_dir)
            output_path = os.path.join(cwd, output_filename)
            
            # Write HTML to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Successfully saved HTML to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to write HTML file: {e}")
            raise RuntimeError(f"Failed to save HTML file: {e}")
