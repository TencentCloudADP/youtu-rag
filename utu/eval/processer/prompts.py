from ...utils import FileUtils

AUGMENTATION_PROMPTS = FileUtils.load_prompts("eval/augmentation_templates.yaml")
JUDGE_PROMPTS = FileUtils.load_prompts("eval/judge_templates.yaml")
