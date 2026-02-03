import asyncio
import importlib.util
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from .path import DIR_ROOT


def get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def schema_to_basemodel(schema: dict, class_name: str = None) -> type[BaseModel]:
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }

    def get_python_type(prop_schema):
        prop_type = prop_schema.get("type")

        if prop_type == "array":
            item_type = prop_schema.get("items", {}).get("type", "string")
            return list[type_map.get(item_type, str)]

        return type_map.get(prop_type, str)

    annotations = {}
    fields = {}
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    for field_name, field_schema in properties.items():
        annotations[field_name] = get_python_type(field_schema)
        field_kwargs = {}
        if "description" in field_schema:
            field_kwargs["description"] = field_schema["description"]
        if field_name not in required_fields:
            field_kwargs["default"] = None
            annotations[field_name] = annotations[field_name] | None
        if field_kwargs:
            fields[field_name] = Field(**field_kwargs)
    attrs = {
        "__annotations__": annotations,
        "__module__": __name__,
    }
    attrs.update(fields)

    class_name = class_name or schema.get("title", "GeneratedModel")
    ModelClass = type(class_name, (BaseModel,), attrs)
    return ModelClass


def load_class_from_file(filepath: str, class_name: str) -> type:
    """Load class from file."""
    if not filepath.startswith("/"):
        filepath = str(DIR_ROOT / filepath)

    filepath = Path(filepath).absolute()
    module_name = filepath.stem
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None:
        raise ImportError(f"Could not load spec from file '{filepath}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    if hasattr(module, class_name):
        return getattr(module, class_name)
    else:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_name}'")
