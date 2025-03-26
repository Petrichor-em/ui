import re
import html

# extract results
def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)
def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))

def _handle_single_line_extraction(#处理一行的抽取结果，抽取结果有三种：实体，关系，属性
    record_attributes: list[str],
):
    if len(record_attributes) == 4 and record_attributes[0] == 'entity':
        return _handle_single_entity_extraction(record_attributes)
    elif len(record_attributes) == 7 and record_attributes[0] == 'attribute':
        return _handle_single_attribute_extraction(record_attributes)
    elif len(record_attributes) == 6 and record_attributes[0] == 'relationship':
        return _handle_single_relationship_extraction(record_attributes)
    else:
        return None

def _handle_single_entity_extraction(
    record_attributes: list[str],
):
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():#strip移除字符串开头和结尾处的指定字符，默认移除的是空白字符
        return None#如果实体名字为空，返回None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
    )

def _handle_single_attribute_extraction(
    record_attributes: list[str],
):
    # add this record as edge
    entity_name = clean_str(record_attributes[1].upper())
    attribute_name = clean_str(record_attributes[2].upper())
    attribute_value = clean_str(record_attributes[3])
    attribute_description = clean_str(record_attributes[4])
    attribute_keywords = clean_str(record_attributes[6])
    attribute_strength = (
        float(record_attributes[5]) if is_float_regex(record_attributes[5]) else 1.0
    )
    return dict(
        entity_name=entity_name,
        attribute_name=attribute_name,
        attribute_value=attribute_value,
        attribute_description=attribute_description,
        attribute_strength=attribute_strength,
        attribute_keywords=attribute_keywords,
    )

def _handle_single_relationship_extraction(
    record_attributes: list[str],
):
    # add this record as edge
    source_entity = clean_str(record_attributes[1].upper())
    target_entity = clean_str(record_attributes[2].upper())
    relationship_description = clean_str(record_attributes[3])
    relationship_strength = (
        float(record_attributes[4]) if is_float_regex(record_attributes[4]) else 1.0
    )
    relationship_keywords = clean_str(record_attributes[5].upper())
    return dict(
        source_entity=source_entity,
        target_entity=target_entity,
        relationship_description=relationship_description,
        relationship_strength=relationship_strength,
        relationship_keywords=relationship_keywords,
    )
