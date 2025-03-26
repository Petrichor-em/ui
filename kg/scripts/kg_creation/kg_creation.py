import os
import sys
import re
import html
import json
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import mammoth
from markdownify import markdownify as md
import networkx as nx
from pyvis.network import Network

sys.path.append("../../")#导入在上级目录中定义的模块（需要修改）

from src.llm import get_llm
from src.prompt import PROMPTS
from src.pre_process.text_split import get_text_splitter
from src.utils import logger
from src.utils.extract_utils import(
     split_string_by_multi_markers,
	 _handle_single_line_extraction, 
)
from src.utils.graphdata_utils import(
	load_graph_data,
	parse_kg_from_json,
	print_nodes_and_edgs
)
from src.utils.file_utils import collect_md_filenames
from src.load.FileProcesser import FileProcesser

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_path = os.path.abspath(os.path.join(os.getcwd(), "../../"))
output_dir = "output/kg_creation"
output_files_dir = os.path.join(project_path, output_dir)
output_json_path = os.path.join(output_files_dir, f"kg_extraction_results_{timestamp}.json")
input_dir = "input/papers"
input_files_path = os.path.join(project_path, input_dir)
llm_config_path = 'input/config/base.yaml'

if not os.path.exists(output_files_dir):
    os.mkdir(output_files_dir)


"""client是带有缓存功能的大模型实例"""
client = get_llm(os.path.join(project_path, llm_config_path))


#为大模型抽取实体提供示例,向prompt中插入example前的处理
kg_extraction_example = PROMPTS["kg_extraction_example"]#一个列表
examples = "\n".join(kg_extraction_example)#将一个列表中的字符串连接成一个单一的字符串，并用换行符 \n 分隔#是一个字符串
language = "Chinese"
example_context_base = dict(#是一个字典
    tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
    record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
    completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    language=language,
)
examples = examples.format(**example_context_base)#按照字典example_context_base解包出的内容填入examples的占位符中
context_base = dict(
    tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
    record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
    completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    examples=examples,
    language=language,
)

#需要抽实体的文章加入
"""
md_file_path_list = collect_md_filenames(input_files_path)
md_file_path = os.path.join(input_files_path,f"{md_file_path_list[0]}")
with open(md_file_path, 'r', encoding='utf-8') as file:
    content = file.read()"""
file_name_list = collect_md_filenames(input_files_path)
file_path = os.path.join(input_files_path,f"{file_name_list[0]}")
file_processer = FileProcesser()
documents = file_processer.load(file_path,comvert_to_pdf=False)
content = documents[0].page_content

#数据预处理，分块
text_splitter = get_text_splitter(chunk_size=2000,chunk_overlap=200)
splitted_chunks = text_splitter.split_text(content)

#实体抽取
all_extraction_result = []#实体抽取的结果
kg_extraction_prompt = PROMPTS["kg_extraction"]
for chunk in tqdm(splitted_chunks):#tqdm显示进度条
	real_prompt_attribute_extraction = kg_extraction_prompt.format(**context_base, input_text=chunk)
	messages = [
			(
					"system",
					"You are a helpful assistant that do entity and attribute extraction.",
			),
			("human", real_prompt_attribute_extraction),
	]
	llm_result = client.invoke(messages)#这是一个块的抽取结果,类型是字符串
	logger.info("chunk: ")
	logger.info(chunk)
	logger.info("llm_result: ")
	logger.info(llm_result)
	all_extraction_result.append(llm_result)
#all_extraction_result是一个字符串数组，数组的长度是块的数目

#第一层for代表处理一个块
#第二层for代表处理一个块中的一个实体
chunk_results = []
for i, extraction_result_str in enumerate(all_extraction_result):
	#records是实体（字符串）列表，split_函数读入实体字符串，根据记录分隔符号将原来的字符串分割为实体字符串数组
	records = split_string_by_multi_markers(extraction_result_str, [context_base["record_delimiter"], context_base["completion_delimiter"]])#返回字符串数组
	
	maybe_nodes = defaultdict(list)#当访问不存在的键时，会自动创建一个空列表作为默认值,字典中值的类型是列表
	maybe_attributes = defaultdict(list)
	maybe_relationships = defaultdict(list)
	for record in records:#一个record就是一个实体记录,每个record是一个字符串，实体信息被包裹在字符案串中的扩号中
		record = re.search(r"\((.*)\)", record)#当 re.search 成功匹配字符串时，会返回一个 re.Match 类型的对象。该对象包含了有关匹配的各种信息，例如匹配的起始位置、结束位置、匹配的内容等。
		if record is None:#实体信息没有被包裹在括号内，格式错误
			continue
		record = record.group(1)#获取正则表达式中第一个捕获组(re.match对象）所匹配到的字符串
		record_attributes = split_string_by_multi_markers(
			record, [context_base["tuple_delimiter"]]
		)#用元组分割符分割实体信息，字符串数组
        
		record_attributes = _handle_single_line_extraction(record_attributes)#存储实体信息的一个字典，作为图中的一个节点

		if record_attributes is None:
			continue
		if 'entity_type' in record_attributes:#检查某个键是否存在
			maybe_nodes[record_attributes["entity_name"]].append(record_attributes)
		elif 'attribute_name' in record_attributes:
			maybe_attributes[record_attributes["entity_name"]].append(record_attributes)
		elif 'source_entity' in record_attributes:
			maybe_relationships[record_attributes["source_entity"]].append(record_attributes)
		
	chunk_result = {
		"chunk_text": splitted_chunks[i],#块的原始全部内容
		"extraction_result": extraction_result_str,#该块实体提取的原始结果（llm返回的字符串）
		"nodes": {k: v for k, v in maybe_nodes.items()},#字典的 items() 方法会返回一个视图对象，该对象包含了字典中所有键值对组成的元组
		"attributes": {k: v for k, v in maybe_attributes.items()},
		"relationships": {k: v for k, v in maybe_relationships.items()}
	}
	chunk_results.append(chunk_result)

#存储结果
"""
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_path = os.path.abspath(os.path.join(os.getcwd(), "../../"))
output_dir = "output/kg_creation"
output_files_dir = os.path.join(project_path, output_dir)
output_json_path = os.path.join(output_files_dir, f"kg_extraction_results_{timestamp}.json")
"""
with open(output_json_path, "w", encoding="utf-8") as f:
	json.dump({
		"total_chunks": len(chunk_results),
		"results": chunk_results
	},f , ensure_ascii=False, indent=2)

logger.info(f"Results saved to {output_json_path}")

#创建图
kg_json_path = output_json_path
G = nx.Graph()

kg_data = load_graph_data(output_json_path)
G = parse_kg_from_json(kg_data, G)
#print_nodes_and_edgs(G)

kg_output_path = os.path.join(output_files_dir, 'graph.graphml')
try:
    nx.write_graphml(G, kg_output_path)
    print("图已成功保存为 GraphML 格式文件。")
except Exception as e:
    print(f"保存图为 GraphML 格式文件时出错: {e}")
