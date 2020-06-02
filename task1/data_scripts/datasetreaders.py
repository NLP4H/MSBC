import os
import pandas as pd
from typing import Dict, List, Iterator, Tuple, Union
import logging
import torch

from overrides import overrides

from transformers import BertTokenizerFast

# AllenNLP imports
from allennlp.data import Instance
from allennlp.data.fields import LabelField, TextField, MetadataField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, PretrainedTransformerIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token


@DatasetReader.register('ms_tokenized_reader')
class ms_tokenized_reader(DatasetReader):
	def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}

	def text_to_instance(self, tokens: List[Token], ids: List[int], labels: List[float] = None) -> Instance:
		note_field = TextField(tokens, self.token_indexers)
		fields = {"note": note_field}

		id_field = MetadataField(ids)
		fields["id"] = id_field

		if labels:
			label_field = LabelField(labels)
			fields["edss_labels"] = label_field

		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			yield self.text_to_instance(tokens=[Token(str(x)) for x in row["tokenized_text"].split("\t")], ids=row["patient_id"], labels = str(row["edss_19"]))


@DatasetReader.register('ms_edss19_reader')
class ms_edss19_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("master_path/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: float = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		id_field = MetadataField([ids])
		fields["ids"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="edss19_labels")
			fields["label"] = label_field
		else:	
			label_field = LabelField(str(0.0), label_namespace="edss19_labels")
			fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			if row["tokenized_text"] == "[101, 102]" or row["edss_19"] == '' or row["edss_19"] is None:
				continue
			if row["edss_19"] < 0 :
				continue
			label = row["edss_19"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)


@DatasetReader.register('ms_predict_edss_19_allen_cnn_reader')
class ms_predict_edss_19_allen_cnn_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("master_path/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: float = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		id_field = MetadataField([ids])
		fields["ids"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="predict_edss_19_allen_cnn_labels")
			fields["label"] = label_field
		else:	
			label_field = LabelField(str(0.0), label_namespace="predict_edss_19_allen_cnn_labels")
			fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			if row["tokenized_text"] == "[101, 102]" or row["predicted_edss_19_allen_cnn"] == '' or row["predicted_edss_19_allen_cnn"] is None:
				continue
			if row["predicted_edss_19_allen_cnn"] < 0 :
				continue
			label = row["predicted_edss_19_allen_cnn"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)


@DatasetReader.register('ms_edss_19_snorkel_rb_allen_cnn_lm_reader')
class ms_edss_19_snorkel_rb_allen_cnn_lm_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("master_path/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: float = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		id_field = MetadataField([ids])
		fields["ids"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="predict_edss_19_snorkel_rb_allen_cnn_lm_labels")
			fields["label"] = label_field
		else:	
			label_field = LabelField(str(0.0), label_namespace="predict_edss_19_snorkel_rb_allen_cnn_lm_labels")
			fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			if row["tokenized_text"] == "[101, 102]" or row["predict_edss_19_snorkel_rb_allen_cnn_lm"] == '' or row["predict_edss_19_snorkel_rb_allen_cnn_lm"] is None:
				continue
			if row["predict_edss_19_snorkel_rb_allen_cnn_lm"] < 0 :
				continue
			label = row["predict_edss_19_snorkel_rb_allen_cnn_lm"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)


@DatasetReader.register('ms_type_reader')
class ms_type_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("master_path/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: float = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		id_field = MetadataField([ids])
		fields["ids"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="ms_type_labels")
			fields["label"] = label_field
		#else:	
		#	label_field = LabelField(str(0.0), label_namespace="ms_type_labels")
		#	fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			if row["tokenized_text"] == "[101, 102]" or row["ms_type"] == '' or row["ms_type"] is None:
				continue
			if row["ms_type"] not in ["PP", "RR", "CIS", "SP"]:
				continue
			label = row["ms_type"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)


@DatasetReader.register('ms_relapse_reader')
class ms_relapse_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("master_path/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: int = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		id_field = MetadataField([ids])
		fields["ids"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="relapse_labels")
			fields["label"] = label_field
		else:
			label_field = LabelField(str(0), label_namespace="relapse_labels")
			fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			if row["tokenized_text"] == "[101, 102]" or row["recent_relapse"] == '' or row["recent_relapse"] is None:
				continue
			if row["recent_relapse"] < 0 or row["recent_relapse"] > 1:
				continue
			label = row["recent_relapse"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)

@DatasetReader.register('ms_future_relapse_reader')
class ms_future_relapse_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("master_path/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: int = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		id_field = MetadataField([ids])
		fields["ids"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="future_relapse_labels")
			fields["label"] = label_field
		else:
			label_field = LabelField(str(0), label_namespace="future_relapse_labels")
			fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			if row["tokenized_text"] == "[101, 102]" or row["future_relapse"] == '' or row["future_relapse"] is None:
				continue
			if row["future_relapse"] < 0 or row["future_relapse"] > 1:
				continue
			label = row["future_relapse"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)

@DatasetReader.register('ms_edss10_reader')
class ms_edss10_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("master_path/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: float = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		#id_field = MetadataField([ids])
		#fields["id"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="edss10_labels")
			fields["label"] = label_field
		else:	
			label_field = LabelField(str(0.0), label_namespace="edss10_labels")
			fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			if row["tokenized_text"] == "[101, 102]" or row["edss_10"] == '' or row["edss_10"] is None:
				continue
			label = row["edss_10"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)


@DatasetReader.register('ms_edss3_reader')
class ms_edss3_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("master_path/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: float = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		#id_field = MetadataField([ids])
		#fields["id"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="edss3_labels")
			fields["label"] = label_field
		else:	
			label_field = LabelField(str(0.0), label_namespace="edss3_labels")
			fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			if row["tokenized_text"] == "[101, 102]" or row["edss_3"] == '' or row["edss_3"] is None:
				continue
			label = row["edss_3"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)


@DatasetReader.register('ms_edss4_reader')
class ms_edss4_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("master_path/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: float = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		#id_field = MetadataField([ids])
		#fields["id"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="edss4_labels")
			fields["label"] = label_field
		else:	
			label_field = LabelField(str(0.0), label_namespace="edss4_labels")
			fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			if row["tokenized_text"] == "[101, 102]" or row["edss_4"] == '' or row["edss_4"] is None:
				continue
			label = row["edss_4"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)


@DatasetReader.register('ms_score_ambulation_subscore_reader')
class ms_score_ambulation_subscore_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("master_path/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: int = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		#id_field = MetadataField([ids])
		#fields["id"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="score_ambulation_subscore_labels")
			fields["label"] = label_field
		else:	
			label_field = LabelField(str(0), label_namespace="score_ambulation_subscore_labels")
			fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			if row["tokenized_text"] == "[101, 102]" or row["score_ambulation_subscore"] == '' or row["score_ambulation_subscore"] is None:
				continue
			if row["score_ambulation_subscore"] < 0:
				continue
			label = row["score_ambulation_subscore"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)

@DatasetReader.register('ms_score_bowel_bladder_subscore_reader')
class ms_score_bowel_bladder_subscore_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("master_path/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: int = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		#id_field = MetadataField([ids])
		#fields["id"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="score_bowel_bladder_subscore_labels")
			fields["label"] = label_field
		else:	
			label_field = LabelField(str(0), label_namespace="score_bowel_bladder_subscore_labels")
			fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			if row["tokenized_text"] == "[101, 102]" or row["score_bowel_bladder_subscore"] == '' or row["score_bowel_bladder_subscore"] is None:
				continue
			if row["score_bowel_bladder_subscore"] < 0:
				continue
			label = row["score_bowel_bladder_subscore"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)


@DatasetReader.register('ms_score_brain_stem_subscore_reader')
class ms_score_brain_stem_subscore_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("master_path/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: int = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		#id_field = MetadataField([ids])
		#fields["id"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="score_brain_stem_subscore_labels")
			fields["label"] = label_field
		else:	
			label_field = LabelField(str(0), label_namespace="score_brain_stem_subscore_labels")
			fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			if row["tokenized_text"] == "[101, 102]" or row["score_brain_stem_subscore"] == '' or row["score_brain_stem_subscore"] is None:
				continue
			if row["score_brain_stem_subscore"] < 0:
				continue
			label = row["score_brain_stem_subscore"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)


@DatasetReader.register('ms_score_cerebellar_subscore_reader')
class ms_score_cerebellar_subscore_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("master_path/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: int = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		#id_field = MetadataField([ids])
		#fields["id"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="score_cerebellar_subscore_labels")
			fields["label"] = label_field
		else:	
			label_field = LabelField(str(0), label_namespace="score_cerebellar_subscore_labels")
			fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			if row["tokenized_text"] == "[101, 102]" or row["score_cerebellar_subscore"] == '' or row["score_cerebellar_subscore"] is None:
				continue
			if row["score_cerebellar_subscore"] < 0:
				continue
			label = row["score_cerebellar_subscore"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)


@DatasetReader.register('ms_score_mental_subscore_reader')
class ms_score_mental_subscore_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("master_path/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: int = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		#id_field = MetadataField([ids])
		#fields["id"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="score_mental_subscore_labels")
			fields["label"] = label_field
		else:	
			label_field = LabelField(str(0), label_namespace="score_mental_subscore_labels")
			fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			if row["tokenized_text"] == "[101, 102]" or row["score_mental_subscore"] == '' or row["score_mental_subscore"] is None:
				continue
			if row["score_mental_subscore"] < 0:
				continue
			label = row["score_mental_subscore"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)


@DatasetReader.register('ms_score_pyramidal_subscore_reader')
class ms_score_pyramidal_subscore_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("master_path/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: int = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		#id_field = MetadataField([ids])
		#fields["id"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="score_pyramidal_subscore_labels")
			fields["label"] = label_field
		else:	
			label_field = LabelField(str(0), label_namespace="score_pyramidal_subscore_labels")
			fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			if row["tokenized_text"] == "[101, 102]" or row["score_pyramidal_subscore"] == '' or row["score_pyramidal_subscore"] is None:
				continue
			if row["score_pyramidal_subscore"] < 0:
				continue
			label = row["score_pyramidal_subscore"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)


@DatasetReader.register('ms_score_sensory_subscore_reader')
class ms_score_sensory_subscore_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("master_path/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: int = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		#id_field = MetadataField([ids])
		#fields["id"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="score_sensory_subscore_labels")
			fields["label"] = label_field
		else:	
			label_field = LabelField(str(0), label_namespace="score_sensory_subscore_labels")
			fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			if row["tokenized_text"] == "[101, 102]" or row["score_sensory_subscore"] == '' or row["score_sensory_subscore"] is None:
				continue
			if row["score_sensory_subscore"] < 0:
				continue
			label = row["score_sensory_subscore"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)


@DatasetReader.register('ms_score_visual_subscore_reader')
class ms_score_visual_subscore_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("master_path/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: int = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		#id_field = MetadataField([ids])
		#fields["id"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="score_visual_subscore_labels")
			fields["label"] = label_field
		else:	
			label_field = LabelField(str(0), label_namespace="score_visual_subscore_labels")
			fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			# import pdb; pdb.set_trace()
			if row["tokenized_text"] == "[101, 102]" or row["score_visual_subscore"] == '' or row["score_visual_subscore"] is None:
				continue
			if row["score_visual_subscore"] < 0:
				continue
			label = row["score_visual_subscore"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)

