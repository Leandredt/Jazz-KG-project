.PHONY: all crawl ner kg expand bonus test validate clean

all: crawl ner kg expand

crawl:
	python3 src/crawl/crawler.py

ner:
	python3 src/ie/ner_pipeline.py

kg:
	python3 src/kg/ontology.py && python3 src/kg/build_kg.py && python3 src/kg/alignment.py

expand:
	python3 src/kg/expand_kb.py

bonus:
	python3 src/kge/kge_model.py && python3 src/reason/reasoner.py

test:
	python3 -m pytest tests/ -v

validate:
	python3 src/orchestrator/pipeline_validator.py

clean:
	rm -f data/crawler_output.jsonl data/extracted_knowledge.csv kg_artifacts/*.ttl kg_artifacts/*.nt
