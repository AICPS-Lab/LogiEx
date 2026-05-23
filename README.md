# LogiEx

**Integrating Formal Logic and LLMs for Explainable Transit Planning**

Published at ACM/IEEE ICCPS 2026. [[Paper]](LogiEx.pdf)

LogiEx generates verifiable, natural-language explanations for ADA paratransit vehicle assignment decisions. It combines Monte Carlo Tree Search (MCTS) with formal logic checking and LLM-based narrative generation to produce explanations that are both factually grounded and human-readable.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # then fill in your OpenAI API key
```

## Project Structure

```
backend/
  backend_main.py            # main entry point
  tasks/
    transit.py               # explanation generation pipeline
    transit_logics.py        # formal logic grammar and parsing
    logic_parameterizer.py   # quantitative and qualitative logic evaluation
    embedding_utils.py       # RAG embeddings for the knowledge base
  prompts/
    transit_prompt.py        # prompt templates for 31 query types
  queries/
    transit_queries_by_type.py
  src/transit/
    openai_calls.py          # OpenAI Assistant API integration
  data/
    transit_knowledge.md     # domain knowledge base covering ADA regulations, MCTS, and FAQ
  algo/MCTS/
    mcts.py                  # Monte Carlo Tree Search algorithm
    routeplanner.py          # route planning with timing constraints
    ctl_checking.py          # Computation Tree Logic verification
    explanation_gen.py       # generates explanations from search trees
    objects.py               # vehicle and route data structures
    requests.py              # passenger request modeling

quantitative_eval/
  eval_query_classify.py     # query type classification accuracy
  eval_factual_consistency.py # BERTScore and FactCC evaluation
  eval_llama_benchmark.py    # Llama baseline comparison
  eval_runtime.py            # runtime performance
  eval_data/                 # 31 sample queries by type
  training_bert/             # ground truth query paraphrases
```

## Usage

```bash
python backend/backend_main.py \
  --backend gpt-4 \
  --logic_enabled \
  --temperature 0.7
```

## Citation

```bibtex
@inproceedings{an2026logiex,
  title={LogiEx: Integrating Formal Logic and LLMs for Explainable Transit Planning},
  author={An, Ziyan and Wang, Xia and Baier, Hendrik and Chen, Zirong and Dubey, Abhishek and Johnson, Taylor T and Sprinkle, Jonathan and Ma, Meiyi},
  booktitle={Proceedings of the 17th ACM/IEEE International Conference on Cyber-Physical Systems (ICCPS 2026)},
  year={2026}
}
```

## License

MIT License
