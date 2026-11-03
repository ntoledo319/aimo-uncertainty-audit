import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path('solution').resolve()))
from uncertainty_profile.config import GenerationConfidenceConfig
from uncertainty_profile.extraction import extract_generation_features, load_model_and_tokenizer, release_model
from uncertainty_profile.artifact import load_artifact
import pandas as pd, numpy as np

t0=time.time()
model, tok = load_model_and_tokenizer('deepseek-ai/DeepSeek-R1-0528-Qwen3-8B', local_files_only=True)
print(f'model loaded in {time.time()-t0:.0f}s, dtype={next(model.parameters()).dtype}')
case = [json.loads(l) for l in open('getting-started/data/val-sample/input/cases.jsonl') if '"qwen3-8b:low"' in l][0]
cfg = GenerationConfidenceConfig(max_new_tokens=64, batch_size=1)
t0=time.time()
rows = extract_generation_features(model=model, tokenizer=tok, problem_texts=[case['problem']], config=cfg)
dt=time.time()-t0
print(f'generated 64 tokens in {dt:.0f}s -> est {dt/64:.2f}s/token; 4096 tokens ~= {dt/64*4096/60:.0f} min')
art = load_artifact('deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')
feats = pd.DataFrame.from_records(rows, columns=list(art.feature_names))
score = float(np.asarray(art.estimator.predict(feats), dtype=float)[0])
print(f'num_tokens={rows[0]["generation_num_tokens"]} score={score:.4f} threshold={art.decision_threshold} -> robust={score < art.decision_threshold}')
release_model()
