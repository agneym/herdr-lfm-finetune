import json, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from split import train_val, eval_holdout

rows = [json.loads(l) for l in open('dataset.jsonl')]
old_rows = [json.loads(l) for l in open('/tmp/dataset_new.jsonl')]  # pre-fix copy (gemini row differs only)
schemas = {s['name']: s for s in json.load(open('reference/herdr_schemas.json'))}
errs = []
cnt = collections.Counter()
seen = set()
for i, r in enumerate(rows):
    q = r['messages'][1]['content']
    if q.lower() in seen:
        errs.append(f"row {i}: dup query")
    seen.add(q.lower())
    for c in r['expected']:
        cnt[c['name']] += 1
        s = schemas[c['name']]
        props = s['parameters']['properties']
        req = set(s['parameters'].get('required', []))
        args = c.get('arguments') or {}
        for k, v in args.items():
            if k not in props or ('enum' in props[k] and v not in props[k]['enum']):
                errs.append(f"row {i}: bad {c['name']}.{k}={v!r}")
        for k in req:
            if k not in args:
                errs.append(f"row {i}: missing {c['name']}.{k}")

print("rows:", len(rows), "off-topic:", sum(1 for r in rows if not r['expected']))
print("schema errors:", errs if errs else "none")
print("\nlabel counts:")
for t, c in sorted(cnt.items(), key=lambda x: -x[1]):
    print(f"  {t:24}{c}")

tr, v, ev = train_val(len(rows))
assert not set(tr) & set(v) and not set(tr) & set(ev) and not set(v) & set(ev)
print("\nsplit: train", len(tr), " val", len(v), " eval", len(ev), "- disjoint OK")

# old 270-row dataset is a prefix of the new one -> old holdout indices keep same rows
prefix = all(old_rows[i]['messages'][1]['content'] == rows[i]['messages'][1]['content']
             and json.dumps(old_rows[i]['expected'], sort_keys=True)
             == json.dumps(rows[i]['expected'], sort_keys=True)
             for i in range(len(old_rows)) if i != 282)
print("append-only vs previous dataset (except fixed gemini row):", prefix)
