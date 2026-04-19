path = r'pipeline\ml\infer.py'
txt = open(path, encoding='utf-8').read()
old = 'm3_risk_scorer_'
new = 'm3_ridge_'
txt = txt.replace(old, new)
# Also fix bundle key
txt = txt.replace('bundle["ridge"]', 'bundle["model"]')
open(path, 'w', encoding='utf-8').write(txt)
print('Patched:', path)
print('Occurrences:', txt.count('m3_ridge_'))
