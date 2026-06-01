import nbformat, re, sys

nb = nbformat.read(sys.argv[1], as_version=4)
ci = 0
found = False
for c in nb.cells:
    if c.cell_type != 'code':
        continue
    for o in c.get('outputs', []):
        if o.output_type == 'error':
            found = True
            tb = re.sub(r'\x1b\[[0-9;]*m', '', '\n'.join(o.traceback))
            print('=== code cell', ci, '===')
            print('SRC:', c.source.splitlines()[0])
            print('ERR:', o.ename, '-', o.evalue)
            lines = [l for l in tb.split('\n') if l.strip()]
            print('\n'.join(lines[-8:]))
            print()
    ci += 1
if not found:
    print('NO ERRORS')
