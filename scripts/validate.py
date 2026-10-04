"""Validate the unfilled course release, without solving any exercises."""
import argparse
import ast
import json
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
GAPS = {
    1: {'tokenize', 'learn_bpe', 'Bigram.probability', 'Bigram.perplexity'},
    2: {'loss_gradient', 'train_logistic', 'cosine', 'paired_bootstrap'},
    3: {'attention', 'causal_loss', 'masked_loss', 'LoRALinear.forward'},
    4: {'reciprocal_rank', 'run_agent'},
    5: {'context_attention', 'beam_search', 'sequence_loss'},
    6: {'frame_signal', 'log_mel', 'ctc_collapse', 'word_errors'},
    7: {'residual_quantize', 'nominal_bitrate', 'paired_sentence_bootstrap'},
}


def exercise_functions(tree):
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            found[node.name] = node
        elif isinstance(node, ast.ClassDef):
            found.update({f'{node.name}.{method.name}': method
                          for method in node.body if isinstance(method, ast.FunctionDef)})
    return found


def is_gap(node):
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body.pop(0)
    return (len(body) == 1 and isinstance(body[0], ast.Raise)
            and isinstance(body[0].exc, ast.Call)
            and isinstance(body[0].exc.func, ast.Name)
            and body[0].exc.func.id == 'NotImplementedError')


def validate(week, smoke=False):
    stem = f'week{week:02}_starter'
    path = ROOT / 'labs' / f'{stem}.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    script = (ROOT / 'labs' / f'{stem}.py').read_text()
    tree = ast.parse(script)
    functions = exercise_functions(tree)
    assert {name for name, node in functions.items() if is_gap(node)} == GAPS[week]
    assert notebook.metadata.colab.name == f'{stem}.ipynb'
    assert not notebook.metadata.colab.get('include_colab_link', False)
    assert not notebook.metadata.colab.get('omit_output', False)
    tags = {}
    starter = []
    for index, cell in enumerate(notebook.cells):
        assert 'attachments' not in cell, path
        assert set(cell.metadata) == {'tags'}, (path, index)
        tag, = cell.metadata.tags
        tags[tag] = cell
        if cell.cell_type == 'code':
            assert cell.execution_count is None and cell.outputs == [], (path, index)
            ast.parse(cell.source)
            if tag.startswith('starter-'):
                starter.extend(ast.parse(cell.source).body)
    assert ast.dump(ast.Module(body=starter, type_ignores=[])) == ast.dump(
        ast.Module(body=[n for n in tree.body if not isinstance(n, ast.If)], type_ignores=[])
    ), f'{stem}: notebook/script mismatch'
    expected_colab = f'https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/{stem}.ipynb'
    assert expected_colab in tags['introduction'].source
    assert expected_colab in (ROOT / 'README.md').read_text()
    assert expected_colab in (ROOT / 'labs' / f'LAB{week:02}.md').read_text()
    assert tags['student-analysis'].source == '1. _Your response:_\n\n2. _Your response:_\n\n3. _Your response:_\n'
    assert tags['student-comparisons'].source == '# TODO: implement and run the remaining required comparisons here.\nadditional_results = {}\n'
    handout = (ROOT / 'labs' / f'LAB{week:02}.md').read_text()
    assert handout.startswith(tags['handout'].source)
    # Keep standalone notebook pins synchronized with the local/CI environment.
    expected_packages = dict(line.split('==') for line in
                             (ROOT / 'requirements.txt').read_text().splitlines()
                             if line and not line.startswith('#'))
    setup_tree = ast.parse(tags['setup'].source)
    package_assignment = next(node for node in setup_tree.body
                              if isinstance(node, ast.Assign)
                              and isinstance(node.targets[0], ast.Name)
                              and node.targets[0].id == 'LAB_PACKAGES')
    assert ast.literal_eval(package_assignment.value) == expected_packages
    torch_pins = [node for node in setup_tree.body if isinstance(node, ast.Assign)
                  and isinstance(node.targets[0], ast.Subscript)]
    if week in (3, 5):
        torch_version = next(line.split('==')[1] for line in
                             (ROOT / 'requirements-torch.txt').read_text().splitlines()
                             if line.startswith('torch=='))
        assert len(torch_pins) == 1
        assert ast.literal_eval(torch_pins[0].value) == torch_version
    else:
        assert not torch_pins
    for word in ['Save a copy in Drive', 'Anyone with the link', 'Viewer', 'course', 'Cloning']:
        assert word in tags['submission'].source + tags['introduction'].source, (week, word)
    # The release should contain neither embedded media nor saved result payloads.
    assert 'data:image/' not in path.read_text()
    if smoke:
        scope = {'__name__': f'lab{week}_smoke'}
        for tag in ['setup', 'environment']:
            exec(compile(tags[tag].source, f'{stem}:{tag}', 'exec'), scope)
        for cell in notebook.cells:
            if cell.cell_type == 'code' and any(tag.startswith('starter-') for tag in cell.metadata.tags):
                exec(compile(cell.source, stem, 'exec'), scope)
        # Only call supplied data/helpers; never replace the exercise functions.
        if week == 1:
            assert list(map(len, scope['corpus']())) == [120, 36, 36]
        elif week == 2:
            texts, labels, train, dev, test = scope['data']()
            assert len(texts) == len(labels) == 320
            assert not (set(train) & set(dev) or set(train) & set(test) or set(dev) & set(test))
        elif week == 3:
            assert tuple(scope['data'](8, 17).shape) == (8, 7)
            scope['TinyTransformer']()
        elif week == 4:
            env = scope['Environment']()
            assert len(env.search('optics lab room', 3)) == 3
            assert isinstance(env.read('D1'), str)
        elif week == 5:
            train, test = scope['data']()
            assert len(train[0]) + len(test[0]) == 16
            scope['Translator']()
        elif week == 6:
            assert scope['mel_filters']().shape == (40, 257)
        try:
            scope['checks']()
        except NotImplementedError as exc:
            assert str(exc) == 'Complete this laboratory task.'
        else:
            raise AssertionError(f'{stem}: unfilled checks unexpectedly passed')
    print(f'Lab {week}: schema, syntax, {len(GAPS[week])} unfilled exercises, empty outputs, script parity' + ('; setup/scaffolding smoke passed' if smoke else ''))


def dependency_api_smoke():
    """Exercise library APIs on separate tiny data, without answering a lab."""
    import warnings
    import numpy as np
    import torch
    from matplotlib.figure import Figure
    from sklearn.decomposition import TruncatedSVD
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import f1_score, confusion_matrix

    raw = TfidfVectorizer().fit_transform(['red square', 'blue circle', 'red circle', 'blue square'])
    dense = TruncatedSVD(n_components=2, random_state=17).fit_transform(raw)
    labels = np.array([0, 1, 0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        model = MLPClassifier(hidden_layer_sizes=(2,), max_iter=2, random_state=17).fit(dense, labels)
    prediction = model.predict(dense)
    assert confusion_matrix(labels, prediction).shape == (2, 2)
    assert np.isfinite(f1_score(labels, prediction, average='macro'))
    assert np.fft.rfft(np.zeros(400), n=512).shape == (257,)
    figure = Figure()
    figure.subplots().plot([0, 1], [0, 1])
    figure.clear()

    torch.set_num_threads(1)
    embedding = torch.nn.Embedding(4, 3)
    recurrent = torch.nn.GRU(3, 3, batch_first=True)
    head = torch.nn.Linear(3, 4)
    optimizer = torch.optim.Adam(list(embedding.parameters()) + list(recurrent.parameters()) + list(head.parameters()))
    values, _ = recurrent(embedding(torch.tensor([[0, 1, 2]])))
    loss = torch.nn.functional.cross_entropy(head(values).reshape(-1, 4), torch.tensor([1, 2, 3]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert values.device.type == 'cpu'
    print('Scientific/ML API smoke passed: TF-IDF, SVD, MLP, metrics, FFT, plotting, CPU GRU/autograd/Adam.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='run setup and supplied scaffolding only')
    args = parser.parse_args()
    assert len(list((ROOT / 'labs').glob('*.ipynb'))) == 7
    for week in range(1, 8):
        validate(week, args.smoke)
    if args.smoke:
        dependency_api_smoke()
    print('PASS: 7 student notebooks; 24 exercise functions remain unfilled.')


if __name__ == '__main__':
    main()
