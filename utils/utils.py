import numpy as np

from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import confusion_matrix

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

from utils.molecular_metrics import MolecularMetrics

import tensorflow as tf
from collections import OrderedDict


def mols2grid_image(mols, molsPerRow):
    mols = [e if e is not None else Chem.RWMol() for e in mols]

    for mol in mols:
        AllChem.Compute2DCoords(mol)

    return Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(150, 150))


def classification_report(data, model, session, sample=False):
    _, _, _, a, x, _, f, _, _ = data.next_validation_batch()

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={model.edges_labels: a, model.nodes_labels: x,
                                                            model.node_features: f, model.training: False,
                                                            model.variational: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    y_true = e.flatten()
    y_pred = a.flatten()
    target_names = [str(Chem.rdchem.BondType.values[int(e)]) for e in data.bond_decoder_m.values()]

    print('######## Classification Report ########\n')
    print(sk_classification_report(y_true, y_pred, labels=list(range(len(target_names))),
                                   target_names=target_names))

    print('######## Confusion Matrix ########\n')
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))

    y_true = n.flatten()
    y_pred = x.flatten()
    target_names = [Chem.Atom(e).GetSymbol() for e in data.atom_decoder_m.values()]

    print('######## Classification Report ########\n')
    print(sk_classification_report(y_true, y_pred, labels=list(range(len(target_names))),
                                   target_names=target_names))

    print('\n######## Confusion Matrix ########\n')
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))


def reconstructions(data, model, session, batch_dim=10, sample=False):
    m0, _, _, a, x, _, f, _, _ = data.next_train_batch(batch_dim)

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={model.edges_labels: a, model.nodes_labels: x,
                                                            model.node_features: f, model.training: False,
                                                            model.variational: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    m1 = np.array([e if e is not None else Chem.RWMol() for e in [data.matrices2mol(n_, e_, strict=True)
                                                                  for n_, e_ in zip(n, e)]])

    mols = np.vstack((m0, m1)).T.flatten()

    return mols


def samples(data, model, session, embeddings, sample=False):
    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={
        model.embeddings: embeddings, model.training: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    return mols


def all_scores(mols, data, norm=False, reconstruction=False):
    m0 = {k: list(filter(lambda e: e is not None, v)) for k, v in {
        'NP score': MolecularMetrics.natural_product_scores(mols, norm=norm),
        'QED score': MolecularMetrics.quantitative_estimation_druglikeness_scores(mols),
        'logP score': MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=norm),
        'SA score': MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=norm),
        'diversity score': MolecularMetrics.diversity_scores(mols, data),
        'drugcandidate score': MolecularMetrics.drugcandidate_scores(mols, data)}.items()}

    m1 = {'valid score': MolecularMetrics.valid_total_score(mols) * 100,
          'unique score': MolecularMetrics.unique_total_score(mols) * 100,
          'novel score': MolecularMetrics.novel_total_score(mols, data) * 100}

    return m0, m1


_graph_replace = tf.contrib.graph_editor.graph_replace
def remove_original_op_attributes(graph):
    """Remove _original_op attribute from all operations in a graph."""
    for op in graph.get_operations():
        op._original_op = None

def graph_replace(*args, **kwargs):
    """Monkey patch graph_replace so that it works with TF 1.0"""
    remove_original_op_attributes(tf.get_default_graph())
    return _graph_replace(*args, **kwargs)


def extract_update_dict(update_ops):
    """Extract variables and their new values from Assign and AssignAdd ops.

    Args:
        update_ops: list of Assign and AssignAdd ops, typically computed using Keras' opt.get_updates()

    Returns:
        dict mapping from variable values to their updated value
    """
    name_to_var = {v.name: v for v in tf.global_variables()}
    updates = OrderedDict()

    print(f"update_ops: {update_ops}")

    for update in update_ops:
    #     # print(f"\nupdate: {update}\n")
    #     try:
    #         print(f"\nupdate.name {update.name}\n")
    #     except:
    #         pass
    #
    #     try:
    #         print(f"\nname_to_var {name_to_var}\n")
    #     except:
    #         pass
    #
    #     try:
    #         print(f"\nname_to_var[update.name] {name_to_var[update.name]}\n")
    #     except:
    #         pass
    #
    #     try:
    #         print(f"\nupdate.op {update.op}\n")
    #     except:
    #         pass
    #
    #     try:
    #         print(f"\nupdate.op_def {update.op_def}\n")
    #     except:
    #         pass
    #
    #     try:
    #         print(f"\nupdate.type {update.type}\n")
    #     except:
    #         pass
    #
    #     try:
    #         print(f"\nupdate.outputs {update.outputs}\n")
    #     except:
    #         pass
    #
    #     try:
    #         print(f"\nupdate.ops {update.ops}\n")
    #     except:
    #         pass
    #
    #     try:
    #         print(f"\nupdate.__dict__ {update.__dict__}\n")
    #     except:
    #         pass
    #
    #     try:
    #         print(f"\nupdate.grad {update[0]}\n")
    #     except:
    #         pass
    #
    #     try:
    #         print(f"\nupdate.vars {update[1]}\n")
    #     except:
    #         pass
    #
    #     try:
    #         print(f"\nupdate.inputs {update.inputs}\n")
    #     except:
    #         pass
    #
    #     try:
    #         print(f"\nupdate.inputs() {update.inputs()}\n")
    #     except:
    #         pass
    #
    #     try:
    #         print(f"\nupdate.inputs._inputs {update.inputs._inputs}\n")
    #     except:
    #         pass
    #
    #     try:
    #         print(f"\nupdate.inputs {update.inputs[0]}\n")
    #     except:
    #         pass
    #
    #     try:
    #         print(f"\nupdate.inputs {update.inputs[1]}\n")
    #     except:
    #         pass

        # var_name = update.op.inputs[0].name
        # var = name_to_var[var_name]
        # value = update.op.inputs[1]
        # print(f"update.op.type: {update.op.type}")
        # if update.op.type in ['AssignVariableOp', 'Assign']:
            # updates[var.value()] = value

        var_name = update.inputs[0].name
        var = name_to_var[var_name]
        value = update.inputs[1]
        print(f"update.type: {update.type}")
        if update.type in ['AssignVariableOp', 'Assign']:
            updates[var.value()] = value
        elif update.type in ['AssignAddVariableOp', 'AssignAdd']:
            updates[var.value()] = var + value
        else:
            raise ValueError("Update op type (%s) must be of type Assign or AssignAdd"%update.type)
    return updates
