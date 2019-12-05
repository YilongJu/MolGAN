import tensorflow as tf

from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.trainer import Trainer
from utils.trainer import log_filename
from utils.utils import *

from models.gan import GraphGANModel
from models import encoder_rgcn, decoder_adj, decoder_dot, decoder_rnn

from optimizers.gan import GraphGANOptimizer
import time

data = SparseMolecularDataset()
# data.load('data/gdb9_9nodes.sparsedataset')
data.load('data/qm9_5k.sparsedataset')
data_name = "qm9_5k"

batch_dim = 32
la = 0.3
dropout = 0
n_critic = 5
# metric = 'validity,sas'
metric = "validity,unique,novelty,logp"
n_samples = 5000 # 5000
z_dim = 32
epochs = 1 # 10
save_every = 1
decoder_units = (128, 256, 512)
discriminator_units = ((64, 32), 128, (128,))
seed = 0

skip_training = False

def Train_MolGAN(data, data_name, batch_dim, la, dropout, n_critic, metric, n_samples, z_dim, epochs, save_every, decoder_units, discriminator_units, seed, skip_training=False, unrolling_steps=1):

    steps = (len(data) // batch_dim)
    np.random.seed(seed)
    directory = f'{data_name}_bd{batch_dim}_la{la}_do{dropout}_nc{n_critic}_me{metric}_ns{n_samples}_zd{z_dim}_epc{epochs}_se{save_every}_gu{repr(decoder_units)}_du{repr(discriminator_units)}_npseed{seed}_ur{unrolling_steps}'

    print(f"\n{'=' * 20}\nCurrent setting: {directory}")

    with open(log_filename, 'a') as f:
        f.write("\n\n" + directory + "\n")

    def train_fetch_dict(i, steps, epoch, epochs, min_epochs, model, optimizer):
        a = [optimizer.train_step_G] if i % n_critic == 0 else [optimizer.train_step_D]
        b = [optimizer.train_step_V] if i % n_critic == 0 and la < 1 else []
        return a + b


    def train_feed_dict(i, steps, epoch, epochs, min_epochs, model, optimizer, batch_dim):
        mols, _, _, a, x, _, _, _, _ = data.next_train_batch(batch_dim)
        embeddings = model.sample_z(batch_dim)

        if la < 1:

            if i % n_critic == 0:
                rewardR = reward(mols)

                n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                                   feed_dict={model.training: False, model.embeddings: embeddings})
                n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
                mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

                rewardF = reward(mols)

                feed_dict = {model.edges_labels: a,
                             model.nodes_labels: x,
                             model.embeddings: embeddings,
                             model.rewardR: rewardR,
                             model.rewardF: rewardF,
                             model.training: True,
                             model.dropout_rate: dropout,
                             optimizer.la: la if epoch > 0 else 1.0}

            else:
                feed_dict = {model.edges_labels: a,
                             model.nodes_labels: x,
                             model.embeddings: embeddings,
                             model.training: True,
                             model.dropout_rate: dropout,
                             optimizer.la: la if epoch > 0 else 1.0}
        else:
            feed_dict = {model.edges_labels: a,
                         model.nodes_labels: x,
                         model.embeddings: embeddings,
                         model.training: True,
                         model.dropout_rate: dropout,
                         optimizer.la: 1.0}

        return feed_dict


    def eval_fetch_dict(i, epochs, min_epochs, model, optimizer):
        return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G,
                'loss RL': optimizer.loss_RL, 'loss V': optimizer.loss_V,
                'la': optimizer.la}


    def eval_feed_dict(i, epochs, min_epochs, model, optimizer, batch_dim):
        mols, _, _, a, x, _, _, _, _ = data.next_validation_batch()
        embeddings = model.sample_z(a.shape[0])

        rewardR = reward(mols)

        n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                           feed_dict={model.training: False, model.embeddings: embeddings})
        n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
        mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

        rewardF = reward(mols)

        feed_dict = {model.edges_labels: a,
                     model.nodes_labels: x,
                     model.embeddings: embeddings,
                     model.rewardR: rewardR,
                     model.rewardF: rewardF,
                     model.training: False}
        return feed_dict


    def test_fetch_dict(model, optimizer):
        return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G,
                'loss RL': optimizer.loss_RL, 'loss V': optimizer.loss_V,
                'la': optimizer.la}


    def test_feed_dict(model, optimizer, batch_dim):
        mols, _, _, a, x, _, _, _, _ = data.next_test_batch()
        embeddings = model.sample_z(a.shape[0], seed=seed)

        rewardR = reward(mols)

        n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                           feed_dict={model.training: False, model.embeddings: embeddings})
        n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
        mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]
        print("MOLS!!!!!!",mols)
        print(f"\nlen(mols): {len(mols)}")
        img_per_row = 22
        mols_short = mols[:img_per_row**2]
        img = mols2grid_image(mols_short, img_per_row)
        img.save(directory + '/mol' + time.strftime("%m%d_%h%m%s") + '.png')
        print("Mols saved.")
        # exit()

        rewardF = reward(mols)

        feed_dict = {model.edges_labels: a,
                     model.nodes_labels: x,
                     model.embeddings: embeddings,
                     model.rewardR: rewardR,
                     model.rewardF: rewardF,
                     model.training: False}
        return feed_dict


    def reward(mols):
        rr = 1.
        for m in ('logp,sas,qed,unique' if metric == 'all' else metric).split(','):

            if m == 'np':
                rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
            elif m == 'logp':
                rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
            elif m == 'sas':
                rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            elif m == 'qed':
                rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
            elif m == 'novelty':
                rr *= MolecularMetrics.novel_scores(mols, data)
            elif m == 'dc':
                rr *= MolecularMetrics.drugcandidate_scores(mols, data)
            elif m == 'unique':
                rr *= MolecularMetrics.unique_scores(mols)
            elif m == 'diversity':
                rr *= MolecularMetrics.diversity_scores(mols, data)
            elif m == 'validity':
                rr *= MolecularMetrics.valid_scores(mols)
            else:
                raise RuntimeError('{} is not defined as a metric'.format(m))

        return rr.reshape(-1, 1)


    def _eval_update(i, epochs, min_epochs, model, optimizer, batch_dim, eval_batch):
        mols = samples(data, model, session, model.sample_z(n_samples), sample=True)
        m0, m1 = all_scores(mols, data, norm=True)
        m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
        m0.update(m1)
        return m0


    def _test_update(model, optimizer, batch_dim, test_batch):
        mols = samples(data, model, session, model.sample_z(n_samples), sample=True)
        m0, m1 = all_scores(mols, data, norm=True)
        m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
        m0.update(m1)
        return m0


    # for mol in data.data:
    #     for bond in mol.GetBonds():
    #         print(f"mol {mol}, bond {bond}")
    print(list(sorted(set(bond.GetBondType() for mol in data.data for bond in mol.GetBonds()))))
    print(f"data.vertexes: {data.vertexes}")
    print(f"data.bond_num_types: {data.bond_num_types}")
    print(f"data.atom_num_types: {data.atom_num_types}")
    # exit()

    # model
    model = GraphGANModel(data.vertexes,
                          data.bond_num_types,
                          data.atom_num_types,
                          z_dim,
                          decoder_units=decoder_units,
                          discriminator_units=discriminator_units, # discriminator_units=((128, 64), 128, (128, 64)),
                          decoder=decoder_adj,
                          discriminator=encoder_rgcn,
                          soft_gumbel_softmax=False,
                          hard_gumbel_softmax=False,
                          batch_discriminator=False,
                          unrolling_steps=unrolling_steps)

    # optimizer
    optimizer = GraphGANOptimizer(model, learning_rate=1e-3, feature_matching=False)

    # session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # trainer
    trainer = Trainer(model, optimizer, session)

    print('Parameters: {}'.format(np.sum([np.prod(e.shape) for e in session.run(tf.trainable_variables())])))

    trainer.train(batch_dim=batch_dim,
                  epochs=epochs,
                  steps=steps,
                  train_fetch_dict=train_fetch_dict,
                  train_feed_dict=train_feed_dict,
                  eval_fetch_dict=eval_fetch_dict,
                  eval_feed_dict=eval_feed_dict,
                  test_fetch_dict=test_fetch_dict,
                  test_feed_dict=test_feed_dict,
                  save_every=save_every,
                  directory=directory, # here users need to first create and then specify a folder where to save the model
                  _eval_update=_eval_update,
                  _test_update=_test_update,
                  skip_training=skip_training)

if __name__ == "__main__":
    # Train_MolGAN(data, data_name, batch_dim, la, dropout, n_critic, metric, n_samples, z_dim, epochs, save_every, decoder_units, discriminator_units, seed, skip_training=skip_training)


    data = SparseMolecularDataset()
    # data.load('data/gdb9_9nodes.sparsedataset')
    data.load('data/qm9_5k.sparsedataset')
    data_name = "qm9_5k"

    batch_dim = 32
    la = 0.9
    dropout = 0
    n_critic = 5
    metric = 'qed'
    # metric = "validity,unique,novelty,logp"
    n_samples = 5000 # 5000
    z_dim = 32
    epochs = 30 # 10
    save_every = 1
    decoder_units = (128, 256, 512)
    discriminator_units = ((128, 64), 128, (128, 64))
    seed = 0
    unrolling_steps = 5

    skip_training = False

    # z_dim_list = [32, 128, 64]
    # dropout_list = [0, 0.25, 0.1]
    # la_list = [0.25, 0.5, 0.75, 1]
    # discriminator_units_list = [((64, 32), 128, (128,)), \
    # ((64, 32), 128, (128, 64)), \
    # ((128, 64), 128, (128, 64)), \
    # ((256, 128), 256, (256, 128)) \
    # ]

    # for z_dim in z_dim_list:
    #     for dropout in dropout_list:
    #         for la in la_list:
    #             for discriminator_units in discriminator_units_list:
    #                 Train_MolGAN(data, data_name, batch_dim, la, dropout, n_critic, metric, n_samples, z_dim, epochs, save_every, decoder_units, discriminator_units, seed, skip_training=skip_training)

    Train_MolGAN(data, data_name, batch_dim, la, dropout, n_critic, metric, n_samples, z_dim, epochs, save_every, decoder_units, discriminator_units, seed, skip_training=skip_training, unrolling_steps=unrolling_steps)
