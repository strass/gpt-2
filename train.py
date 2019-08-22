#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./train --dataset <file|directory|glob>

import argparse
import json
import os
import numpy as np
import tensorflow as tf
import time
import tqdm
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

import model, sample, encoder
from load_dataset import load_dataset, Sampler
from accumulate import AccumulatingOptimizer
import memory_saving_gradients

CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = 'samples'


parser = argparse.ArgumentParser(
    description='Fine-tune GPT-2 on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, required=True, help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')
parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='Concatenate input files with <|endoftext|> separator into chunks of this minimum size')

parser.add_argument('--batch_size', metavar='SIZE', type=int, default=1, help='Batch size')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=0.00002, help='Learning rate for Adam')
parser.add_argument('--accumulate_gradients', metavar='N', type=int, default=1, help='Accumulate gradients across N minibatches.')
parser.add_argument('--memory_saving_gradients', default=False, action='store_true', help='Use gradient checkpointing to reduce vram usage.')
parser.add_argument('--only_train_transformer_layers', default=False, action='store_true', help='Restrict training to the transformer blocks.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer. <adam|sgd>.')
parser.add_argument('--noise', type=float, default=0.0, help='Add noise to input training data to regularize against typos.')

parser.add_argument('--top_k', type=int, default=40, help='K for top-k sampling.')
parser.add_argument('--top_p', type=float, default=0.0, help='P for top-p sampling. Overrides top_k if set > 0.')

parser.add_argument('--restore_from', type=str, default='latest', help='Either "latest", "fresh", or a path to a checkpoint file')
parser.add_argument('--run_name', type=str, default='run1', help='Run id. Name of subdirectory in checkpoint/ and samples/')
parser.add_argument('--sample_every', metavar='N', type=int, default=100, help='Generate samples every N steps')
parser.add_argument('--sample_length', metavar='TOKENS', type=int, default=-1, help='Sample this many tokens')
parser.add_argument('--sample_num', metavar='N', type=int, default=1, help='Generate this many samples')
parser.add_argument('--save_every', metavar='N', type=int, default=1000, help='Write a checkpoint every N steps')

parser.add_argument('--val_dataset', metavar='PATH', type=str, default=None, help='Dataset for validation loss, defaults to --dataset.')
parser.add_argument('--val_batch_size', metavar='SIZE', type=int, default=1, help='Batch size for validation.')
parser.add_argument('--val_batch_count', metavar='N', type=int, default=80, help='Number of batches for validation.')
parser.add_argument('--val_every', metavar='STEPS', type=int, default=0, help='Calculate validation loss every STEPS steps.')

parser.add_argument('--storage_bucket', metavar='BUCKET', type=str, default='gs://sgappa-multi/gpt-2/', help='Cloud storage bucket name (when using TPU)')
parser.add_argument('--init_tpu', default=False, action='store_true', help='Initialize TPU session.')

parser.add_argument('--fresh_model', default=False, action='store_true', help="Don't load model from disk; initialize model weights to random values")
parser.add_argument('--save_on_ctrlc', default=False, action='store_true', help='When execution is interrupted, should we save the model to disk?')

# 1.5B
#parser.add_argument('--n_ctx', type=int, default=1024, help='For a fresh model, how large should n_ctx be?')
#parser.add_argument('--n_embd', type=int, default=1600, help='For a fresh model, how large should n_embd be?')
#parser.add_argument('--n_head', type=int, default=25, help='For a fresh model, how large should n_head be?')
#parser.add_argument('--n_layer', type=int, default=48, help='For a fresh model, how large should n_layer be?')

# 345M
#parser.add_argument('--n_ctx', type=int, default=1024, help='For a fresh model, how large should n_ctx be?')
#parser.add_argument('--n_embd', type=int, default=1024, help='For a fresh model, how large should n_embd be?')
#parser.add_argument('--n_head', type=int, default=16, help='For a fresh model, how large should n_head be?')
#parser.add_argument('--n_layer', type=int, default=24, help='For a fresh model, how large should n_layer be?')

# 117M
parser.add_argument('--n_ctx', type=int, default=1024, help='For a fresh model, how large should n_ctx be?')
parser.add_argument('--n_embd', type=int, default=768, help='For a fresh model, how large should n_embd be?')
parser.add_argument('--n_head', type=int, default=12, help='For a fresh model, how large should n_head be?')
parser.add_argument('--n_layer', type=int, default=12, help='For a fresh model, how large should n_layer be?')

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


def randomize(context, hparams, p):
    if p > 0:
        mask = tf.random.uniform(shape=tf.shape(context)) < p
        noise = tf.random.uniform(shape=tf.shape(context), minval=0, maxval=hparams.n_vocab, dtype=tf.int32)
        return tf.where(mask, noise, context)
    else:
        return context


def main(tpu_cluster=None):
    args = parser.parse_args()
    BUCKET = args.storage_bucket if tpu_cluster else ''
    enc = encoder.get_encoder(args.model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', args.model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    if args.fresh_model:
        hparams.n_ctx=args.n_ctx
        hparams.n_embd=args.n_embd
        hparams.n_head=args.n_head
        hparams.n_layer=args.n_layer

    if args.sample_length < 0:
        args.sample_length = hparams.n_ctx - 1
    if args.sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    if args.model_name == '345M':
        args.memory_saving_gradients = True
        if args.optimizer == 'adam':
            args.only_train_transformer_layers = True

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
    with tf.Session(tpu_cluster, config=config) as sess:
        if tpu_cluster and args.init_tpu:
            print("initializing TPU system...")
            sess.run(tpu.initialize_system())
        if tpu_cluster:
            print("Using TPU %s" % tpu_cluster)
        context = tf.placeholder(tf.int32, [args.batch_size, None])
        context_in = randomize(context, hparams, args.noise)
        output = model.model(hparams=hparams, X=context_in)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1]))

        if args.val_every > 0:
            val_context = tf.placeholder(tf.int32, [args.val_batch_size, None])
            val_output = model.model(hparams=hparams, X=val_context)
            val_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=val_context[:, 1:], logits=val_output['logits'][:, :-1]))
            val_loss_summary = tf.summary.scalar('val_loss', val_loss)


        tf_sample = sample.sample_sequence(
            hparams=hparams,
            length=args.sample_length,
            context=context,
            batch_size=args.batch_size,
            temperature=1.0,
            top_k=args.top_k,
            top_p=args.top_p)

        all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
        train_vars = [v for v in all_vars if '/h' in v.name] if args.only_train_transformer_layers else all_vars

        parameter_count = sum([np.prod(v.shape.as_list()) for v in train_vars])
        print("This model is using %d parameters (%.2fM)" % (parameter_count, parameter_count/(1024.0*1024.0)))

        if args.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        elif args.optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
        else:
            exit('Bad optimizer:', args.optimizer)
        
        if tpu_cluster:
            # https://pulsejet.github.io/blog/posts/tpu-without-estimator/
            from tensorflow.contrib.tpu.python.tpu import tpu_function
            #tpu_function.get_tpu_context().set_number_of_shards(8)
            #opt = tf.contrib.tpu.CrossShardOptimizer(opt)

        if args.accumulate_gradients > 1:
            if args.memory_saving_gradients:
                exit("Memory saving gradients are not implemented for gradient accumulation yet.")
            opt = AccumulatingOptimizer(
                opt=opt,
                var_list=train_vars)
            opt_reset = opt.reset()
            opt_compute = opt.compute_gradients(loss)
            opt_apply = opt.apply_gradients()
            summary_loss = tf.summary.scalar('loss', opt_apply)
        else:
            if args.memory_saving_gradients:
                opt_grads = memory_saving_gradients.gradients(loss, train_vars)
            else:
                opt_grads = tf.gradients(loss, train_vars)
            opt_grads = list(zip(opt_grads, train_vars))
            opt_apply = opt.apply_gradients(opt_grads)
            summary_loss = tf.summary.scalar('loss', loss)

        summary_lr = tf.summary.scalar('learning_rate', args.learning_rate)
        summaries = tf.summary.merge([summary_lr, summary_loss])

        summary_log = tf.summary.FileWriter(
            os.path.join(CHECKPOINT_DIR, args.run_name))

        saver = tf.train.Saver(
            var_list=all_vars,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=2)
        sess.run(tf.global_variables_initializer())

        if args.restore_from == 'latest':
            ckpt = tf.train.latest_checkpoint(
                os.path.join(CHECKPOINT_DIR, args.run_name))
            if ckpt is None:
                # Get fresh GPT weights if new run.
                ckpt = tf.train.latest_checkpoint(
                    os.path.join('models', args.model_name))
        elif args.restore_from == 'fresh':
            ckpt = tf.train.latest_checkpoint(
                os.path.join('models', args.model_name))
        else:
            ckpt = tf.train.latest_checkpoint(args.restore_from)
        if ckpt:
            ckpt = os.path.join(BUCKET, ckpt)
        if not args.fresh_model:
            print('Loading checkpoint', ckpt)
            saver.restore(sess, ckpt)

        print('Loading dataset...')
        chunks = load_dataset(enc, args.dataset, args.combine)
        data_sampler = Sampler(chunks)
        if args.val_every > 0:
            val_chunks = load_dataset(enc, args.val_dataset, args.combine) if args.val_dataset else chunks
        print('dataset has', data_sampler.total_size, 'tokens')
        print('Training...')

        if args.val_every > 0:
            # Sample from validation set once with fixed seed to make
            # it deterministic during training as well as across runs.
            val_data_sampler = Sampler(val_chunks, seed=1)
            val_batches = [[val_data_sampler.sample(hparams.n_ctx) for _ in range(args.val_batch_size)]
                           for _ in range(args.val_batch_count)]

        counter = 1
        counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'counter')
        if os.path.exists(counter_path):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(counter_path, 'r') as fp:
                counter = int(fp.read()) + 1

        def save():
            maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
            print(
                'Saving',
                os.path.join(BUCKET, CHECKPOINT_DIR, args.run_name,
                             'model-{}').format(counter))
            saver.save(
                sess,
                os.path.join(BUCKET, CHECKPOINT_DIR, args.run_name, 'model'),
                global_step=counter)
            with open(counter_path, 'w') as fp:
                fp.write(str(counter) + '\n')

        def generate_samples():
            print('Generating samples...')
            context_tokens = data_sampler.sample(1)
            all_text = []
            index = 0
            while index < args.sample_num:
                out = sess.run(
                    tf_sample,
                    feed_dict={context: args.batch_size * [context_tokens]})
                for i in range(min(args.sample_num - index, args.batch_size)):
                    text = enc.decode(out[i])
                    text = '======== SAMPLE {} ========\n{}\n'.format(
                        index + 1, text)
                    print(text)
                    all_text.append(text)
                    index += 1
            maketree(os.path.join(SAMPLE_DIR, args.run_name))
            with open(
                    os.path.join(SAMPLE_DIR, args.run_name,
                                 'samples-{}').format(counter), 'w') as fp:
                fp.write('\n'.join(all_text))

        def validation():
            print('Calculating validation loss...')
            losses = []
            for batch in tqdm.tqdm(val_batches):
                losses.append(sess.run(val_loss, feed_dict={val_context: batch}))
            v_val_loss = np.mean(losses)
            v_summary = sess.run(val_loss_summary, feed_dict={val_loss: v_val_loss})
            summary_log.add_summary(v_summary, counter)
            summary_log.flush()
            print(
                '[{counter} | {time:2.4f}] validation loss = {loss:2.4f}'
                .format(
                    counter=counter,
                    time=time.time() - start_time,
                    loss=v_val_loss))

        start_time = time.time()
        
        def elapsed():
            return time.time() - start_time

        def say(msg):
            print('[{counter} | {time:2.4f}] {msg}'.format(counter=counter, time=elapsed(), msg=msg))

        def sample_batch():
            #return [data_sampler.sample(hparams.n_ctx) for _ in range(args.batch_size)]
            #say('Sampling batch...')
            r = []
            times = []
            for _ in range(args.batch_size):
                start = time.time()
                sample = data_sampler.sample(hparams.n_ctx)
                end = time.time()
                elapsed = (end - start)
                r += [sample]
                times += [elapsed]
            total = sum(times)
            avg = total / len(times)
            #say('Sampled %d batches in %.4f seconds (avg per batch: %.4f)' % (args.batch_size, total, avg))
            return r

        prev_time = time.time()
        avg_loss = (0.0, 0.0)

        try:
            while True:
                if counter % args.save_every == 0:
                    save()
                if counter % args.sample_every == 0:
                    generate_samples()
                if args.val_every > 0 and (counter % args.val_every == 0 or counter == 1):
                    validation()

                if args.accumulate_gradients > 1:
                    #say('Running opt_reset...')
                    sess.run(opt_reset)
                    for _ in range(args.accumulate_gradients):
                        batch = sample_batch()
                        say('Running opt_compute...')
                        sess.run(opt_compute, feed_dict={context: batch})
                    say('Running opt_apply...')
                    (v_loss, v_summary) = sess.run((opt_apply, summaries))
                else:
                    batch = sample_batch()
                    say('Running opt_apply...')
                    (_, v_loss, v_summary) = sess.run(
                        (opt_apply, loss, summaries),
                        feed_dict={context: batch})

                say('Adding summary...')
                summary_log.add_summary(v_summary, counter)

                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                now = time.time()
                print(
                        '[{counter} | {time:2.4f} | {delta:2.2f} | {ops:2.6f}/s] loss={loss:2.4f} avg={avg:2.4f}'
                    .format(
                        counter=counter,
                        time=now - start_time,
                        delta=now - prev_time,
                        ops=args.batch_size / (now - prev_time),
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]))
                prev_time = now

                counter += 1
        except KeyboardInterrupt:
            print('interrupted')
            if args.save_on_ctrlc:
                save()
        if tpu_cluster and args.init_tpu:
            print('Shutting down TPU system...')
            sess.run(tpu.shutdown_system())

def main_tpu():
    # Get the TPU's location
    # tpu_cluster = TPUClusterResolver().get_master()
    # print(tpu_cluster)
    with tf.Session(tpu_cluster) as sess:
        sess.run(tpu.initialize_system())
        main()
        sess.run(tpu.shutdown_system())
    main(tpu_cluster=tpu_cluster)

if __name__ == '__main__':
    main_tpu()
