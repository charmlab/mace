# python3 batchTest.py -d compass -m lr -n one_norm -a SAT -s 5
# python3 batchTest.py -d compass -m lr -n one_norm -a MO -s 5
# python3 batchTest.py -d compass -m lr -n one_norm -a SAT MO -s 5

# for f in $(ls | grep SAT); do echo $f; ls $f/__explanation_log/ | wc -l; done
# ls -1 | grep 2019.05 | xargs rm -rf

DATASET_VALUES = ['adult', 'credit', 'compass']
MODEL_CLASS_VALUES = ['tree', 'forest', 'lr', 'mlp']
NORM_VALUES = ['zero_norm', 'one_norm', 'infty_norm']
APPROACHES_VALUES = ['SAT']

NUM_BATCHES = 100
NUM_NEG_SAMPLES_PER_BATCH = 5

sub_file = open('test.sub','w')

for dataset_string in DATASET_VALUES:

  for model_class_string in MODEL_CLASS_VALUES:

    for norm_type_string in NORM_VALUES:

      for approach_string in APPROACHES_VALUES:

        request_memory = 8192

        for batch_number in range(NUM_BATCHES):

          print('executable = /home/amir/dev/mace/_venv/bin/python', file=sub_file)
          print(f'arguments = batchTest.py' + \
             f' -d {dataset_string}' \
             f' -m {model_class_string}' \
             f' -n {norm_type_string}' \
             f' -a {approach_string}' \
             f' -b {batch_number}' \
             f' -s {NUM_NEG_SAMPLES_PER_BATCH}', \
          file=sub_file)
          print('error = _cluster_logs/test.$(Process).err', file=sub_file)
          print('output = _cluster_logs/test.$(Process).out', file=sub_file)
          print('log = _cluster_logs/test.$(Process).log', file=sub_file)
          print(f'request_memory = {request_memory}', file=sub_file)
          print('request_cpus = 1', file=sub_file)
          print('queue', file=sub_file)
          print('\n', file=sub_file)








# # For Actionable Recourse only
# python batchTest.py -d adult -m lr -n one_norm -a AR -b 0 -s 500
# python batchTest.py -d adult -m lr -n infty_norm -a AR -b 0 -s 500
# python batchTest.py -d credit -m lr -n one_norm -a AR -b 0 -s 500
# python batchTest.py -d credit -m lr -n infty_norm -a AR -b 0 -s 500
# python batchTest.py -d compass -m lr -n one_norm -a AR -b 0 -s 500
# python batchTest.py -d compass -m lr -n infty_norm -a AR -b 0 -s 500


# python batchTest.py -d adult -m lr -n one_norm -a MO -b 0 -s 500
# python batchTest.py -d adult -m lr -n infty_norm -a MO -b 0 -s 500
# python batchTest.py -d credit -m lr -n one_norm -a MO -b 0 -s 500
# python batchTest.py -d credit -m lr -n infty_norm -a MO -b 0 -s 500
# python batchTest.py -d compass -m lr -n one_norm -a MO -b 0 -s 500
# python batchTest.py -d compass -m lr -n infty_norm -a MO -b 0 -s 500


# python batchTest.py -d adult -m lr -n one_norm -a SAT -b 0 -s 500
# python batchTest.py -d adult -m lr -n infty_norm -a SAT -b 0 -s 500
# python batchTest.py -d credit -m lr -n one_norm -a SAT -b 0 -s 500
# python batchTest.py -d credit -m lr -n infty_norm -a SAT -b 0 -s 500
# python batchTest.py -d compass -m lr -n one_norm -a SAT -b 0 -s 500
# python batchTest.py -d compass -m lr -n infty_norm -a SAT -b 0 -s 500
