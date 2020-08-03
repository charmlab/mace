# python3 batchTest.py -d compass -m lr -n one_norm -a MACE_eps_1e-5 -s 5
# python3 batchTest.py -d compass -m lr -n one_norm -a MO -s 5
# python3 batchTest.py -d compass -m lr -n one_norm -a MACE_eps_1e-5 MO -s 5

# for f in $(ls | grep MACE_eps_1e-5); do echo $f; ls $f/__explanation_log/ | wc -l; done
# ls -1 | grep 2019.05 | xargs rm -rf
# scp -r amir@login.cluster.is.localnet:~/dev/mace/_experiments/__merged _results/

DATASET_VALUES = ['credit', 'adult']
# MODEL_CLASS_VALUES = ['forest', 'tree', 'lr']
# MODEL_CLASS_VALUES = ['mlp2x10']
MODEL_CLASS_VALUES = ['lr']
NORM_VALUES = ['zero_norm', 'one_norm', 'two_norm', 'infty_norm']
APPROACHES_VALUES = ['MACE_MIP_OBJ_eps_1e-3', 'MACE_MIP_EXP_eps_1e-3']
# APPROACHES_VALUES = ['MACE_MIP_SAT_eps_1e-3', 'MACE_SAT_eps_1e-3']

NUM_BATCHES = 1
NUM_SAMPLES_PER_BATCH = 500
GEN_CF_FOR = 'neg_and_pos'
# REPEAT = 10

# NUM_BATCHES = 100
# NUM_NEG_SAMPLES_PER_BATCH = 5

request_memory = 8192

sub_file_name = f'{NUM_BATCHES}_BATCH_{NUM_SAMPLES_PER_BATCH}_SAMPLES__'

sub_file_name += 'DATASET_'
for dataset_string in DATASET_VALUES:
  sub_file_name += dataset_string + '_'
sub_file_name += '_MODEL__'
for model_class_string in MODEL_CLASS_VALUES:
  sub_file_name += model_class_string + '_'
sub_file_name += '_NORM__'
for norm_type_string in NORM_VALUES:
  sub_file_name += norm_type_string + '_'
sub_file_name += '_APPROACH__'
for approach_string in APPROACHES_VALUES:
  sub_file_name += approach_string + '_'


sub_file = open(sub_file_name + '.sub','w')
print('executable = /home/kmohammadi/anaconda3/envs/_mip_smt/bin/python', file=sub_file)
print('error = _cluster_logs/test.$(Process).err', file=sub_file)
print('output = _cluster_logs/test.$(Process).out', file=sub_file)
print('log = _cluster_logs/test.$(Process).log', file=sub_file)
print(f'request_memory = {request_memory}', file=sub_file)
print('request_cpus = 1', file=sub_file)
print('\n' * 4, file=sub_file)


for dataset_string in DATASET_VALUES:

  for model_class_string in MODEL_CLASS_VALUES:

    for norm_type_string in NORM_VALUES:

      for approach_string in APPROACHES_VALUES:

        for batch_number in range(NUM_BATCHES):


          print(f'arguments = batchTest.py' + \
             f' -d {dataset_string}' \
             f' -m {model_class_string}' \
             f' -n {norm_type_string}' \
             f' -a {approach_string}' \
             f' -b {batch_number}' \
             f' -s {NUM_SAMPLES_PER_BATCH}', \
             f' -g {GEN_CF_FOR}', \
             f' -p $(Process)', \
                file=sub_file)
          # print(f'requirements = CpuModel =?= "Intel(R) Xeon(R) Gold 5220 CPU @ 2.20GHz"', file=sub_file)
          print(f'queue', file=sub_file)
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


# python batchTest.py -d adult -m lr -n one_norm -a MACE_eps_1e-5 -b 0 -s 500
# python batchTest.py -d adult -m lr -n infty_norm -a MACE_eps_1e-5 -b 0 -s 500
# python batchTest.py -d credit -m lr -n one_norm -a MACE_eps_1e-5 -b 0 -s 500
# python batchTest.py -d credit -m lr -n infty_norm -a MACE_eps_1e-5 -b 0 -s 500
# python batchTest.py -d compass -m lr -n one_norm -a MACE_eps_1e-5 -b 0 -s 500
# python batchTest.py -d compass -m lr -n infty_norm -a MACE_eps_1e-5 -b 0 -s 500
