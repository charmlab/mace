
def measureSensitiveAttributeChange():
  df_all_distances = pickle.load(open(f'_results/df_all_distances', 'rb'))
  df_all_distances = df_all_distances.where(
    (df_all_distances['counterfactual found'] == True) &
    (df_all_distances['counterfactual plausible'] == True)
  ).dropna()
  for model_class_string in MODEL_CLASS_VALUES:
    for approach_string in APPROACHES_VALUES:
      for dataset_string in DATASET_VALUES:
        for norm_type_string in NORM_VALUES:
          df = df_all_distances.where(
            (df_all_distances['dataset'] == dataset_string) &
            (df_all_distances['model'] == model_class_string) &
            (df_all_distances['norm'] == norm_type_string) &
            (df_all_distances['approach'] == approach_string),
          ).dropna()
          if df.shape[0]: # if any tests exist for this setup
            age_changed = df.where(df['changed age'] == True).dropna()
            age_not_changed = df.where(df['changed age'] == False).dropna()
            gender_changed = df.where(df['changed gender'] == True).dropna()
            gender_not_changed = df.where(df['changed gender'] == False).dropna()
            race_changed = df.where(df['changed race'] == True).dropna()
            race_not_changed = df.where(df['changed race'] == False).dropna()

            sensitive_changed = df.where(
              (df['changed age'] == True) |
              (df['changed gender'] == True) |
              (df['changed race'] == True)
            ).dropna()
            sensitive_not_changed = df.where(
              (df['changed age'] == False) &
              (df['changed gender'] == False) &
              (df['changed race'] == False)
            ).dropna()

            assert df.shape[0] == age_changed.shape[0] + age_not_changed.shape[0]
            assert df.shape[0] == gender_changed.shape[0] + gender_not_changed.shape[0]
            assert df.shape[0] == race_changed.shape[0] + race_not_changed.shape[0]
            assert df.shape[0] == sensitive_changed.shape[0] + sensitive_not_changed.shape[0]

            print(f'{dataset_string}-{model_class_string}-{norm_type_string}-{approach_string}:'.ljust(40), end = '')
            print(f'\t\tpercent age change: % {100 * age_changed.shape[0] / df.shape[0]:.2f}', end = '')
            print(f'\t\tpercent gender change: % {100 * gender_changed.shape[0] / df.shape[0]:.2f}', end = '')
            if dataset_string == 'compass':
              print(f'\t\tpercent race change: % {100 * race_changed.shape[0] / df.shape[0]:.2f}', end = '')
            print(f'\t\tsensitive_changed: {sensitive_changed.shape[0]}, \t sensitive_not_changed: {sensitive_not_changed.shape[0]}, \t percent: % {100 * sensitive_changed.shape[0] / df.shape[0]:.2f}', end = '\n')


def measureEffectOfAgeCompass():
  df_all_distances = pickle.load(open(f'_results/df_all_distances', 'rb'))
  df_all_distances = df_all_distances.where(
    (df_all_distances['counterfactual found'] == True) &
    (df_all_distances['counterfactual plausible'] == True)
  ).dropna()
  model_class_string = 'lr'
  dataset_string = 'compass'
  norm_type_string = 'zero_norm'
  for approach_string in ['MACE_eps_1e-5', 'MO']:

    df = df_all_distances.where(
      (df_all_distances['dataset'] == dataset_string) &
      (df_all_distances['model'] == model_class_string) &
      (df_all_distances['norm'] == norm_type_string) &
      (df_all_distances['approach'] == approach_string),
    ).dropna()
    if df.shape[0]: # if any tests exist for this setup
      age_changed = df.where(df['changed age'] == True).dropna()
      age_not_changed = df.where(df['changed age'] == False).dropna()
      assert df.shape[0] == age_changed.shape[0] + age_not_changed.shape[0]

      print(f'\n\n{dataset_string}-{model_class_string}-{norm_type_string}-{approach_string}:'.ljust(40))

      # print(f'\tage_changed: {age_changed.shape[0]} samples')
      # uniques, counts = np.unique(age_changed["counterfactual distance"], return_counts = True)
      # for idx, elem in enumerate(uniques):
      #   print(f'\t\t{counts[idx]} samples at distance {uniques[idx]}')

      # print(f'\tage_not_changed: {age_not_changed.shape[0]} samples')
      # uniques, counts = np.unique(age_not_changed["counterfactual distance"], return_counts = True)
      # for idx, elem in enumerate(uniques):
      #   print(f'\t\t{counts[idx]} samples at distance {uniques[idx]}')

      print(f'\tage_changed: {age_changed.shape[0]} samples')
      for unique_distance in np.unique(age_changed["counterfactual distance"]):
        unique_distance_df = age_changed.where(age_changed['counterfactual distance'] == unique_distance).dropna()
        print(f'\t\t{unique_distance_df.shape[0]} samples at distance {unique_distance}')
        # unique_attr_changes, counts = np.unique(unique_distance_df["changed attributes"], return_counts = True)
        # for idx, unique_attr_change in enumerate(unique_attr_changes):
        #   print(f'\t\t\t{counts[idx]} samples changing attributes {unique_attr_change}')

      print(f'\tage_not_changed: {age_not_changed.shape[0]} samples')
      for unique_distance in np.unique(age_not_changed["counterfactual distance"]):
        unique_distance_df = age_not_changed.where(age_not_changed['counterfactual distance'] == unique_distance).dropna()
        print(f'\t\t{unique_distance_df.shape[0]} samples at distance {unique_distance}')
        # unique_attr_changes, counts = np.unique(unique_distance_df["changed attributes"], return_counts = True)
        # for idx, unique_attr_change in enumerate(unique_attr_changes):
        #   print(f'\t\t\t{counts[idx]} samples changing attributes {unique_attr_change}')

      print(f'\tpercent: % {100 * age_changed.shape[0] / df.shape[0]:.2f}', end = '\n')


def measureEffectOfRaceCompass():
  pairs = [
    (
      'compass-lr-one_norm-AR',
      '/Users/a6karimi/dev/mace/_experiments/2019.05.23_14.10.06__compass__lr__one_norm__AR__batch0__samples500/_minimum_distances',
      '/Users/a6karimi/dev/mace/_experiments/2019.05.23_14.10.40__compass__lr__one_norm__AR__batch0__samples500/_minimum_distances',
    ), \
    # ('compass-lr-infty_norm-AR', ), \
  ]
  print('\n\n<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n\n')
  for pair in pairs:
    unrestricted_file = pickle.load(open(pair[1], 'rb'))
    restricted_file = pickle.load(open(pair[2], 'rb'))
    increase_in_distance = []
    unrestricted_distances = []
    restricted_distances = []
    for factual_sample_index in unrestricted_file.keys():

      restricted_factual_sample = restricted_file[factual_sample_index]['factual_sample']
      unrestricted_factual_sample = unrestricted_file[factual_sample_index]['factual_sample']
      restricted_counterfactual_sample = restricted_file[factual_sample_index]['counterfactual_sample']
      unrestricted_counterfactual_sample = unrestricted_file[factual_sample_index]['counterfactual_sample']
      assert restricted_factual_sample == unrestricted_factual_sample

      # if it changes race
      if not np.isclose(unrestricted_factual_sample['x3'], unrestricted_counterfactual_sample['x3']):
        unrestricted_distance = unrestricted_counterfactual_sample['counterfactual_distance']
        restricted_distance = restricted_counterfactual_sample['counterfactual_distance']
        try:
          assert (restricted_distance >= unrestricted_distance) or np.isclose(restricted_distance, unrestricted_distance, 1e-3, 1e-3)
        except:
          print(f'\t Unexpected: \t\t restricted_distance - unrestricted_distance = {restricted_distance - unrestricted_distance}')
        increase_in_distance.append(restricted_distance / unrestricted_distance)
        unrestricted_distances.append(unrestricted_distance)
        restricted_distances.append(restricted_distance)

    print(f'{pair[0]}:')
    print(f'\t\tMean unrestricted distance = {np.mean(unrestricted_distances):.4f}')
    print(f'\t\tMean restricted distance = {np.mean(restricted_distances):.4f}')
    print(f'\t\tMean increase in distance (restricted / unrestricted) = {np.mean(increase_in_distance):.4f}')

      # factual_sample = restricted_factual_sample # or unrestricted_factual_sample
      # restricted_changed_attributes = []
      # unrestricted_changed_attributes = []
      # for attr in factual_sample.keys():
      #   if not np.isclose(factual_sample[attr], restricted_counterfactual_sample[attr]):
      #     restricted_changed_attributes.append((attr, factual_sample[attr], restricted_counterfactual_sample[attr]))
      #   if not np.isclose(factual_sample[attr], unrestricted_counterfactual_sample[attr]):
      #     unrestricted_changed_attributes.append((attr, factual_sample[attr], unrestricted_counterfactual_sample[attr]))

      # # restricted_changed_attributes.pop('y')
      # # unrestricted_changed_attributes.pop('y')

      # print(f'Sample: {factual_sample_index}')
      # print(f'\trestricted_changed_attributes: {restricted_changed_attributes}')
      # print(f'\tunrestricted_changed_attributes: {unrestricted_changed_attributes}')


def measureEffectOfAgeAdultPart1():
  df_all_distances = pickle.load(open(f'_results/df_all_distances', 'rb'))
  df_all_distances = df_all_distances.where(
    (df_all_distances['counterfactual found'] == True) &
    (df_all_distances['counterfactual plausible'] == True)
  ).dropna()

  model_class_string = 'forest'
  dataset_string = 'adult'
  for approach_string in ['MACE_eps_1e-5', 'MO', 'PFT']:
    for norm_type_string in ['zero_norm', 'one_norm', 'infty_norm']:

      df = df_all_distances.where(
        (df_all_distances['dataset'] == dataset_string) &
        (df_all_distances['model'] == model_class_string) &
        (df_all_distances['norm'] == norm_type_string) &
        (df_all_distances['approach'] == approach_string),
      ).dropna()
      if df.shape[0]: # if any tests exist for this setup

        age_constant = df.where(df['age constant'] == True).dropna()
        age_increased = df.where(df['age increased'] == True).dropna()
        age_decreased = df.where(df['age decreased'] == True).dropna()
        assert df.shape[0] == age_constant.shape[0] + age_increased.shape[0] + age_decreased.shape[0]

        test_name = f'{dataset_string}-{model_class_string}-{norm_type_string}-{approach_string}'

        print(f'\n\n{test_name}:'.ljust(40))

        print(f'\tage_increased: {age_increased.shape[0]} samples \t % {100 * age_increased.shape[0] / df.shape[0]} \t average cost: {np.mean(age_increased["counterfactual distance"]):.4f}')
        # for unique_distance in np.unique(age_increased["counterfactual distance"]):
        #   unique_distance_df = age_increased.where(age_increased['counterfactual distance'] == unique_distance).dropna()
        #   print(f'\t\t{unique_distance_df.shape[0]} samples at distance {unique_distance}')
          # unique_attr_changes, counts = np.unique(unique_distance_df["changed attributes"], return_counts = True)
          # for idx, unique_attr_change in enumerate(unique_attr_changes):
          #   print(f'\t\t\t{counts[idx]} samples changing attributes {unique_attr_change}')

        print(f'\tage_constant: {age_constant.shape[0]} samples \t % {100 * age_constant.shape[0] / df.shape[0]} \t average cost: {np.mean(age_constant["counterfactual distance"]):.4f}')
        # for unique_distance in np.unique(age_constant["counterfactual distance"]):
        #   unique_distance_df = age_constant.where(age_constant['counterfactual distance'] == unique_distance).dropna()
        #   print(f'\t\t{unique_distance_df.shape[0]} samples at distance {unique_distance}')
          # unique_attr_changes, counts = np.unique(unique_distance_df["changed attributes"], return_counts = True)
          # for idx, unique_attr_change in enumerate(unique_attr_changes):
          #   print(f'\t\t\t{counts[idx]} samples changing attributes {unique_attr_change}')

        print(f'\tage_decreased: {age_decreased.shape[0]} samples \t % {100 * age_decreased.shape[0] / df.shape[0]} \t average cost: {np.mean(age_decreased["counterfactual distance"]):.4f}')
        # for unique_distance in np.unique(age_decreased["counterfactual distance"]):
        #   unique_distance_df = age_decreased.where(age_decreased['counterfactual distance'] == unique_distance).dropna()
        #   print(f'\t\t{unique_distance_df.shape[0]} samples at distance {unique_distance}')
          # unique_attr_changes, counts = np.unique(unique_distance_df["changed attributes"], return_counts = True)
          # for idx, unique_attr_change in enumerate(unique_attr_changes):
          #   print(f'\t\t\t{counts[idx]} samples changing attributes {unique_attr_change}')

        pickle.dump(age_increased, open(f'_results/{test_name}_age_increased_df', 'wb'))
        pickle.dump(age_decreased, open(f'_results/{test_name}_age_decreased_df', 'wb'))


def measureEffectOfAgeAdultPart2():

  # pairs = [
  #   ('adult-forest-zero_norm-SAT', '_results/adult-forest-zero_norm-SAT_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_21_restricted_results_no_age_reduction/2019.05.21_17.17.39__adult__forest__zero_norm__SAT/_minimum_distances'), \
  #   ('adult-forest-one_norm-SAT', '_results/adult-forest-one_norm-SAT_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_21_restricted_results_no_age_reduction/2019.05.21_17.37.32__adult__forest__one_norm__SAT/_minimum_distances'), \
  #   ('adult-forest-infty_norm-SAT', '_results/adult-forest-infty_norm-SAT_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_21_restricted_results_no_age_reduction/2019.05.21_17.46.25__adult__forest__infty_norm__SAT/_minimum_distances'), \
  #   ('adult-forest-zero_norm-MO', '_results/adult-forest-zero_norm-MO_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_21_restricted_results_no_age_reduction/2019.05.21_17.19.16__adult__forest__zero_norm__MO/_minimum_distances'), \
  #   ('adult-forest-one_norm-MO', '_results/adult-forest-one_norm-MO_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_21_restricted_results_no_age_reduction/2019.05.21_17.37.35__adult__forest__one_norm__MO/_minimum_distances'), \
  #   ('adult-forest-infty_norm-MO', '_results/adult-forest-infty_norm-MO_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_21_restricted_results_no_age_reduction/2019.05.21_17.54.09__adult__forest__infty_norm__MO/_minimum_distances'), \
  # ]
  pairs = [
    ('adult-forest-zero_norm-SAT', '_results/adult-forest-zero_norm-SAT_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_22_restricted_results_no_age_change/2019.05.22_17.42.00__adult__forest__zero_norm__SAT/_minimum_distances'), \
    ('adult-forest-one_norm-SAT', '_results/adult-forest-one_norm-SAT_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_22_restricted_results_no_age_change/2019.05.22_18.17.48__adult__forest__one_norm__SAT/_minimum_distances'), \
    ('adult-forest-infty_norm-SAT', '_results/adult-forest-infty_norm-SAT_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_22_restricted_results_no_age_change/2019.05.22_18.36.35__adult__forest__infty_norm__SAT/_minimum_distances'), \
    ('adult-forest-zero_norm-MO', '_results/adult-forest-zero_norm-MO_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_22_restricted_results_no_age_change/2019.05.22_18.00.02__adult__forest__zero_norm__MO/_minimum_distances'), \
    ('adult-forest-one_norm-MO', '_results/adult-forest-one_norm-MO_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_22_restricted_results_no_age_change/2019.05.22_18.20.50__adult__forest__one_norm__MO/_minimum_distances'), \
    ('adult-forest-infty_norm-MO', '_results/adult-forest-infty_norm-MO_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_22_restricted_results_no_age_change/2019.05.22_18.32.33__adult__forest__infty_norm__MO/_minimum_distances'), \
  ]
  print('\n\n<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n\n')
  for pair in pairs:
    unrestricted_df = pickle.load(open(pair[1], 'rb'))
    restricted_file = pickle.load(open(pair[2], 'rb'))
    increase_in_distance = []
    unrestricted_distances = []
    restricted_distances = []
    for index, row in unrestricted_df.iterrows():
      factual_sample_index = row.loc['factual sample index']
      unrestricted_distance = row.loc['counterfactual distance']
      restricted_distance = restricted_file[factual_sample_index]['counterfactual_distance']
      try:
        assert (restricted_distance >= unrestricted_distance) or np.isclose(restricted_distance, unrestricted_distance, 1e-3, 1e-3)
      except:
        print(f'\t Unexpected: \t\t restricted_distance - unrestricted_distance = {restricted_distance - unrestricted_distance}')
      increase_in_distance.append(restricted_distance / unrestricted_distance)
      unrestricted_distances.append(unrestricted_distance)
      restricted_distances.append(restricted_distance)
    print(f'{pair[0]}:')
    print(f'\t\tMean unrestricted distance = {np.mean(unrestricted_distances):.4f}')
    print(f'\t\tMean restricted distance = {np.mean(restricted_distances):.4f}')
    print(f'\t\tMean increase in distance (restricted / unrestricted) = {np.mean(increase_in_distance):.4f}')


def measureEffectOfAgeAdultPart3():

  pairs = [(
    'adult-forest-zero_norm-SAT',
    '_results/adult-forest-zero_norm-SAT_age_decreased_df',
    '_results/adult-forest-zero_norm-SAT_age_increased_df',
    '/Volumes/amir/dev/mace/_experiments/_may_20_all_results/2019.05.20_17.44.50__adult__forest__zero_norm__SAT/_minimum_distances',
    '/Volumes/amir/dev/mace/_experiments/_may_22_restricted_results_no_age_change/2019.05.22_17.42.00__adult__forest__zero_norm__SAT/_minimum_distances'
  )]
  print('\n\n<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n\n')
  for pair in pairs:
    unrestricted_df1 = pickle.load(open(pair[1], 'rb'))
    unrestricted_df2 = pickle.load(open(pair[2], 'rb'))
    unrestricted_df = unrestricted_df1.append(unrestricted_df2, ignore_index=True)
    unrestricted_file = pickle.load(open(pair[3], 'rb'))
    restricted_file = pickle.load(open(pair[4], 'rb'))
    # assert unrestricted_df.shape[0] == 18 # %3.6 x 500
    for index, row in unrestricted_df.iterrows():

      factual_sample_index = row.loc['factual sample index']

      # unrestricted_distance = row.loc['counterfactual distance']
      # restricted_distance = restricted_file[factual_sample_index]['counterfactual_distance']
      # assert unrestricted_distance == restricted_distance

      restricted_factual_sample = restricted_file[factual_sample_index]['factual_sample']
      unrestricted_factual_sample = unrestricted_file[factual_sample_index]['factual_sample']
      restricted_counterfactual_sample = restricted_file[factual_sample_index]['counterfactual_sample']
      unrestricted_counterfactual_sample = unrestricted_file[factual_sample_index]['counterfactual_sample']
      assert restricted_factual_sample == unrestricted_factual_sample

      factual_sample = restricted_factual_sample # or unrestricted_factual_sample
      restricted_changed_attributes = []
      unrestricted_changed_attributes = []
      for attr in factual_sample.keys():
        if not np.isclose(factual_sample[attr], restricted_counterfactual_sample[attr]):
          restricted_changed_attributes.append((attr, factual_sample[attr], restricted_counterfactual_sample[attr]))
        if not np.isclose(factual_sample[attr], unrestricted_counterfactual_sample[attr]):
          unrestricted_changed_attributes.append((attr, factual_sample[attr], unrestricted_counterfactual_sample[attr]))

      # restricted_changed_attributes.pop('y')
      # unrestricted_changed_attributes.pop('y')

      print(f'Sample: {factual_sample_index}')
      print(f'\trestricted_changed_attributes: {restricted_changed_attributes}')
      print(f'\tunrestricted_changed_attributes: {unrestricted_changed_attributes}')

