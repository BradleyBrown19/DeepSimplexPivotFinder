# python spinningup/spinup/utils/test_policy.py spinningup/data/sunday_euc4/sunday_euc4_s0/ --exp_name euc4 --data_dir /Users/bradleybrown/Desktop/Waterloo/Courses/3A/CO255/DeepSimplexPivotFinder/data_generation/data/4_euclid_test/

# python spinningup/spinup/utils/test_policy.py spinningup/data/sunday_heuristic/sunday_heuristic_s0/ --exp_name heuristic --data_dir /Users/bradleybrown/Desktop/Waterloo/Courses/3A/CO255/DeepSimplexPivotFinder/data_generation/data/4_euclid_test/ --itr 20 --heuristic True

# python spinningup/spinup/utils/test_policy.py spinningup/data/sunday_non_euc4/sunday_non_euc4_s0/ --exp_name non_euc4 --data_dir /Users/bradleybrown/Desktop/Waterloo/Courses/3A/CO255/DeepSimplexPivotFinder/data_generation/data/4_euclid_test/ --itr 20 

python spinningup/spinup/utils/test_policy.py spinningup/data/sunday_objonly/sunday_objonly_s0/ --exp_name obj_only --data_dir /Users/bradleybrown/Desktop/Waterloo/Courses/3A/CO255/DeepSimplexPivotFinder/data_generation/data/4_euclid_test/ --itr 20 --full_tableau False

python spinningup/spinup/utils/test_policy.py spinningup/data/deep_simplex_clone/deep_simplex_clone_s0/ --exp_name ds_clone --data_dir /Users/bradleybrown/Desktop/Waterloo/Courses/3A/CO255/DeepSimplexPivotFinder/data_generation/data/4_euclid_test/ --full_tableau False --heuristic True