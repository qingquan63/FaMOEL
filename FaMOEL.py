# -*- coding: utf-8 -*-
import FairEMOL as ea
from FairEMOL.FairProblem import FairProblem
import time
import numpy as np
import sys
import argparse


def run(parameters):
    start_time = parameters['start_time']
    print('The time is ', start_time)
    """===============================实例化问题对象============================"""
    problem = FairProblem(M=len(parameters['objectives_class']), learning_rate=parameters['learning_rate'],
                          batch_size=parameters['batch_size'],
                          sensitive_attributions=parameters['sensitive_attributions'],
                          epoches=parameters['epoches'], dataname=parameters['dataname'],
                          objectives_class=parameters['objectives_class'],
                          dirname='Result/' + parameters['start_time'],
                          seed_split_traintest=parameters['seed_split_traintest'],
                          start_time=parameters['start_time'],
                          is_ensemble=parameters["is_ensemble"],
                          parameters=parameters)
    """==================================种群设置==============================="""
    Encoding = 'NN'
    NIND = parameters['NIND']
    Field = None
    population = ea.Population(Encoding, Field, NIND, n_feature=problem.getFeature(),
                               n_hidden=parameters['n_hidden'],
                               n_output=parameters['n_output'],
                               parameters=parameters,
                               logits=np.zeros([NIND, problem.test_data.shape[0]]),
                               is_ensemble=parameters["is_ensemble"])  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = ea.Fair_MOEA_template(problem=problem, start_time=start_time,
                                        population=population,
                                        muta_mu=parameters['muta_mu'],
                                        muta_var=parameters['muta_var'],
                                        calculmetric=parameters['logMetric'],
                                        run_id=parameters['run_id'],
                                        MOEAs=parameters['MOEAs'],
                                        mutation_p=parameters["mutation_p"],
                                        crossover_p=parameters["crossover_p"],
                                        record_parameter=parameters["record_parameter"],
                                        is_ensemble=parameters["is_ensemble"],
                                        parameters=parameters)
    myAlgorithm.MAXGEN = parameters['MAXGEN']
    myAlgorithm.logTras = parameters['logTras']
    myAlgorithm.verbose = parameters['verbose']
    myAlgorithm.drawing = parameters['drawing']

    myAlgorithm.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='run FairMOEL')
    parser.add_argument('--dataname', type=str, default='german', help='the name of dataset')
    parser.add_argument('--dirname', type=str, default='Result_all', help='')  # Result_all Test
    parser.add_argument('--kfold', type=int, default=5, help='the numbe of kfold')
    parser.add_argument('--fold_id', type=int, default=4, help='the i-th fold')
    parser.add_argument('--start_run', type=int, default=0, help='the starting run')
    parser.add_argument('--end_run', type=int, default=0, help='the ending run')
    parser.add_argument('--CA_size', type=int, default=100, help='CA_size')
    parser.add_argument('--recall_gen', type=int, default=10, help='recall gen')
    parser.add_argument('--throd', type=float, default=0.22, help='recall gen')
    parser.add_argument('--is_smaller', type=int, default=0, help='recall gen')

    args = parser.parse_args()

    dataname = args.dataname
    start_run = args.start_run
    end_run = args.end_run
    kfold = args.kfold
    fold_id = args.fold_id
    CA_size = args.CA_size
    recall_gen = args.recall_gen
    throd = args.throd
    dirname = args.dirname
    is_smaller = args.is_smaller

    list_objs = ['accuracy',
                'true_positive_rate_difference',
                'false_positive_rate_difference',
                'false_negative_rate_difference',
                'false_omission_rate_difference',
                'false_discovery_rate_difference',
                'false_positive_rate_ratio',
                'false_negative_rate_ratio',
                'false_omission_rate_ratio',
                'false_discovery_rate_ratio',
                'average_odds_difference',
                'average_abs_odds_difference',
                'error_rate_difference',
                'error_rate_ratio',
                'disparate_impact',
                'statistical_parity_difference',
                'generalized_entropy_index',
                'betweeen_all_groups_generalized_entropy_index',
                'betweeen_group_generalized_entropy_index',
                'theil_index',
                'coefficient_of_variation',
                'between_group_theil_index',
                'between_group_coefficient_of_variation',
                'between_all_groups_theil_index',
                'between_all_groups_coefficient_of_variation',
                'differential_fairness_bias_amplification']

    mutation_p, crossover_p = 1, 1
    batch_size = np.inf
    MOEAs = 'TwoArch2'  
    popsize = 100
    objectives_class = list_objs

    is_ensemble = False  
    drawing = 0      
    logMetric = 10
    preserve_sens_in_net = 0
    logTras = 1
    n_output = 1
    muta_mu = 0
    verbose = False
    dirName = '{}/{}'.format(dirname, MOEAs)
    dirName += '_roc_on_ave{}RNCC_{}'.format(recall_gen, int(throd*100)) 
    start_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))  
    seed_split_traintest = 20230223
    Encoding = 'BG'  
    use_gpu = True
    Parameters_all = {'german': {'sens_attri': ['sex'], 'lr': 0.0001, 'weight_decay': 0.01, 'muta_var': 0.05, 'epoch': 1, 'MAXGEN': 100, 'CA_size': 100, "n_hidden": 64},}
    Parameters_algo = Parameters_all[dataname]
    Parameters_algo['CA_size'] = CA_size
    for run_id in range(start_run, end_run+1):
        sensitive_attributions = Parameters_algo['sens_attri']
        learning_rate = Parameters_algo['lr']
        weight_decay = Parameters_algo['weight_decay']
        muta_var = Parameters_algo['muta_var']
        epoches = Parameters_algo['epoch']
        MAXGEN = Parameters_algo['MAXGEN']
        n_hidden = [Parameters_algo['n_hidden']]
        CA_size = Parameters_algo['CA_size']
        start_time = str(run_id)
        record_parameter = ['learning_rate', 'batch_size', 'n_hidden', 'n_output', 'epoches',
                            'muta_mu', 'muta_var', 'NIND', 'MAXGEN', 'dataname', 'sensitive_attributions',
                            'objectives_class', 'preserve_sens_in_net', 'weight_decay', "cal_obj_plan",
                            "dropout", "MOEAs", "crossover_p", "mutation_p", "obj_is_logits", "is_ensemble",
                            "kfold", "fold_id", "CA_size", "throd"]
        parameters = {'dataname': dataname, 'start_time': start_time, 'learning_rate': learning_rate, 'batch_size': batch_size,
                      'n_hidden': n_hidden, 'n_output': n_output, 'epoches': epoches, 'muta_mu': muta_mu,
                      'muta_var': muta_var, 'NIND': popsize, 'MAXGEN': MAXGEN+1,
                      'logTras': logTras, 'verbose': verbose, 'drawing': drawing, 'dirName': dirName,
                      'sensitive_attributions': sensitive_attributions, 'logMetric': logMetric,
                      'preserve_sens_in_net': preserve_sens_in_net, 'seed_split_traintest': seed_split_traintest,
                      'run_id': run_id, "MOEAs": MOEAs, "crossover_p": crossover_p, "mutation_p": mutation_p,
                      'record_parameter': record_parameter, "is_ensemble": is_ensemble, 'objectives_class': objectives_class,
                      'kfold': kfold, 'fold_id': fold_id, "use_gpu": use_gpu, "CA_size": CA_size,
                      "recall_gen": recall_gen, "throd": throd, "is_smaller": is_smaller}

        print(parameters)
        run(parameters)

