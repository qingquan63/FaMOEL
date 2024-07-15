# -*- coding: utf-8 -*-
import numpy as np
import FairEMOL as ea
from sys import path as paths
from os import path
import torch
import time
import os
import copy
from FairEMOL.templates.select_objectives import Select_Ojbectives_cut
from FairEMOL.templates.select_objectives import RNCC as RNCC_fun


paths.append(path.split(path.split(path.realpath(__file__))[0])[0])


def save_model(population, gen, filepath='nets/gen%d_net%s.pth', start_time=None, base_res_path=None):
    for idx in range(len(population)):
        NN = population.Chrom[idx]
        save_filename = filepath % (gen, idx)
        save_path = os.path.join(base_res_path + start_time, save_filename)
        torch.save(NN.state_dict(), save_path)


class Fair_MOEA_template(ea.MoeaAlgorithm):

    def __init__(self, problem, start_time, population, muta_mu=0, muta_var=0.001, objectives=None,
                 calculmetric=20, run_id=0, MOEAs=1, mutation_p=0.2, crossover_p=0.8,
                 record_parameter=None, is_ensemble=False, parameters=None):
        ea.MoeaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        if objectives is None:
            objectives = ['Individual_fairness', 'Group_fairness']
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'NSGA2'
        if self.problem.M < 10:
            self.ndSort = None  # 采用ENS_SS进行非支配排序
        else:
            self.ndSort = None  # 高维目标采用T_ENS进行非支配排序，速度一般会比ENS_SS要快
        self.selFunc = 'tour'  # 选择方式，采用锦标赛选择

        self.recOper = ea.Crossover_NN(crossover_p)
        self.mutOper = ea.Mutation_NN(mu=muta_mu, var=muta_var, p=mutation_p)

        self.start_time = start_time
        self.dirName = 'Result/'
        self.objectives_class = objectives
        self.calculmetric = calculmetric
        self.run_id = run_id
        self.MOEAs = MOEAs
        self.mutation_p = mutation_p
        self.crossover_p = crossover_p
        self.muta_mu = muta_mu
        self.muta_var_org = muta_var
        self.muta_var = muta_var
        self.record_parameter = record_parameter
        self.is_ensemble = is_ensemble
        self.all_metric_valid = {}
        self.all_metric_test = {}
        self.all_metric_train = {}
        self.parameters = parameters
        self.select_objectivs = None
        self.select_objectivs_record = {}
        self.CA = None
        self.CA_size = self.parameters["CA_size"]
        self.RNCC = None
        self.RNCC_record = {}
        self.ALL_objs = None
        self.objs_record = {}
        self.gen = 0
        self.recall_gen = self.parameters["recall_gen"]


    def reinsertion(self, population, offspring, NUM, isNN=1):
        population = population + offspring
        population.setisNN(1)

        Objs_now = copy.deepcopy(population.ObjV)
        Objs_now = Objs_now[:, self.select_objectivs]

        DA_now = copy.deepcopy(Objs_now)
        p = 1.0 / DA_now.shape[1]
        chooseFlag = ea.update_DA(DA_now, NUM, p)

        CA = self.CA + offspring
        CA_obj = copy.deepcopy(CA.ObjV)

        CA_obj = CA_obj[:, self.select_objectivs]
        chooseCA = ea.update_CA(CA_obj, self.CA_size)
        self.CA = CA[chooseCA]

        return population[chooseFlag], chooseFlag

    def update_passtime(self):
        self.passTime += time.time() - self.timeSlot

    def update_timeslot(self):
        self.timeSlot = time.time()

    def add_gen2metricinfo(self, population, gen):
        self.all_metric_valid[str(gen)] = population.pred_logits_valid
        self.all_metric_test[str(gen)] = population.pred_logits_test
        self.all_metric_train[str(gen)] = population.pred_logits_train

    def record_reduced_obj(self, base_res_path):
        gen_list = self.select_objectivs_record.keys()
        save_name = base_res_path+self.start_time
        for gen in gen_list:
            reduced_objs = self.select_objectivs_record[gen]
            np.savetxt(save_name+"/detect/select_objs_gen{}.txt".format(gen), reduced_objs)

    def record_RNCC(self, base_res_path):
        gen_list = self.RNCC_record.keys()
        save_name = base_res_path + self.start_time
        for gen in gen_list:
            RNCC = self.RNCC_record[gen]
            np.savetxt(save_name + "/detect/RNCC_gen{}.txt".format(gen), RNCC)

    def get_aveRNCC(self, gen):
        objs = None
        count = 0
        num = list(self.RNCC_record.keys())
        num = num[-int(min(gen, len(num))):]
        for i in num:
            if objs is None:
                # objs = copy.deepcopy(self.RNCC_record[i])
                objs =  copy.deepcopy(self.RNCC_record[i])
            else:
                objs +=  copy.deepcopy(self.RNCC_record[i])
            count += 1
        objs = objs/count
        return objs

    def run(self, prophetPop=None):
        base_res_path = '{}/{}/'.format(self.parameters["dirName"], self.problem.dataname)
        if not os.path.exists(base_res_path):
            os.makedirs(base_res_path)

        self.population.printPare(self.problem.test_org, self.record_parameter, base_res_path)

        population = self.population
        NIND = population.sizes
        self.problem.do_pre()
        self.initialization(is_ensemble=self.is_ensemble)
        population.initChrom(dataname=self.problem.dataname)
        self.update_passtime()
        population.save(dirName=base_res_path, Gen=-1)
        self.update_timeslot()
        gen = 1
        self.call_aimFunc(population, gen=gen, dirName=base_res_path)
        self.objs_record[str(1)] = copy.deepcopy(self.population.ObjV)

        CA_ID = np.arange(self.CA_size)
        self.CA = population[CA_ID].copy()
        self.update_passtime()
        self.add_gen2info(population, gen)
        self.add_gen2metricinfo(population, gen)
        self.update_timeslot()

        while True:

            gen += 1
            self.gen = gen
            if self.gen >= self.parameters["MAXGEN"]:
                break
            print()
            MOEA_sel_num = NIND

            self.objs_record[str(gen)] = copy.deepcopy(population.ObjV)
            self.RNCC = RNCC_fun(copy.deepcopy(population.ObjV))
            self.RNCC_record[str(gen)] = self.RNCC

            self.update_timeslot()
            Analysis_objs = copy.deepcopy(population.ObjV)
            aveRNCC = self.get_aveRNCC(self.recall_gen)
            se_objs, RNCC = Select_Ojbectives_cut(Analysis_objs, aveRNCC, self.parameters["throd"])
            self.select_objectivs = np.sort(se_objs)
            self.select_objectivs_record[str(gen)] = self.select_objectivs
                        
            if gen < self.recall_gen:
                self.select_objectivs = np.arange(26)
            
            Objs_now = copy.deepcopy(population.ObjV)
            Objs_now = Objs_now[:, self.select_objectivs]

            better_CA = np.random.randint(int(self.CA_size), size=int(np.ceil(self.CA_size)))
            better_CA = list(better_CA.astype(int))
            off_CA = self.CA[better_CA].copy()
            better_DA = np.random.randint(len(population), size=int(np.ceil(NIND)))
            better_DA = list(better_DA.astype(int))
            off_DA = population[better_DA].copy()
            offspring = off_CA + off_DA

            if self.crossover_p > 0:
                offspring.Chrom[0:np.int32(np.floor(MOEA_sel_num / 2) * 2)] = self.recOper.do(
                    offspring.Chrom[0:np.int32(np.floor(MOEA_sel_num / 2))],
                    offspring.Chrom[np.int32(np.floor(MOEA_sel_num / 2)):np.int32(np.floor(MOEA_sel_num / 2) * 2)],
                    np.random.uniform(0, 1, 1))
            offspring.Chrom = self.mutOper.do(offspring.Chrom)
            self.call_aimFunc(offspring, gen=gen, dirName=base_res_path, loss_type=-1)
            population, chooseidx = self.reinsertion(population, offspring, NIND, isNN=1)
            print('Gen', gen, '  Num: ', len(self.select_objectivs), " ", self.select_objectivs)
            self.update_passtime()
            self.add_gen2info(population, gen)
            self.add_gen2metricinfo(population, gen)
            self.update_timeslot()


        population.save(dirName=base_res_path + self.start_time, Gen=gen,
                        All_objs_train=self.all_objetives_train,
                        All_objs_valid=self.all_objetives_valid,
                        All_objs_test=self.all_objetives_test,
                        All_objs_ensemble=self.all_objetives_ensemble,
                        passtime=self.passTime)

        population.save_metrics(dirName=base_res_path + self.start_time, Gen=gen,
                        All_objs_train=self.all_metric_train,
                        All_objs_valid=self.all_metric_valid,
                        All_objs_test=self.all_metric_test,
                        All_objs_ensemble=self.all_objetives_ensemble)

        self.record_reduced_obj(base_res_path)
        self.record_RNCC(base_res_path)

        save_model(population, gen, filepath='nets/gen%d_net%s.pth', start_time=self.start_time,
                   base_res_path=base_res_path)

        print("Run ID ", self.run_id, "finished!")
        return self.finishing(population)
