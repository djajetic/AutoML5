# Linear enseble of best models
#Damir Jajetic, 2015


import libscores
import multiprocessing
import time
import shutil
import os
import numpy as np
import data_io
import psutil
import data_converter
import copy
from sklearn.utils import shuffle
from operator import itemgetter
from sklearn.pipeline import Pipeline
from scipy import stats
import itertools


class no_transform:
	def fit_transform(self, preds):
		return preds
			
def blend2(x1,x2,y, metric, task, x1valid, x2valid, x1test, x2test):
	try:
		mm = no_transform()
		mbest_score = -2
		for w1 in np.arange(0.2, 1, 0.1):
			w2 = 1- w1
			x = mm.fit_transform(x1)*w1  +  mm.fit_transform(x2)*w2
			exec('score = libscores.'+ metric  + '(y, x, "' + task + '")')
			try:
				if score <= 0:
					exec('CVscore_auc = libscores.auc_metric(y, x, "' + task + '")')
					score += CVscore_auc/10
			except:
				pass
			
			if score > mbest_score:
				mbest_score = score
				mbest_w1 = w1
				mbest_x  = x
		mbest_w2 = 1- mbest_w1
		xvalid = mm.fit_transform(x1valid) * mbest_w1 +  mm.fit_transform(x2valid)* mbest_w2
		xtest =  mm.fit_transform(x1test) * mbest_w1 +  mm.fit_transform(x2test) * mbest_w2

		return mbest_score, xvalid, xtest
	except:
		return 0.01, x1valid, x1test
	
def blend3(x1,x2, x3, y, metric, task, x1valid, x2valid, x3valid, x1test, x2test, x3test):
	try:
		mm = no_transform()
		mbest_score = -2
		for w1 in np.arange(0.2, 1, 0.2):
			for w2 in np.arange(0.1, 0.6, 0.2):
				w3 = 1- w1 - w2
				if w3 > 0:
					x = mm.fit_transform(x1)*w1  +  mm.fit_transform(x2)*w2 +  mm.fit_transform(x3)*w3
					exec('score = libscores.'+ metric  + '(y, x, "' + task + '")')
					try:
						if score <= 0:
							exec('CVscore_auc = libscores.auc_metric(y, x, "' + task + '")')
							score += CVscore_auc/10
					except:
						pass
					if score > mbest_score:
						mbest_score = score
						mbest_w1 = w1
						mbest_w2 = w2
		
		mbest_w3 = 1- mbest_w1- mbest_w2
		xvalid = mm.fit_transform(x1valid) * mbest_w1 +  mm.fit_transform(x2valid)* mbest_w2 +  mm.fit_transform(x3valid)* mbest_w3
		xtest =  mm.fit_transform(x1test) * mbest_w1 +  mm.fit_transform(x2test) * mbest_w2 +  mm.fit_transform(x3test) * mbest_w3

		return mbest_score, xvalid, xtest
	except:
		return 0.01, x1valid, x1test

def blender (sd, srd, srf, src, Nworkers, stop_writing, output_dir, basename, Lstart, Ltime_budget, train_split, test_split):
	try:

		cycle = 0 #cycle 0 is all zeros
		best_score = -2
		atbest = -2
		
		while(1):
			try:
				time.sleep(0.5)
				temp_workers_data = []
				workers_data = []
				for wr_no in range(Nworkers):
					exec("wr_data =  sd.worker"+str(wr_no))
					if wr_data['done'] > 0:
						temp_workers_data.append(wr_data)
				wgroups = [i['blend_group'] for i in temp_workers_data]
				for group in np.unique(wgroups):
					twdata = [i for i in temp_workers_data if i['blend_group'] == group]
					twdata = sorted(twdata, key=itemgetter('score'), reverse=True)
					

					workers_data.append(twdata[0])
					try:
						workers_data.append(twdata[1])
					except:
						pass
				
				workers_data_raw = []
				raw0_data =  srd.raw_model
				if raw0_data['done'] ==1:
					workers_data_raw.append(raw0_data)
					
				raw1_data =  srd.raw_model1
				if raw1_data['done'] ==1:
					workers_data_raw.append(raw1_data)
					
				raw2_data =  srd.raw_model2
				if raw2_data['done'] ==1:
					workers_data_raw.append(raw2_data)
					
				raw3_data =  srd.raw_model3
				if raw3_data['done'] ==1:
					workers_data_raw.append(raw3_data)
				
				raw4_data =  srd.raw_model4
				if raw4_data['done'] ==1:
					workers_data_raw.append(raw4_data)
				
					
				raw5_data =  srf.model1
				if raw5_data['done'] ==1:
					workers_data_raw.append(raw5_data)
				
				raw6_data =  src.model1
				if raw6_data['done'] ==1:
					workers_data_raw.append(raw6_data)

				
				if len(workers_data_raw) > 0:
				
					workers_data_raw = sorted(workers_data_raw, key=itemgetter('score'), reverse=True)
					workers_data.append(workers_data_raw[0])
					try:
						workers_data.append(workers_data_raw[1])
					except:
						pass
					try:
						workers_data.append(workers_data_raw[2])
					except:
						pass
					try:
						workers_data.append(workers_data_raw[3])
					except:
						pass
					try:
						workers_data.append(workers_data_raw[4])
					except:
						pass
					try:
						workers_data.append(workers_data_raw[5])
					except:
						pass
				
				workers_data = sorted(workers_data, key=itemgetter('score'), reverse=True)
				
				
				if len(workers_data) > 0:
					worker0 = workers_data[0]
					preds_valid = worker0['preds_valid'] 
					preds_test = worker0['preds_test'] 
					
					y = sd.yt_raw[test_split:]
					
					x = worker0['preds_2fld']
					
					exec('s0 = libscores.'+ sd.LD.info['metric']  + '(y, x, "' + sd.LD.info['task'] + '")')
					try:
						if sd.LD.info['task'] != 'regression' and s0 <= 0:
							exec('CVscore_auc = libscores.auc_metric(sd.yt_raw[test_split:], preds, "' + sd.LD.info['task'] + '")')
							s0 += CVscore_auc/10
					except:
						pass
					best_score = s0
					
					try:
						if s0 > atbest:
							atbest = best_score
							if sd.LD.info['target_num']  == 1:
								if sd.LD.info['task'] != 'regression':
									preds_valid = preds_valid[:,1]
									preds_test = preds_test[:,1]
							if sd.LD.info['task'] != 'regression':
								preds_valid = np.clip(preds_valid,0,1)
								preds_test = np.clip(preds_test,0,1)
							filename_valid = basename + '_valid_' + str(cycle).zfill(3) + '.predict'
							data_io.write(os.path.join(output_dir,filename_valid), preds_valid)
							filename_test = basename + '_test_' + str(cycle).zfill(3) + '.predict'
							data_io.write(os.path.join(output_dir,filename_test), preds_test)
					except:
						pass
						
					
					Lsample = 4
					Lssample = Lsample - 1
					
					for iter_worker in itertools.combinations(workers_data[:Lsample], 2):
						worker0 = iter_worker[0]
						worker1 = iter_worker[1]
						s01, validt, testt = blend2(worker0['preds_2fld'],worker1['preds_2fld'],y, sd.LD.info['metric'] , sd.LD.info['task'],
										    worker0['preds_valid'], worker1['preds_valid'], 
										    worker0['preds_test'], worker1['preds_test'])
					
						if s01 > best_score:
							best_score = s01
							preds_valid = validt
							preds_test = testt
					
					for iter_worker in itertools.combinations(workers_data[:Lssample], 3):
						worker0 = iter_worker[0]
						worker1 = iter_worker[1]
						worker2 = iter_worker[2]
						s012, validt, testt = blend3(worker0['preds_2fld'],worker1['preds_2fld'],worker2['preds_2fld'],y, sd.LD.info['metric'] ,  sd.LD.info['task'],
										    worker0['preds_valid'], worker1['preds_valid'], worker2['preds_valid'], 
										    worker0['preds_test'], worker1['preds_test'], worker2['preds_test'])
						if s012 > best_score:
							best_score = s012
							preds_valid = validt
							preds_test = testt
					
					print "blend", atbest
					if stop_writing.is_set() == False and best_score > atbest:
						atbest = best_score
						
						if  sd.LD.info['target_num']  == 1:
							if sd.LD.info['task'] != 'regression':
								preds_valid = preds_valid[:,1]
								preds_test = preds_test[:,1]
						if sd.LD.info['task'] != 'regression':	
							preds_valid = np.clip(preds_valid,0,1)
							preds_test = np.clip(preds_test,0,1)
						filename_valid = basename + '_valid_' + str(cycle).zfill(3) + '.predict'
						data_io.write(os.path.join(output_dir,filename_valid), preds_valid)
						filename_test = basename + '_test_' + str(cycle).zfill(3) + '.predict'
						data_io.write(os.path.join(output_dir,filename_test), preds_test)
					
						#cycle += 1 
					
			except Exception as e:
				print 'exception in blender process' + '     ' +  str(e)
				# in case of any problem, let's try again
	except Exception as e:
				print 'exception in blender main process' + '     ' +  str(e)
