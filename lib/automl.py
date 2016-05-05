#Damir Jajetic, 2015

from sklearn.externals import joblib
from sklearn import *
from sklearn.utils import shuffle
import libscores
import multiprocessing
import time
import os
import numpy as np
import data_io
import psutil
import data_converter
import automl_worker
import automl_models
import automl_blender
import automl_sb
import automl_rf
import automl_gb

def baseline(output_dir, basename, valid_num, test_num, target_num):
	preds_valid = np.zeros([valid_num , target_num])
	preds_test = np.zeros([test_num , target_num])
	
	cycle = 0
	filename_valid = basename + '_valid_' + str(cycle).zfill(3) + '.predict'
	data_io.write(os.path.join(output_dir,filename_valid), preds_valid)
	filename_test = basename + '_test_' + str(cycle).zfill(3) + '.predict'
	data_io.write(os.path.join(output_dir,filename_test), preds_test)

def predict(LD, Loutput_dir, Lstart, Ltime_budget, Lbasename, running_on_codalab):
	try:	

		train_split = int(len(LD.data['Y_train'])*0.6)
		test_split = int(len(LD.data['Y_train'])*0.8)
		
		baseline(Loutput_dir, Lbasename, LD.info['valid_num'], LD.info['test_num'], LD.info['target_num'])		
	
		LD.data['X_train'], LD.data['Y_train'] = shuffle(LD.data['X_train'], LD.data['Y_train'] , random_state=1)
		
		try:
			if LD.info['task'] != 'regression':
				for Le in range(10): #stratiffied split consume more time
					if len(np.unique(LD.data['Y_train'][:train_split])) != len(np.unique(LD.data['Y_train'][test_split:])):
						try:
							LD.data['X_train'], LD.data['Y_train'] = shuffle(LD.data['X_train'], LD.data['Y_train'] , random_state=1)
						except:
							pass
		except:
			pass
		
		if LD.info['task'] != 'regression':
			try: 
				yt_raw = np.array(data_converter.convert_to_bin(LD.data['Y_train'], len(np.unique(LD.data['Y_train'])), False))
			except:
				yt_raw = LD.data['Y_train']
		else:
				yt_raw = LD.data['Y_train']	
		
		#Strategy is that we will have N workers that will try prediction models listed in separate file (will be described later)
		# in shared data they will push CV score, and predictions for train, valid and test data
		# separate blender worker will use this data to create linear ensemble.
		
		
		# regardless of strategy, for competition purposes, it is good to have work in separate process, that can easily be killed
		# there are 2 events visible to all workers
		# a) stop writing - just to be sure that we don't kill process in the middle of writing predictions
		# b) start_growing - after this point, stop searching for best parameters, just build new trees or other similar strategy
		
		
		stop_writing_event = multiprocessing.Event()
		exploit_event = multiprocessing.Event()
		
		manager = multiprocessing.Manager()
		shared_data = manager.Namespace()
		shared_data.LD = LD
		shared_data.yt_raw =yt_raw


		#these are 3 dedicated workers
		# 1. fast predictors with increasing data sample
		# 2. ranomized decision trees predictor with increasing number of predictors (warm start)
		# 3. boosting predictor with increasing number of predictors (warm start)
		
		#1.
		try:
			shared_sb_data = manager.Namespace()
			shared_sb_data.raw_model = {"done":0, "score":0, "preds_valid":None, "preds_test":None, "preds_2fld":None}
			shared_sb_data.raw_model1 = {"done":0, "score":0, "preds_valid":None, "preds_test":None, "preds_2fld":None}
			shared_sb_data.raw_model2 = {"done":0, "score":0, "preds_valid":None, "preds_test":None, "preds_2fld":None}
			shared_sb_data.raw_model3 = {"done":0, "score":0, "preds_valid":None, "preds_test":None, "preds_2fld":None}
			shared_sb_data.raw_model4 = {"done":0, "score":0, "preds_valid":None, "preds_test":None, "preds_2fld":None}
					
			sb_process = multiprocessing.Process(target=automl_sb.worker, args=([ shared_data, shared_sb_data, Lstart, Ltime_budget, train_split, test_split]))
			sb_process.start()
		except:
			pass
		
		#2.
		try:
			shared_rf_data = manager.Namespace()
			shared_rf_data.model1 = {"done":0, "score":0, "preds_valid":None, "preds_test":None, "preds_2fld":None}
			
			rf_process = multiprocessing.Process(target=automl_rf.worker, args=([ shared_data, shared_rf_data, Lstart, Ltime_budget, train_split, test_split]))
			rf_process.start()
		except Exception as e:
			print e

		
		#3.
		try:
			shared_gb_data = manager.Namespace()
			shared_gb_data.model1 = {"done":0, "score":0, "preds_valid":None, "preds_test":None, "preds_2fld":None}
			
			gb_process= multiprocessing.Process(target=automl_gb.worker, args=([ shared_data, shared_gb_data, Lstart, Ltime_budget, train_split, test_split]))
			gb_process.start()
		except Exception as e:
			print e
		
		#This is main part of strategy
		# We will create namespace for sharing data between workers
		shared_data = manager.Namespace()
		shared_data.LD = LD
		shared_data.yt_raw =yt_raw
		
		#In automl_models.py should be listed all models (sklearn format only) with addtional properties
		# model -  model instance format - see examples
		# blend_group - we will linear ensemble N best models, but don't want best of similar ones.
		#			  to avoid this from same group only best one will be ensembled
		# getter - updater - setter - after signal, getter is function that will read "interesting" parameter
		#                                        from model, updater will update it, and setter will change that parameter. This will repeat until end.
		#                                        example is "number of estimators  get 80, update 80+20, push 100, next time read 100, update 120 push 120..."
		# generator - parameters that will be changed in model in every iteration. 
		
		models = automl_models.get_models(shared_data)
		models_count = len(models)
		
		# We will create semaphore for workers
		if psutil.virtual_memory()[2] > 50:
			Lncpu = 1
		elif psutil.virtual_memory()[2] > 40:
			Lncpu = 2
		else:
			Lncpu = 6
		
		semaphore = multiprocessing.Semaphore(Lncpu)
		try:
			#Creating N workers
			for Lnum in range(models_count): 
				exec("shared_data.worker"+str(Lnum) + ' = {"done":0, "score":0, ' +
					 '"preds_valid": None, "preds_test": None, ' +
					 '"model":' + models[Lnum] ["model"]+ ', ' +
					  '"blend_group": "%s", ' +  
					 '"getter": "%s", ' +  
					 '"updater": "%s", ' +  
					 '"setter": "%s", ' +  
					 '"generator": "%s" ' +  
					 '}')  % (models[Lnum] ['blend_group'],models[Lnum] ['getter'], models[Lnum] ['updater'], models[Lnum] ['setter'], models[Lnum] ['generator'])
			
			
			workers = [multiprocessing.Process(target=automl_worker.worker, args=([tr_no, shared_data, exploit_event, semaphore, train_split, test_split])) for tr_no in range(models_count)]
			for wr in workers:
				wr.start()
		except Exception as e:
			print e
		
		try:
			Lnw = Lnum
		except:
			Lnw = 0
		
		blender_process = multiprocessing.Process(target=automl_blender.blender, args=([ shared_data, shared_sb_data, shared_rf_data, shared_gb_data,  Lnw,  stop_writing_event, Loutput_dir, Lbasename, Lstart, Ltime_budget, train_split, test_split]))
		blender_process.start()
		
		try:
			explore_time =  max(Ltime_budget - (time.time() - Lstart) - 60,0)
		
			time.sleep (explore_time)
			exploit_event.set()
		except:
			pass
		
		while( (Ltime_budget - 40) >  (time.time() - Lstart)):
			time.sleep (1) 

		print "Stop signal sent", time.ctime(),  "time left",  Ltime_budget - (time.time() - Lstart)
		stop_writing_event.set()
		time.sleep (8)
		
		try:
			for wr in workers:
				try:
					wr.terminate()
				except:
					pass
		except:
			pass
		
		try:
			sb_process.terminate()
		except:
			pass
			
		try:
			rf_process.terminate()
		except:
			pass


		try:
			gb_process.terminate()
		except:
			pass

		try:
			blender_process.terminate()
		except:
			pass
			
		print "Done", time.ctime(),  "time left",  Ltime_budget - (time.time() - Lstart)
	except Exception as e:
		print "exception in automl_automl", time.ctime(),  "left=",  Ltime_budget - (time.time() - Lstart), str(e)
		
		try:
			for wr in workers:
				try:
					wr.terminate()
				except:
					pass
		except:
			pass
		
		try:
			sb_process.terminate()
		except:
			pass

		try:
			rf_process.terminate()
		except:
			pass

		try:
			gb_process.terminate()
		except:
			pass

		try:
			blender_process.terminate()
		except:
			pass