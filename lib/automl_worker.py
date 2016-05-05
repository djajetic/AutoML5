#Damir Jajetic, 2015
import copy
from sklearn import *
import libscores
import time
import numpy as np
import data_io
import psutil
import data_converter
import copy

def worker (Lnum, sd, exploit, semaphore, train_split, test_split):
 try:	 
	# this generator will yield parameters that model should try
	# parameters are defined in automl_models.py
	def f_parameter_generator(parameters_list):
		for parameters in parameters_list:
			yield parameters
		yield "done"
	
	time.sleep(Lnum+4)
	
	semaphore.acquire()

	exec("wd =  sd.worker"+str(Lnum))
	
	#Sometimes we will need to roll back to last or best model
	#this will increase memmory consumptions, pickle creation should be considered as alternative
	model = wd['model']
	model_all = copy.deepcopy(wd['model']) 
	model_last = copy.deepcopy(wd['model']) 
	model_best = copy.deepcopy(wd['model']) 
	best_CVscore = 0
	
	gen_list = wd['generator'].split('@@')
	parameter_generator = f_parameter_generator(gen_list)
		
	use_generator = 1
	
	tries_left = 10
	while (tries_left > 0):

		if psutil.virtual_memory()[2] > 60:
			time.sleep(2)
			continue 
	
		semaphore.acquire()
		
		try:
			if exploit.is_set() == True:
				use_generator = 0
			if use_generator == 1:
				setter = parameter_generator.next()
				if setter == "done":
					use_generator = 0

					model_last = copy.deepcopy(model_best)
					model = copy.deepcopy(model_best)
					model_all = copy.deepcopy(model_best)
					
					exec(wd['getter'])
					exec(wd['updater'])
					setter = wd['setter']
			else:
				model = copy.deepcopy(model_best) 
				exec(wd['getter'])
				exec(wd['updater'])
				setter = wd['setter']


			if psutil.virtual_memory()[2] > 60:
				time.sleep(1)
			if psutil.virtual_memory()[2] > 65:
				destroy = this_worker  #will go to exception	
				
			exec("model.set_params(" + setter + ")")
			
			
			model.fit(sd.LD.data['X_train'][:train_split], sd.LD.data['Y_train'][:train_split])
			
			if psutil.virtual_memory()[2] > 60:
				time.sleep(1)
			if psutil.virtual_memory()[2] > 65:
				destroy = this_worker
			
			try:
				preds = model.predict_proba(sd.LD.data['X_train'][test_split:])
			except:
				preds = model.predict(sd.LD.data['X_train'][test_split:])
			
			if psutil.virtual_memory()[2] > 60:
				time.sleep(4)
			if psutil.virtual_memory()[2] > 65:
				destroy = this_worker

			if sd.LD.info['task'] == 'multilabel.classification':
				try:
					preds = np.array(preds)
					preds = preds[:, :, 1]
					preds = preds.T
				except:
					pass

			exec('CVscore = libscores.'+ sd.LD.info['metric']  + '(sd.yt_raw[test_split:], preds, "' + sd.LD.info['task'] + '")')
			try:
				if sd.LD.info['task'] != 'regression' and CVscore <= 0:
					exec('CVscore_auc = libscores.auc_metric(sd.yt_raw[test_split:], preds, "' + sd.LD.info['task'] + '")')
					CVscore += CVscore_auc/10
			except:
				pass
						
			model_last = copy.deepcopy(model) 
			
			if CVscore > best_CVscore:
				if psutil.virtual_memory()[2] > 60:
					time.sleep(1)				
				if psutil.virtual_memory()[2] > 65:
					destroy = this_worker

				model_best = copy.deepcopy(model)
				
				exec("model_all.set_params(" + setter + ")") #because of warm start
				
				if psutil.virtual_memory()[2] > 60:
					time.sleep(1)				
				if psutil.virtual_memory()[2] > 65:
					destroy = this_worker
					
					
				time.sleep(0.05)
				best_CVscore = CVscore
			
				model_all.fit(sd.LD.data['X_train'], sd.LD.data['Y_train']) 
				
				if psutil.virtual_memory()[2] > 60:
					time.sleep(4)				
				if psutil.virtual_memory()[2] > 65:
					destroy = this_worker
				
				try:
					preds_valid = model_all.predict_proba(sd.LD.data['X_valid'])
				except:
					preds_valid = model_all.predict(sd.LD.data['X_valid'])
				try:
					preds_test = model_all.predict_proba(sd.LD.data['X_test'])
				except:
					preds_test = model_all.predict(sd.LD.data['X_test'])
				
				
				if sd.LD.info['task'] == 'multilabel.classification':
					try:
						preds_valid = np.array(preds_valid)
						preds_valid = preds_valid[:, :, 1]
						preds_valid = preds_valid.T
						
						preds_test = np.array(preds_test)
						preds_test = preds_test[:, :, 1]
						preds_test = preds_test.T
					except:
						pass

				wd['score'] = CVscore				
				wd['preds_2fld'] = preds
							
				wd['preds_valid'] = preds_valid
				wd['preds_test'] = preds_test
				
				if wd['done'] == 0: wd['done'] = 1
				exec("sd.worker"+str(Lnum) + " =  wd")
				
				if use_generator == 0:
					tries_left = 10
			else:
				if use_generator == 0:
					tries_left -= 1  #try 10 times after  best model
			
			semaphore.release()
		except Exception as e:
			try:
				print 'exception in worker ' + '     ' +  str(e)
				print "error model = ", wd['model']
			except:
				pass
			try:
				semaphore.release()
				break
			except:
				pass

	exec("wd =  sd.worker"+str(Lnum))
	if wd['done'] == 1: wd['done'] = 2
	exec("sd.worker"+str(Lnum) + " =  wd")
 except Exception as e:
	print 'out exception in worker ' + '     ' +  str(e)
	try:
		print "error model = ", wd['model']
	except:
		pass
	try:
		
		semaphore.release()
	except:
		pass
