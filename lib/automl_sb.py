#Damir Jajetic, 2015

from sklearn import *
import libscores
import time
import psutil
import numpy as np
				
def worker (sd, srd, Lstart, Ltime_budget, train_split, test_split):
	try:
		
		start_step = int(round(train_split/10 /10)*10+1)
		train_len_step = int(round(train_split/6/10)*10+1)
	
		Lnum = -1
		
		if sd.LD.info['task'] != 'regression':
			if sd.LD.info['task'] == 'multilabel.classification':
				models =  [ensemble.RandomForestClassifier(n_estimators=10,  min_samples_split=8, max_depth=3, random_state=1),
					ensemble.RandomForestClassifier(n_estimators=2,  max_depth=2, random_state=1),
					ensemble.RandomForestClassifier(n_estimators=6, min_samples_split=10, random_state=1),
					ensemble.RandomForestClassifier(n_estimators=10,  max_depth=3, min_samples_split=4, random_state=1),
					ensemble.RandomForestClassifier(n_estimators=10,  max_depth=3, random_state=1),
					]
			else:
				models = [ensemble.RandomForestClassifier(n_estimators=20,  max_depth=3, random_state=1),
					ensemble.RandomForestClassifier(n_estimators=2,  max_depth=2, min_samples_split=4, random_state=1),
					ensemble.RandomForestClassifier(n_estimators=2,  min_samples_split=8, random_state=1),
					naive_bayes.MultinomialNB(),
					ensemble.RandomForestClassifier(n_estimators=10,  max_depth=2, random_state=1)
					]
		else:
			models = [linear_model.Ridge (alpha = 0.5),
					ensemble.RandomForestRegressor(n_estimators=2,  max_depth=2, random_state=1),
					linear_model.LinearRegression(),
					ensemble.RandomForestRegressor(n_estimators=10,  max_depth=3, random_state=1),
					ensemble.GradientBoostingRegressor(n_estimators=8, max_depth=3, random_state=1)
					]
					
		for train_lenx in range(start_step, test_split*2, train_len_step):
			try:
				if train_lenx > test_split:
					train_len = test_split
				else:
					train_len = train_lenx
				
				Lnum = train_len%5
				model = models[Lnum]
					
				if (time.time() - Lstart) / Ltime_budget > 0.95:
					break
				if psutil.virtual_memory()[2] > 80:
					time.sleep(2)
				if psutil.virtual_memory()[2] > 80:
					break
			
				model.fit(sd.LD.data['X_train'][:train_len], sd.LD.data['Y_train'][:train_len])
				
				if psutil.virtual_memory()[2] > 90:
					break
				
				try:
					preds = model.predict_proba(sd.LD.data['X_train'][test_split:])
				except:
					preds = model.predict(sd.LD.data['X_train'][test_split:])
					
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
				
					
				if psutil.virtual_memory()[2] > 90:
						break
				if psutil.virtual_memory()[2] > 90:
						break
						
				try:
					preds_valid = model.predict_proba(sd.LD.data['X_valid'])
				except:
					preds_valid = model.predict(sd.LD.data['X_valid'])
				try:
					preds_test = model.predict_proba(sd.LD.data['X_test'])
				except:
					preds_test = model.predict(sd.LD.data['X_test'])

				if sd.LD.info['task'] == 'multilabel.classification':		
					try:
						preds_valid = np.array(preds_valid)
						preds_valid = preds_valid[:, :, 1]
						preds_valid = preds_valid.T
					except:
						pass
					
					try:
						preds_test = np.array(preds_test)
						preds_test = preds_test[:, :, 1]
						preds_test = preds_test.T
					except:
						pass	


				if Lnum == 0:
					wd =  srd.raw_model
					wd['preds_valid'] = preds_valid
					wd['preds_test'] = preds_test
					wd['preds_2fld'] = preds
					wd['score'] = CVscore
					wd['done'] = 1
					srd.raw_model = wd					
				if Lnum == 1:
					wd1 =  srd.raw_model1
					wd1['preds_valid'] = preds_valid
					wd1['preds_test'] = preds_test
					wd1['preds_2fld'] = preds
					wd1['score'] = CVscore
					wd1['done'] = 1
					srd.raw_model1 = wd1
				if Lnum == 2:
					wd2 =  srd.raw_model2
					wd2['preds_valid'] = preds_valid
					wd2['preds_test'] = preds_test
					wd2['preds_2fld'] = preds
					wd2['score'] = CVscore
					wd2['done'] = 1
					srd.raw_model2 = wd2
				if Lnum == 3:
					wd3 =  srd.raw_model3
					wd3['preds_valid'] = preds_valid
					wd3['preds_test'] = preds_test
					wd3['preds_2fld'] = preds
					wd3['score'] = CVscore
					wd3['done'] = 1
					srd.raw_model3 = wd3
				if Lnum == 4:
					wd4 =  srd.raw_model4
					wd4['preds_valid'] = preds_valid
					wd4['preds_test'] = preds_test
					wd4['preds_2fld'] = preds
					wd4['score'] = CVscore
					wd4['done'] = 1
					srd.raw_model4 = wd4
			except  Exception as e:
				print 'exception in sb loop in ' + '     ' +  str(e)
	except Exception as e:
			print 'exception in sb worker ' + '     ' +  str(e)

 