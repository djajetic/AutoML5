#Damir Jajetic, 2015
from sklearn import *
import libscores
import time
import psutil
import numpy as np
				
def worker (sd, srd, Lstart, Ltime_budget, train_split, test_split):
	try:
	
		old_score = -1
		CVscore = -1
		
		time.sleep(2)
		
		if psutil.virtual_memory()[2] > 50:
			Lnj = 1
		else:
			Lnj = 2
		
		from sklearn.pipeline import Pipeline
		from sklearn import feature_selection
		if sd.LD.info['task'] != 'regression':			
			model = Pipeline([("pre", feature_selection.VarianceThreshold(threshold=20) ),("rf", ensemble.RandomForestClassifier(n_estimators=1,  max_depth=5, random_state=1, n_jobs=Lnj, warm_start=False))])
		else:
			model = Pipeline([("pre", feature_selection.VarianceThreshold(threshold=20) ),("rf", ensemble.RandomForestRegressor(n_estimators=1,  max_depth=5, random_state=1,  n_jobs=Lnj, warm_start=False))])
		Lind_fs = 1
		for Lmd in range(20, 1000, 40):
			try:
				
				Lthreshold = model.get_params()['pre__threshold']
				while(Lind_fs==1):
					try:
						if Lthreshold < 1:
							Lthreshold = 0
							model.set_params(pre__threshold=Lthreshold)
							model.fit(sd.LD.data['X_train'][:test_split], sd.LD.data['Y_train'][:test_split])
							model.set_params(rf__warm_start=True)
							Lind_fs=0
						else:
							Lthreshold *= 0.5
						fs_temp = feature_selection.VarianceThreshold(threshold=Lthreshold) 
						fs_temp.fit(sd.LD.data['X_train'][:test_split])
						x_temp = fs_temp.transform(sd.LD.data['X_train'][:test_split])
						break
					except Exception as e:
						pass
						
				
				model.set_params(rf__n_estimators=Lmd)
				model.set_params(pre__threshold=Lthreshold)
					
				if (time.time() - Lstart) / Ltime_budget > 0.9:
					break	
				if psutil.virtual_memory()[2] > 80:
					time.sleep(2)
				if psutil.virtual_memory()[2] > 80:
					break
			
				model.fit(sd.LD.data['X_train'][:test_split], sd.LD.data['Y_train'][:test_split])

				if psutil.virtual_memory()[2] > 80:
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
						CVscore_auc += CVscore_auc/10
				except:
					pass
				
				if CVscore > old_score:
					
					old_score = CVscore

					if psutil.virtual_memory()[2] > 80:
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

					wd =  srd.model1
					wd['preds_valid'] = preds_valid
					wd['preds_test'] = preds_test
					wd['preds_2fld'] = preds
					wd['score'] = CVscore 
					wd['done'] = 1
					srd.model1 = wd
					
					Lno_ngain = 1
				else:
					try:
						if Lno_ngain > 1 and Lind_fs==1:
							Lthreshold *=4 
							Lind_fs=0
							model.set_params(pre__threshold=Lthreshold)
							model.fit(sd.LD.data['X_train'][:test_split], sd.LD.data['Y_train'][:test_split])
							model.set_params(rf__warm_start=True)
						Lno_ngain += 1
					except:
						Lno_ngain = 1
					if Lno_ngain == 10:
						break
			except Exception as e:
				print e
				pass
	except Exception as e:
			print 'exception in RF worker ' + '     ' +  str(e)


 