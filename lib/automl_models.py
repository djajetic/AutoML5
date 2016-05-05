# List of models that will be tested
# Damir Jajetic, 2015
def get_models(sd):

	#this models that will be cross validated on test data
	if sd.LD.info['task'] == 'regression':
		models = [
			{"model": 'linear_model.LinearRegression()',
			  "blend_group":   "LR",  
			   "getter":   "Lalpha = model_last.get_params()['alpha']",
			    "updater":   "tries_left = 0",
			     "setter":   "return in updater will end process",
			     "generator":  "alpha=0.01 @@ " \
						  "alpha=0.1 @@ " \
						  "alpha=0.4 @@ " \
						  "alpha=1.0 @@ " \
						  "alpha=0.001 @@ " \
						  "alpha=8.0 @@ " 
			},
			{"model": 'linear_model.Ridge(alpha=2.0)',
			  "blend_group":   "LR",  
			  "getter":   "Lalpha = model_last.get_params()['alpha']",
			    "updater":   "tries_left = 0",
			     "setter":   "return in updater will end process",
			     "generator":  "alpha=0.01 @@ " \
						  "alpha=0.1 @@ " \
						  "alpha=0.4 @@ " \
						  "alpha=1.0 @@ " \
						  "alpha=0.001 @@ " \
						  "alpha=8.0 @@ "
			},
			
			
			{"model": 'ensemble.RandomForestRegressor(n_estimators=40, random_state=Lnum, n_jobs=1)',
			"blend_group":   "RDT3",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 20",
			    "setter":   "n_estimators = Lestimators",
			    "generator": "max_depth=4 @@" \
						"max_depth=8 @@" \
						"max_depth=16 @@"
			},
			{"model": 'ensemble.GradientBoostingRegressor(n_estimators=40, random_state=Lnum)',
			"blend_group":   "RDT3",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 20",
			    "setter":   "n_estimators = Lestimators",
			    "generator": "max_depth=4 @@" \
						"max_depth=8 @@" \
						"max_depth=16 @@" 
			},
			{"model": 'neighbors.KNeighborsRegressor(n_neighbors=2)',
			"blend_group":   "KNN",  
			    "getter":   "Lalpha = model_last.get_params()['n_neighbors']",
			    "updater":   "tries_left = 0",
			     "setter":   "return in updater will end process",
			     "generator":  "n_neighbors=1 @@ " \
						  "n_neighbors=2 @@ " \
						  "n_neighbors=4 @@ " 
			}
		]
		return models
	if sd.LD.info['is_sparse'] == 1:
		models = [
			{"model": 'naive_bayes.BernoulliNB(alpha=0.1)',
			  "blend_group":   "NB",  
			    "getter":   "Lalpha = model_last.get_params()['alpha']",
			    "updater":   "tries_left = 0",
			     "setter":   "return in updater will end process",
			     "generator":  "alpha=0.01 @@ " \
						  "alpha=0.1 @@ " \
						  "alpha=0.4 @@ " \
						  "alpha=1.0 @@ " \
						  "alpha=0.001 @@ " \
						  "alpha=8.0 @@ " 
			},
			{"model": 'naive_bayes.MultinomialNB(alpha=0.1)',
			  "blend_group":   "NB",  
			    "getter":   "Lalpha = model_last.get_params()['alpha']",
			    "updater":   "tries_left = 0",
			     "setter":   "return in updater will end process",
			     "generator":  "alpha=0.01 @@ " \
						  "alpha=0.1 @@ " \
						  "alpha=0.4 @@ " \
						  "alpha=1.0 @@ " \
						  "alpha=0.001 @@ " \
						  "alpha=8.0 @@ " 
			},
			{"model": 'ensemble.RandomForestClassifier(n_estimators=40, random_state=Lnum, n_jobs=1)',
			"blend_group":   "RDT1",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 20",
			    "setter":   "n_estimators = Lestimators, warm_start = False",
			    "generator": 
						"max_depth=4 @@" \
						"max_depth=8 @@" \
						"max_depth=16 @@" \
						"min_samples_split=8 @@" 
			},
			{"model": 'linear_model.LogisticRegression(C=0.1, random_state=Lnum)',
			    "blend_group":   "LC",  
			    "getter":   "Lalpha = model_last.get_params()['alpha']",
			    "updater":   "tries_left = 0",
			     "setter":   "return in updater will end process",
			     "generator":  "C=0.01 @@ " \
						  "C=0.1 @@ " \
						  "C=4.0 @@ " \
						  "C=1.0 @@ " \
						  "C=0.001 @@ " \
						  "C=8.0 @@ " 
			},
			{"model": 'neighbors.KNeighborsClassifier(n_neighbors=2)',
			"blend_group":   "KNN",  
			    "getter":   "Lalpha = model_last.get_params()['n_neighbors']",
			    "updater":   "tries_left = 0",
			     "setter":   "return in updater will end process",
			     "generator":  "n_neighbors=1 @@ " \
						  "n_neighbors=2 @@ "  \
						  "n_neighbors=4 @@ " 
			}
			]
		return models
	else:
		models = [
			{"model": 'linear_model.LogisticRegression(random_state=1)',
			"blend_group":   "LC",  
			    "getter":   "Lestimators = model_last.get_params()['penalty']",
			    "updater":   "tries_left = 0",
			     "setter":   "return in updater will end process",
			     "generator":  "penalty='l2', dual=False, C=1.0 @@ " \
						  "penalty='l1', dual=False, C=1.0 @@ " \
						  "penalty='l2', dual=False, C=2.0 @@ " \
						  "penalty='l2', dual=False, C=0.5 @@ " \
						  "penalty='l2', dual=True, C=1.0 @@ " \
						  "penalty='l2', dual=True, C=0.5 @@ " 
			},
			{"model": 'naive_bayes.GaussianNB()',
			"blend_group":   "NB",  
			    "getter":   "Lestimators = model_last.get_params()",
			    "updater":   "tries_left = 0",
			     "setter":   "",
			     "generator":  "" 
			},
			{"model": 'ensemble.RandomForestClassifier(n_estimators=40, random_state=Lnum, n_jobs=1)',
			"blend_group":   "RDT1",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 20",
			    "setter":   "n_estimators = Lestimators, warm_start = False",
			    "generator": 
						"max_depth=4 @@" \
						"max_depth=8 @@" \
						"max_depth=16 @@" \
						"min_samples_split=8 @@" 
			},
			{"model": 'ensemble.GradientBoostingClassifier(n_estimators=40, random_state=Lnum)',
			"blend_group":   "GB",  
			    "getter":   "Lestimators = model_last.get_params()['n_estimators']",
			    "updater":   "Lestimators += 10",
			     "setter":   "n_estimators = Lestimators , warm_start = False",
			     "generator": ""
			},
			{"model": 'neighbors.KNeighborsClassifier(n_neighbors=2)',
			"blend_group":   "KNN",  
			    "getter":   "Lalpha = model_last.get_params()['n_neighbors']",
			    "updater":   "tries_left = 0",
			     "setter":   "return in updater will end process",
			     "generator":  "n_neighbors=1 @@ " \
						  "n_neighbors=2 @@ "  \
						  "n_neighbors=4 @@ " 
			}
			]
		return models	
	return models
