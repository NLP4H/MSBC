{"baselines": {
	"data_dir_train": "master_path/data/neurology_notes/processed_data/Final Splits/train_data.csv",
	"data_dir_valid": "master_path/neurology_notes/processed_data/Final Splits/test_data.csv",
	"results_dir": "master_pathdon/repo/ML4H_MSProject/data/baseline_data/",
	"logreg": {
		"penalty": "l1",
		"solver": "saga",
		"C": 10
		// "class_weight": "balanced"
		// "tol": 1e-8
	},
	"multinomial_nb":{
		"alpha": 1.0
	},
	"complement_nb":{
		"alpha": 1.0
	},
	"svm_rbf": {
		"kernel": "rbf",
		"C": 0.1,
		"gamma": "scale",
		"class_weight": "balanced"
	},
	"svm_polynomial": {
		"kernel": "polynomial",
		"C":  0.1,
		"gamma": "scale",
		"class_weight": "balanced",
		"coef0": 1
	},
	"linear_svc":{
		"penalty": "l1",
		"max_iter": 10000,
		"class_weight": "balanced",
		"dual": false
	},
	"svm_sigmoid": {
		"kernel": "sigmoid",
		"gamma": "scale",
		"coef0": 1,
		"class_weight": "balanced"
	},
	"lda": {
	},
	"qda":{
		"reg_param": 1.0,
		"store_covariance": false
	}
}}