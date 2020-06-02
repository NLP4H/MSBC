"""
Runs predict_cnn, predict_rb, predict_sequential, predict_tfidf for specified label_type 
to obtain label prediction results for model combinations such as:

predict_<label_type>_word2vec_cnn
predict_<label_type>_log_reg_baseline
predict_<label_type>_linear_svc
predict_<label_type>_svc_rbf
predict_<label_type>_lda
predict_<label_type>_rb_baseline
predict_<label_type>_rb_allen_cnn
predict_<label_type>_rb_word2vec_cnn
predict_<label_type>_snorkel_rb_allen_cnn_<snorkel_model_type>
predict_<label_type>_snorkel_rb_word2vec_cnn_<snorkel_model_type>
predict_<label_type>_snorkel_rb_tfidf_word2vec_cnn_<snorkel_model_type>
predict_<label_type>_snorkel_rb_tfidf_allen_cnn_<snorkel_model_type>

note: predict_<label_type>_allen_cnn --> generated in another file

"""
import predict_cnn
from predict_cnn import predict_cnn
# import predict_rb
# from predict_rb import predict_rb 
# import predict_tfidf 
# from predict_tfidf import predict_tfidf
import predict_sequential 
from predict_sequential import predict_sequential

def run_results(label_type, tfidf_models, model_types, train_path, val_path, test_path, unlabeled_path, results_paths, snorkel_best_eval):

    # Obtain result path for label_type
    result_path = results_paths[label_type]
    tfidf_predict_col = []

    # for each tfidf models
    for model_type in tfidf_models:
        # run log_reg_baseline, linear_svc, svc_rbf or lda results
        if model_type in ["lda", "log_reg_baseline", "svc_rbf", "linear_svc"]:
            tfidf_predict_col.append("predict_" + str(label_type) + "_" + model_type)
            predict_tfidf(label_type, df_train_path, df_val_path, df_test_path, df_unlabeled_path, model_type)

    # For each model_type to test
    for model_type in model_types:
        # run word2vec_cnn results
        if model_type == "word2vec_cnn":
            predict_cnn(label_type, train_path,val_path,test_path,unlabeled_path)

        # run rb_baseline results
        elif model_type == "_rb_baseline":
            predict_rb(label_type, train_path,val_path,test_path,unlabeled_path, model_type) 

        # run rb_allen_cnn results
        elif model_type == "_rb_allen_cnn":
            # Note: allen nlp files are saved with prefix "predicted"
            col1 = "predict_" + str(label_type) + "_rb_baseline"
            col2 = "predicted_" + str(label_type) + "_allen_cnn"
            prediction_col = [col1, col2]      
            predict_sequential(label_type, result_path, prediction_col, model_type)   
        
        # run rb_word2vec_cnn results
        elif model_type == "_rb_word2vec_cnn":
            col1 = "predict_" + str(label_type) + "_rb_baseline"
            col2 = "predict_" + str(label_type) + "_word2vec_cnn"
            prediction_col = [col1, col2]      
            predict_sequential(label_type, result_path,prediction_col,model_type)  
        
        # run snorkel_rb_allen_cnn results
        elif model_type == "snorkel_rb_allen_cnn":
            # Note: allen nlp files are saved with prefix "predicted"
            col1 = "predicted_" + str(label_type) + "_allen_cnn"
            prediction_col = [col1]  
            # obtain snorkel model that yields highest accuracy on train, test, val
            eval_type = snorkel_best_eval[label_type]
            if eval_type == "label_model":
                model_type = model_type + "_lm"
            elif eval_type == "majority_vote":
                model_type = model_type + "_mv"

            predict_snorkel(result_path,prediction_col,eval_type,model_type) 

        # run snorkel_rb_word2vec_cnn
        elif model_type == "snorkel_rb_word2vec_cnn":
            col1 = "predict_" + str(model_type) + "_word2vec_cnn"
            prediction_col = [col1]  
            # obtain snorkel model that yields highest accuracy on train, test, val
            eval_type = snorkel_best_eval[label_type]
            if eval_type == "label_model":
                model_type = model_type + "_lm"
            elif eval_type == "majority_vote":
                model_type = model_type + "_mv"
            
            predict_snorkel(result_path,prediction_col,eval_type,model_type) 

        # run snorkel_rb_tfidf_word2vec_cnn
        elif model_type == "snorkel_rb_tfidf_word2vec_cnn":
            eval_type = snorkel_best_eval[label_type]
            if eval_type == "label_model":
                model_type = model_type + "_lm"
            elif eval_type == "majority_vote":
                model_type = model_type + "_mv"
            
            predict_snorkel(result_path,tfidf_predict_col,eval_type,model_type) 
        
        # run snorkel_rb_tfidf_allen_cnn
        elif model_type == "snorkel_rb_tfidf_allen_cnn":
            eval_type = snorkel_best_eval[label_type]
            if eval_type == "label_model":
                model_type = model_type + "_lm"
            elif eval_type == "majority_vote":
                model_type = model_type + "_mv"
            
            predict_snorkel(result_path,tfidf_predict_col,eval_type,model_type)

# RUN

# Data paths
train_path = "master_path/data/neurology_notes/processed_data/Final Splits/train_data.csv"
val_path = "master_path/data/neurology_notes/processed_data/Final Splits/val_data.csv"
test_path = "master_path/data/neurology_notes/processed_data/Final Splits/test_data.csv"
unlabeled_path = "master_path/data/neurology_notes/processed_data/Final Splits/unlabeled_data.csv"

# dictionary of result_paths
results_paths = {
    "edss_19" : "master_path/results/task1/edss_19_results.csv",
    "score_ambulation_subscore" : "master_path/results/task1/score_ambulation_subscore_results.csv",
    "score_bowel_bladder_subscore" : "master_path/results/task1/score_bowel_bladder_subscore_results.csv",
    "score_brain_stem_subscore" : "master_path/results/task1/score_brain_stem_subscore_results.csv",
    "score_cerebellar_subscore" : "master_path/results/task1/score_cerebellar_subscore_results.csv",
    "score_mental_subscore" : "master_path/results/task1/score_mental_subscore_results.csv",
    "score_pyramidal_subscore" : "master_path/results/task1/score_pyramidal_subscore_results.csv",
    "score_sensory_subscore" : "master_path/results/task1/score_sensory_subscore_results.csv",
    "score_visual_subscore" : "master_path/results/task1/score_visual_subscore_results.csv"
}

snorkel_best_eval = {
    "edss_19" : "label_model"
}

# sub_scores = ['score_ambulation_subscore','score_bowel_bladder_subscore', 'score_brain_stem_subscore', 'score_cerebellar_subscore', 
#     'score_mental_subscore', 'score_pyramidal_subscore', 'score_sensory_subscore', 'score_visual_subscore']
label_type = "score_visual_subscore" 
tfidf_models = [] #"log_reg_baseline", "linear_svc", "svc_rbf", "lda"
model_types = ["_rb_allen_cnn", "_rb_word2vec_cnn"] #"rb_baseline", "_rb_allen_cnn", "_rb_word2vec_cnn", "snorkel_rb_allen_cnn", "snorkel_rb_word2vec_cnn", "snorkel_rb_tfidf_word2vec_cnn", "snorkel_rb_tfidf_allen_cnn"
run_results(label_type, tfidf_models, model_types, train_path, val_path, test_path, unlabeled_path, results_paths, snorkel_best_eval)
