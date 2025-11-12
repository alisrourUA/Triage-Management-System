from triage_logic import assign_priority          
from console_admit import run_admission_session   
from nb_priority import nb_train, nb_predict      

if __name__ == "__main__":
    # Set use_ml=True to show NB “suggested priority” after enough cases
    queue = run_admission_session(
        assign_priority_fn=assign_priority,
        use_ml=True,
        nb_train_fn=nb_train,
        nb_predict_fn=nb_predict,
        n_min=10,           # wait for at least 8 cases before NB suggestions
    )

    #we print here a tiny summary 
    counts = {1:0, 2:0, 3:0, 4:0}
    for p in queue:
        counts[p["Priority"]] += 1
    print("\nSession summary:", counts)
