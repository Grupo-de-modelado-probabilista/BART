#!/bin/sh
source utils.sh

BASE="../case_studies/"
bart_version=$(pip freeze | grep pymc-bart | sed "s/==/-/g")
timestamp=$(date +%s)
mkdir -p "$(pwd)/results/"

models=(coal biking space_influenza friedman)
number_trees=(50 100 200)
particle=(20 40 60)
number_iters=(500)

function profile() {
    # SECONDS=0
    output="results/$1_$2_$3.prof"
    if [ -f $output ];then
        e_warning "profile run skipped already found output $output"
        return
    fi
    program="$BASE/bart_case_$1.py"
    kernprof -lv -o "$output" $program --trees $2 --particle $3 --iters $4
    exit_status=$?
    # Elapsed seconds isn't useful b/c of the overhead of the profiler
    # elapsedSeconds=$SECONDS
    # to_csv $FUNCNAME $exit_status $1 $2 $3 $4 $elapsedSeconds
    if [ $exit_status -eq 0 ]; then
        e_success "profile | t:$2 p:$3 i:$4 $output elapsed: $(textifyDuration $elapsedSeconds)"
    else
        e_error "profile | t:$2 p:$3 i:$4 $output elapsed: $(textifyDuration $elapsedSeconds)"
        rm -f $output
    fi
}

e_header "$bart_version"
for model in "${models[@]}"; do
    e_underline "$model"
    for ntree in ${number_trees[@]}; do
        for par in ${particle[@]}; do
            for iter in ${number_iters[@]}; do
                e_bold "Number of tree: $ntree - Number of particle: $par - Number of iterations: $iter"
                profile "$model" "$ntree" "$par" "$iter"
            done
        done
    done
done