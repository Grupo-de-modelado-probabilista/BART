#!/bin/sh
source utils.sh

##This script will run both use case and run those with different size of trees
##To see the results you should do:
##snakeviz -s $FILE.prof
##mprof plot $FILE.dat
## pycallgraph ?? pip install pycallgraph   | pycallgraph graphviz -- ./mypythonscript.py

BASE="case_studies/"

##pymc_version=$(get_python_package_version pymc) || exit 1
bart_version=$(get_python_package_version pymc_bart) || exit 1
cores=4

timestamp=$(date +%s)
mkdir -p "$(pwd)/results/benchmark/$bart_version"
mkdir -p "$(pwd)/results/memory/$bart_version"


models=( biking coal friedman space_influenza ) ## biking coal friedman space_influenza
number_trees=(50 100 200)                     ## 50 100 200
particle=(20 40 60)                                 ## 20 40 60

csv="$(pwd)/results/result.csv"

mem="$(pwd)/results/mem.csv"
function to_csv() {
    printf "$timestamp,pymc-bart:$bart_version,$1,$2,$3,$4,$5,$6,$7\n" >> $csv
}

function to_mem(){
    m=$(echo $5 | sed 's/\./,/')
    printf "$timestamp;pymc-bart:$bart_version;$1;$2;$3;$m\n" >> $mem
}

function profile() {
  SECONDS=0
  output="results/benchmark/$bart_version/$1_$2_$3.prof"
  if [ -f $output ];then
    e_warning "profile run skipped already found output $output"
    return
  fi
  program="$BASE/bart_case_$1.py"
  python -m cProfile -o $output $program --trees $2 --particle $3 --cores $cores
  exit_status=$?
  elapsedSeconds=$SECONDS
  to_csv $FUNCNAME $exit_status $1 $2 $3 $elapsedSeconds $output
  if [ $exit_status -eq 0 ]; then
    e_success "profile | t:$2 p:$3 $output elapsed: $(textifyDuration $elapsedSeconds)"
  else
    e_error "profile | t:$2 p:$3 $output elapsed: $(textifyDuration $elapsedSeconds)"
    rm -f $output
  fi
}


function memory() {
  SECONDS=0
  output="results/memory/$bart_version/$1_$2_$3.dat"
  if [ -f $output ];then
    e_warning "memory run skipped already found output $output - $(mprof peak $output)"
    to_mem $1 $2 $3 $(mprof peak $output)
    return
  fi
  program="$BASE/bart_case_$1.py"
  mprof run --output $output -E -C $program --trees $2 --particle $3 --cores $cores
  #python -m memory_profiler $program --trees $2 --particle $3 -o $output --multiprocess || --multiprocess
  exit_status=$?
  elapsedSeconds=$SECONDS
  to_csv $FUNCNAME $exit_status $1 $2 $3 $elapsedSeconds $output
  if [ $exit_status -eq 0 ]; then
    e_success "memory | t:$2 p:$3 $output elapsed: $(textifyDuration $elapsedSeconds)"
    mprof peak $output
  else
    e_error "memory | t:$2 p:$3 $output elapsed: $(textifyDuration $elapsedSeconds)"
    rm -f $output
  fi
}

e_header "pymc: 5.0.1 pymc-bart: $bart_version" ##$pymc_version
for model in "${models[@]}"; do
  for ntree in ${number_trees[@]}; do
    for par in ${particle[@]}; do
      e_note "$model - Number of tree: $ntree - Number of particle: $par"
      profile "$model" "$ntree" "$par"
      memory "$model" "$ntree" "$par"
    done
  done
done
