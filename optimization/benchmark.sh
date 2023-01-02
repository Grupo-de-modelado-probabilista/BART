#!/bin/sh
source utils.sh

##This script will run both use case and run those with different size of trees
##To see the results you should do:
##snakeviz $FILE.prof
##mprof plot $FILE.dat
## pycallgraph ?? pip install pycallgraph   | pycallgraph graphviz -- ./mypythonscript.py

BASE="case_studies/"
bart_version=$(pip freeze | grep pymc-bart | sed "s/==/-/g")
timestamp=$(date +%s)
mkdir -p "$(pwd)/results/benchmark/$bart_version"
mkdir -p "$(pwd)/results/memory/$bart_version"


models=(coal biking space_influenza friedman) ## coal biking space_influenza friedman
number_trees=(50 100 200)                     ## 50 100 200
particle=(20 40 60)                                 ## 20 40 60

csv="$(pwd)/results/result.csv"
function to_csv() {
    printf "$bart_version,$timestamp,$1,$2,$3,$4,$5,$6\n" >> $csv
}
function profile() {
  SECONDS=0
  output="results/benchmark/$bart_version/$1_$2_$3.prof"
  if [ -f $output ];then
    e_warning "profile run skipped already found output $output"
    return
  fi
  program="$BASE/bart_case_$1.py"
  python -m cProfile -o $output $program --trees $2 --particle $3
  exit_status=$?
  elapsedSeconds=$SECONDS
  to_csv $FUNCNAME $exit_status $1 $2 $3 $elapsedSeconds
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
    e_warning "memory run skipped already found output $output"
    return
  fi
  program="$BASE/bart_case_$1.py"
  mprof run --output $output --include-children $program --trees $2 --particle $3
  exit_status=$?
  elapsedSeconds=$SECONDS
  to_csv $FUNCNAME $exit_status $1 $2 $3 $elapsedSeconds
  if [ $exit_status -eq 0 ]; then
    e_success "memory | t:$2 p:$3 $output elapsed: $(textifyDuration $elapsedSeconds)"
  else
    e_error "memory | t:$2 p:$3 $output elapsed: $(textifyDuration $elapsedSeconds)"
    rm -f $output
  fi
}

e_header "$bart_version"
for model in "${models[@]}"; do
  e_underline "$model"
  for ntree in ${number_trees[@]}; do
    for par in ${particle[@]}; do
      e_bold "Number of tree: $ntree - Number of particle: $par"
      profile "$model" "$ntree" "$par"
      memory "$model" "$ntree" "$par"
    done
  done
done
