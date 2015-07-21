// shot gun
/*
Usage: ./shot_gun
	 -train 	trianing set in libsvm format
	 -test  test set in libsvm format
	 -a algorithm (1=lasso, 2=logitic regresion, 3 = find min lambda for all zero solution)
	 -t convergence threshold (default 1e-5)
	 -k solution path length (for lasso)
	 -i max_iter (default 100)
	 -n num_threads (default 2)
	 -lambda positive weight constant (default 0.000001)
	 -V verbose: 1=verbose, 0=quiet (default 0) 
*/
package main

import (
	"flag"
	"fmt"
//	"github.com/cswhjiang/machine-learning-tools-with-golang/logistic_regression/solver"
	//	"github.com/cswhjiang/machine-learning-tools-with-golang/utils/mathOperator"
	"github.com/cswhjiang/machine-learning-tools-with-golang/utils/readData"
	"log"
	//	"math"
	"os"
	"runtime/pprof"
	"time"
)

func main() {
	fmt.Printf("logistic_regression only supports binary classification for now\n")

	var train_file_name string
	var test_file_name string
	var lambda float64
	var alg_type int
	
	flag.StringVar(&train_file_name, "train", "", "training file (libsvm format)")
	flag.StringVar(&test_file_name, "test", "", "testing file (libsvm format)")
	flag.Float64Var(&lambda, "lambda", 0.000001, "positive weight constant (default 0.000001)")
	flag.IntVar(&alg_type, "a", 1, "algorithm (1=lasso, 2=logitic regresion, 3 = find min lambda for all zero solution)")
	
	var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

	if len(os.Args) <= 3 {
		fmt.Printf("Usage: \n")
		flag.PrintDefaults()
		os.Exit(0)
	}
	flag.Parse()

	//for profiling
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	start := time.Now()

	p, _ := readData.ReadData(train_file_name, true)
	elapsed := time.Since(start)
	fmt.Printf("took %s to read data \n", elapsed)

	p.PrintProblem()
	p.Lambda = float32(lambda)
	p.Epsilon = 0.001

	start = time.Now()
	var sigma float32
	var r float32
	sigma = 0.8
	r = 0.8
	
	if alg_type == 1{
		
	}else if alg_type == 2{
		
	}else if alg_type == 3{
		
	}else{
		//error
	}
	
	
	
//	solver.Solve_lr_new_glmnet_cdn(p, sigma, r) //newGLMNET, can be speeded up by using more tricks
	elapsed = time.Since(start)
	fmt.Printf("took %s to train \n", elapsed)

	start = time.Now()
	p_test, _ := readData.ReadData(train_file_name, true)
	p_test.X = p.X
	fmt.Printf("testing... ")
	loss_test := get_acc(p_test)
	fmt.Printf("acc: %f\n", loss_test)
	elapsed = time.Since(start)
	fmt.Printf("took %s to test \n", elapsed)
}