// logistic_regression
package main

import (
	"flag"
	"fmt"
	"github.com/cswhjiang/machine-learning-tools-with-golang/logistic_regression/solver"
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
	flag.StringVar(&train_file_name, "train", "", "training file (libsvm format)")
	flag.StringVar(&test_file_name, "test", "", "testing file (libsvm format)")
	flag.Float64Var(&lambda, "lambda", 0.000001, "trade-off parameter")
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
	//	solver.Solve_lr_CD(p) // glmnet method
	var sigma float32
	var r float32
	sigma = 0.8
	r = 0.8
	solver.Solve_lr_new_glmnet_cdn(p, sigma, r) //newGLMNET
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

func get_acc(p *readData.Problem) float32 {
	var acc float32
	acc = 0
	for i := 0; i < p.L; i++ {
		pred := p.A_rows[i].Multiply_dense_array(p.X)
		//		pred := p.Ax[i]
		if float32(p.Labels[i])*pred > 0 {
			acc = acc + 1.0
		}
	}
	acc = acc / float32(p.L) * 100.0
	return acc
}
