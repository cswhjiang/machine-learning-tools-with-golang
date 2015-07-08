// logistic_regression
package main

import (
	"fmt"
	"github.com/cswhjiang/machine-learning-tools-with-golang/utils/readData"
	"time"
)

type TestTruct struct {
	a int
}

func (t TestTruct) getA() int {
	return t.a
}
func main() {
	fmt.Printf("logistic_regression only support binary classification for now")
	var train_file_name string
	var test_file_name string
	var lambda float64
	flag.StringVar(&train_file_name, "train", "", "training file (libsvm format)")
	flag.StringVar(&test_file_name, "test", "", "testing file (libsvm format)")
	flag.Float64Var(&lambda, "lambda", 0.000001, "trade-off parameter")

	if len(os.Args) <= 3 {
		fmt.Printf("Usage: \n")
		flag.PrintDefaults()
		os.Exit(0)
	}
	flag.Parse()

	start := time.Now()

	p, _ := readData.ReadData(train_file_name, true)
	p.PrintProblem()
	p.Lambda = float32(lambda)
	p.Epsilon = 0.0001
	//	solve_lr_CD(p)
	elapsed := time.Since(start)

	fmt.Printf("took %s \n", elapsed)

	p_test, _ := readData.ReadData(train_file_name, true)
	p_test.X = p.X
	fmt.Printf("testing...")
	loss_test := get_acc_as_reg(p_test)
	fmt.Printf("testing error: %e", loss_test)
}
func solve_lr_CD(p *readData.Problem) {

}
